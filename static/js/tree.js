/**
 * tree.js — Node Hierarchy Tree, Filtering & Detail Panel
 *
 * Renders the backend node_tree as an interactive hierarchy:
 *   - Path-based tree when '/' present in node names, op_type grouping otherwise
 *   - Expand/collapse for branch nodes
 *   - Click a leaf row to toggle selection (highlighted, no checkboxes)
 *   - Op_type badge + output shape per leaf
 *   - Data-driven filters: op-type chips, dormant, energy, exec-order range
 *   - Layers list + filters live in a collapsible section (closed by default)
 *   - Search input filters tree by node name
 *   - 2D heatmap detail panel for single-node selection
 *   - Attention multi-output renders separate heatmap per output
 *
 * Exports: { loadNodeTree }
 */

import { viewer, sampleColormap } from './viewer3d.js';

// ── DOM References ──────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const els = {
  search:        $('tree-search'),
  presets:       $('tree-presets'),
  opChips:       $('tree-optype-chips'),
  deadCheck:     $('tree-filter-dead'),
  energy:        $('tree-filter-energy'),
  energyVal:     $('tree-filter-energy-val'),
  rangeLo:       $('tree-filter-range-lo'),
  rangeHi:       $('tree-filter-range-hi'),
  rangeVal:      $('tree-filter-range-val'),
  layersToggle:  $('tree-layers-toggle'),
  layersBody:    $('tree-layers-body'),
  layersCount:   $('tree-layers-count'),
  empty:         $('tree-empty'),
  loading:       $('tree-loading'),
  error:         $('tree-error'),
  errorText:     $('tree-error-text'),
  container:     $('tree-container'),
  // detail panel created dynamically
};

// ── State ──────────────────────────────────────────────────────

let _nodeTree = null;     // {type, root} from backend
let _allNodes = [];        // flat list of {name, op_type, exec_order, outputs}
let _selectedNames = new Set();
let _searchQuery = '';
let _detailPanel = null;
// Set of node names that have an activation frame in the 3D cube, populated
// after inference (null until then). Nodes not in this set are metadata/shape
// tensors excluded from the cube — rendered greyed-out and non-selectable.
let _activeNames = null;

// Data-driven filters, applied as an intersection over the candidate set
// (manual checkbox picks, or all activation layers when none picked). Each is
// inactive in its default state. Dispatched as a `multi_node` set so the cube
// hides non-matching layers; no change needed to viewer.setFilter.
let _filters = {
  opTypes: new Set(),   // active op-type chips (union include)
  dead: false,          // hide dormant (is_uniform) layers
  range: null,          // [lo, hi] exec_order band, null = off
};
let _metaByName = null;     // node_name -> frame meta (built after inference)
let _maxEnergy = 1;          // max frame energy (kept for diagnostics; slider is 0-1)
let _execBounds = [0, 0];    // [minExec, maxExec] across frames
// Per-layer hotspot threshold (0-1). NOT a layer filter — it does not flow
// through _recomputeFilter. The slider drives viewer.setEnergyThreshold, which
// hides individual points within each layer whose per-layer-normalized
// brightness (grid[r][c]) is below the threshold, revealing each layer's
// hotspots regardless of absolute magnitude.
let _energyThreshold = 0;

// Activation op types for preset
const ACTIVATION_OPS = new Set(['Relu', 'Relu6', 'Sigmoid', 'Tanh', 'Gelu', 'Elu', 'LeakyRelu', 'Softplus', 'Softsign', 'HardSigmoid', 'HardSwish']);

// ── Helpers ────────────────────────────────────────────────────

/** Flatten tree into list of leaf node metadata. */
function _flattenNodes(root) {
  const out = [];
  function walk(node) {
    if (node.nodes && node.nodes.length > 0) {
      for (const n of node.nodes) out.push(n);
    }
    if (node.children && node.children.length > 0) {
      for (const c of node.children) walk(c);
    }
  }
  walk(root);
  return out;
}

/** Escape HTML text content. */
function _esc(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

// ── Tree Rendering ─────────────────────────────────────────────

/**
 * Recursively render tree nodes into container.
 * @param {Object} nodeTree - {type: "path"|"op_type", root: {...}}
 * @param {HTMLElement} container - DOM element to render into
 */
function renderTree(nodeTree, container) {
  container.innerHTML = '';
  // The root is a pure container (no leaves of its own); render its children
  // directly into the panel rather than wrapping them in a redundant
  // "root" branch header.
  const root = nodeTree.root;
  if (root.children && root.children.length > 0) {
    for (const child of root.children) _renderNode(child, container, 0);
  }
  if (root.nodes && root.nodes.length > 0) {
    for (const leaf of root.nodes) _renderLeaf(leaf, container, 0);
  }
  // With a leveled dataflow tree, a deep model can have hundreds of small
  // "depth N" groups. Rendering them all expanded is an overwhelming wall of
  // rows, so when there are many top-level groups, collapse every group except
  // the first few — the user expands what they want to inspect.
  const topBranches = container.querySelectorAll(':scope > .tree-node:not(.tree-leaf)');
  const KEEP_EXPANDED = 3;
  if (topBranches.length > 25) {
    topBranches.forEach((branch, i) => {
      if (i < KEEP_EXPANDED) return;
      const row = branch.querySelector(':scope > .tree-row');
      const kids = branch.querySelector(':scope > .tree-children');
      if (row && kids) {
        row.setAttribute('aria-expanded', 'false');
        const toggle = row.querySelector('.tree-toggle');
        if (toggle) toggle.textContent = '▶';
        kids.hidden = true;
      }
    });
  }
}

/**
 * Render a single tree node (branch or leaf) into parent.
 * @param {Object} node - {name, children?, nodes?, op_type?, count?}
 * @param {HTMLElement} parent - DOM parent
 * @param {number} depth - indentation level
 */
function _renderNode(node, parent, depth) {
  const hasChildren = node.children && node.children.length > 0;
  const hasLeaves = node.nodes && node.nodes.length > 0;
  // A node is a branch if it has child branches, OR it is an explicit group
  // (e.g. a dataflow "depth N" group with leaves but no child branches).
  const isBranch = hasChildren || (hasLeaves && node.group === true);

  // Branch node (path segment, op_type group, or dataflow depth group)
  if (isBranch || (hasLeaves && hasChildren)) {
    const branch = document.createElement('div');
    branch.className = 'tree-node';

    const row = document.createElement('div');
    row.className = 'tree-row';
    row.setAttribute('role', 'treeitem');
    row.setAttribute('aria-expanded', 'true');
    row.tabIndex = 0;

    const toggle = document.createElement('span');
    toggle.className = 'tree-toggle';
    toggle.textContent = '▼';

    row.appendChild(toggle);

    const label = document.createElement('span');
    label.className = 'tree-label';
    const count = node.count || (node.nodes ? node.nodes.length : 0);
    label.textContent = count > 0 ? `${node.name} (${count})` : node.name;
    row.appendChild(label);

    if (node.op_type) {
      const badge = document.createElement('span');
      badge.className = 'tree-badge';
      badge.textContent = node.op_type;
      row.appendChild(badge);
    }

    branch.appendChild(row);

    const childContainer = document.createElement('div');
    childContainer.className = 'tree-children';
    branch.appendChild(childContainer);

    // Render children
    if (hasChildren) {
      for (const child of node.children) {
        _renderNode(child, childContainer, depth + 1);
      }
    }
    // Render leaves
    if (hasLeaves) {
      for (const leaf of node.nodes) {
        _renderLeaf(leaf, childContainer, depth + 1);
      }
    }

    // Expand/collapse toggle.
    row.addEventListener('click', () => {
      const expanded = row.getAttribute('aria-expanded') === 'true';
      row.setAttribute('aria-expanded', String(!expanded));
      toggle.textContent = expanded ? '▶' : '▼';
      childContainer.hidden = expanded;
    });

    row.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        row.click();
      }
    });

    parent.appendChild(branch);
    return;
  }

  // Leaf-only node (no children, has nodes)
  if (hasLeaves && !hasChildren) {
    for (const leaf of node.nodes) {
      _renderLeaf(leaf, parent, depth);
    }
    return;
  }

  // Fallback: render as leaf if it has a name and op_type
  if (node.op_type && node.name) {
    _renderLeaf(node, parent, depth);
  }
}

/**
 * Render a leaf node (graph node) with name, badge, shape. Click the row to
 * toggle selection (highlighted via .tree-row-selected — no checkboxes).
 * @param {Object} leaf - {name, op_type, exec_order, outputs}
 * @param {HTMLElement} parent - DOM parent
 * @param {number} depth - indentation level
 */
function _renderLeaf(leaf, parent, depth) {
  const node = document.createElement('div');
  node.className = 'tree-node tree-leaf';
  node.dataset.nodeName = leaf.name;
  node.dataset.opType = leaf.op_type;
  node.dataset.execOrder = leaf.exec_order;

  const row = document.createElement('div');
  row.className = 'tree-row';
  row.setAttribute('role', 'treeitem');
  row.tabIndex = 0;
  row.dataset.nodeName = leaf.name;
  if (_selectedNames.has(leaf.name)) row.classList.add('tree-row-selected');

  // Spacer where a branch toggle would be (aligns leaves with branches).
  const spacer = document.createElement('span');
  spacer.className = 'tree-toggle';
  spacer.textContent = '';
  row.appendChild(spacer);

  // Node name
  const label = document.createElement('span');
  label.className = 'tree-label';
  label.textContent = leaf.name;
  row.appendChild(label);

  // Op_type badge
  const badge = document.createElement('span');
  badge.className = 'tree-badge';
  badge.textContent = leaf.op_type;
  row.appendChild(badge);

  // Output shape (from outputs array)
  if (leaf.outputs && leaf.outputs.length > 0) {
    const shape = document.createElement('span');
    shape.className = 'tree-shape';
    shape.textContent = leaf.outputs.length === 1
      ? `[${leaf.outputs[0]}]`
      : `${leaf.outputs.length} outputs`;
    row.appendChild(shape);
  }

  node.appendChild(row);
  parent.appendChild(node);

  // Apply search filter
  if (_searchQuery && !leaf.name.toLowerCase().includes(_searchQuery)) {
    node.style.display = 'none';
  }

  // Non-activation leaves (no 3D frame) are non-selectable: row clicks are
  // ignored and a hover tooltip explains why.
  const isNonAct = () => node.classList.contains('tree-leaf-nonact');

  // Row click → toggle selection + highlight + pivot to this layer.
  row.addEventListener('click', (e) => {
    if (isNonAct()) return;
    e.preventDefault();
    const selecting = !_selectedNames.has(leaf.name);
    if (selecting) _selectedNames.add(leaf.name);
    else _selectedNames.delete(leaf.name);
    row.classList.toggle('tree-row-selected', selecting);
    _dispatchFilter();
    _updateDetailPanel();
    if (selecting) viewer.focusPivot(leaf.name);
  });

  row.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      row.click();
    }
  });
}

/** Get tree type safely. */
function nodeTree_type() {
  return _nodeTree ? _nodeTree.type : 'path';
}

// ── Filter Dispatch ────────────────────────────────────────────

/** Build node_name → frame-meta index + energy/exec bounds from viewer.frames. */
function _buildMetaIndex() {
  _metaByName = new Map();
  let maxE = 0, minEx = Infinity, maxEx = -Infinity;
  for (const f of (viewer.frames || [])) {
    if (!f || !f.node_name) continue;
    _metaByName.set(f.node_name, f);
    const e = Math.max(Math.abs(f.raw_min || 0), Math.abs(f.raw_max || 0));
    if (e > maxE) maxE = e;
    const ex = f.exec_order ?? 0;
    if (ex < minEx) minEx = ex;
    if (ex > maxEx) maxEx = ex;
  }
  _maxEnergy = maxE || 1;
  _execBounds = [minEx === Infinity ? 0 : minEx, maxEx === -Infinity ? 0 : maxEx];
}

/** True when any data-driven layer filter is active (energy is per-point, not here). */
function _filtersActive() {
  return _filters.opTypes.size > 0 || _filters.dead || _filters.range != null;
}

/**
 * Recompute the visible layer set and dispatch it to the viewer.
 *
 * candidates = manual checkbox picks (if any) else all activation layers.
 * Each active filter is applied as a predicate (AND). The result is sent as a
 * `multi_node` set (hides the rest); when nothing is selected and no filter is
 * active, dispatch null (show all). An empty result still dispatches an empty
 * multi_node so filtered-out layers stay hidden (not "show all").
 */
function _recomputeFilter() {
  if (!_metaByName) _buildMetaIndex();
  const hasManual = _selectedNames.size > 0;
  if (!hasManual && !_filtersActive()) {
    viewer.setFilter(null);
    _syncRowHighlights();
    _updateDetailPanel();
    return;
  }
  const candidates = hasManual ? _selectedNames : (_activeNames || _metaByName.keys());
  const keep = [];
  for (const name of candidates) {
    const m = _metaByName.get(name);
    if (!m) continue;                 // not an activation frame
    if (_filters.opTypes.size && !_filters.opTypes.has(m.op_type)) continue;
    if (_filters.range && (m.exec_order < _filters.range[0] || m.exec_order > _filters.range[1])) continue;
    if (_filters.dead && m.is_uniform) continue;
    keep.push(name);
  }
  if (keep.length === 1) viewer.setFilter({ type: 'node', value: keep[0] });
  else viewer.setFilter({ type: 'multi_node', value: keep });
  _syncRowHighlights();
  _updateDetailPanel();
}

/** Apply current selection as filter to viewer. (Delegates to _recomputeFilter.) */
function _dispatchFilter() { _recomputeFilter(); }

// ── Preset Filters ─────────────────────────────────────────────

/** Apply a preset filter by name. (Only "all" remains: clear selection +
 *  filters and show every activation layer.) */
function _applyPreset(preset) {
  if (preset !== 'all') return;
  _resetFilters();
  _selectedNames.clear();
  _clearRowHighlights();
  viewer.setFilter(null);
  _closeDetailPanel();
}

/** Reset all data-driven filters to inactive + reflect in the controls. */
function _resetFilters() {
  _filters.opTypes.clear();
  _filters.dead = false;
  _filters.range = null;
  _energyThreshold = 0;
  if (els.deadCheck) els.deadCheck.checked = false;
  if (els.energy) { els.energy.value = '0'; _updateEnergyVal(); }
  if (viewer && typeof viewer.setEnergyThreshold === 'function') viewer.setEnergyThreshold(0);
  if (els.rangeLo && els.rangeHi) {
    els.rangeLo.value = String(_execBounds[0]);
    els.rangeHi.value = String(_execBounds[1]);
    _updateRangeVal();
  }
  _syncOpTypeChips();
}

// ── Data-driven filter controls ─────────────────────────────────

/** Build the op-type chip row from the op types present in the loaded frames. */
function _buildOpTypeChips() {
  if (!els.opChips) return;
  els.opChips.innerHTML = '';
  const types = viewer.getOpTypes ? viewer.getOpTypes() : [];
  for (const t of types) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'tree-optype-chip';
    chip.dataset.optype = t;
    chip.textContent = t;
    chip.title = `Show only ${t} layers.`;
    chip.addEventListener('click', () => {
      if (_filters.opTypes.has(t)) _filters.opTypes.delete(t);
      else _filters.opTypes.add(t);
      chip.classList.toggle('active', _filters.opTypes.has(t));
      _recomputeFilter();
    });
    els.opChips.appendChild(chip);
  }
}

/** Sync chip active states with _filters.opTypes. */
function _syncOpTypeChips() {
  if (!els.opChips) return;
  for (const chip of els.opChips.querySelectorAll('.tree-optype-chip')) {
    chip.classList.toggle('active', _filters.opTypes.has(chip.dataset.optype));
  }
}

/** Update the energy slider's readout. 0 = "off" (show all points). */
function _updateEnergyVal() {
  if (!els.energyVal) return;
  const v = _energyThreshold;
  els.energyVal.textContent = v > 0 ? '≥ ' + v.toFixed(2) : 'off';
}

/** Update the exec-order range readout ("all" when it spans the full extent). */
function _updateRangeVal() {
  if (!els.rangeVal) return;
  if (_filters.range == null) { els.rangeVal.textContent = 'all'; return; }
  const [lo, hi] = _filters.range;
  els.rangeVal.textContent = (lo === _execBounds[0] && hi === _execBounds[1]) ? 'all' : `${lo}–${hi}`;
}

/** Clear every leaf row's selected highlight. */
function _clearRowHighlights() {
  if (!els.container) return;
  const rows = els.container.querySelectorAll('.tree-row.tree-row-selected');
  for (const row of rows) row.classList.remove('tree-row-selected');
}

/** Re-apply the selected highlight on every leaf row to match _selectedNames. */
function _syncRowHighlights() {
  if (!els.container) return;
  const rows = els.container.querySelectorAll('.tree-leaf > .tree-row');
  for (const row of rows) {
    const name = row.dataset.nodeName;
    row.classList.toggle('tree-row-selected', name && _selectedNames.has(name));
  }
}

// ── Search Filtering ──────────────────────────────────────────

/** Filter tree nodes by search query. */
function _applySearch() {
  const query = _searchQuery.toLowerCase();
  const leaves = els.container.querySelectorAll('.tree-leaf');
  for (const leaf of leaves) {
    const name = (leaf.dataset.nodeName || '').toLowerCase();
    leaf.style.display = query && !name.includes(query) ? 'none' : '';
  }

  // Hide branches that have no visible leaves
  const branches = els.container.querySelectorAll('.tree-node:not(.tree-leaf)');
  for (const branch of branches) {
    const visibleLeaves = branch.querySelectorAll('.tree-leaf');
    let anyVisible = false;
    for (const vl of visibleLeaves) {
      if (vl.style.display !== 'none') { anyVisible = true; break; }
    }
    if (query && !anyVisible && visibleLeaves.length > 0) {
      branch.style.display = 'none';
    } else {
      branch.style.display = '';
    }
  }
}

// ── Detail Panel (2D Heatmap) ─────────────────────────────────

/** Show or update detail panel when exactly 1 node selected. */
async function _updateDetailPanel() {
  if (_selectedNames.size !== 1) {
    _closeDetailPanel();
    return;
  }

  const name = [..._selectedNames][0];
  const node = _allNodes.find(n => n.name === name);
  if (!node) return;

  _openDetailPanel(node);
}

/** Create detail panel DOM if not exists. */
function _ensureDetailPanel() {
  if (_detailPanel) return _detailPanel;

  _detailPanel = document.createElement('div');
  _detailPanel.className = 'tree-detail-panel';
  _detailPanel.id = 'tree-detail-panel';
  _detailPanel.hidden = true;

  const header = document.createElement('div');
  header.className = 'tree-detail-header';

  const title = document.createElement('span');
  title.className = 'tree-detail-title';
  title.id = 'tree-detail-title';
  header.appendChild(title);

  const closeBtn = document.createElement('button');
  closeBtn.className = 'btn btn-icon tree-detail-close';
  closeBtn.setAttribute('aria-label', 'Close detail panel');
  closeBtn.textContent = '✕';
  closeBtn.addEventListener('click', () => {
    _selectedNames.clear();
    _syncRowHighlights();
    _dispatchFilter();
    _closeDetailPanel();
  });
  header.appendChild(closeBtn);

  _detailPanel.appendChild(header);

  const body = document.createElement('div');
  body.className = 'tree-detail-body';
  body.id = 'tree-detail-body';
  _detailPanel.appendChild(body);

  // Insert into tree panel body, after tree-container
  const panelBody = els.container.parentElement;
  panelBody.appendChild(_detailPanel);

  return _detailPanel;
}

/** Open detail panel and render heatmaps for node outputs. */
async function _openDetailPanel(node) {
  const panel = _ensureDetailPanel();
  panel.hidden = false;

  const title = $('tree-detail-title');
  title.textContent = `${node.name} [${node.op_type}]`;

  const body = $('tree-detail-body');
  body.innerHTML = '';

  // Render one heatmap per output
  const outputs = node.outputs || [];
  const outputCount = outputs.length || 1;

  for (let i = 0; i < outputCount; i++) {
    const outputName = outputs[i] || `${node.name}_output_${i}`;
    const wrapper = document.createElement('div');
    wrapper.className = 'tree-detail-output';
    if (outputCount > 1) {
      const outLabel = document.createElement('div');
      outLabel.className = 'tree-detail-output-label';
      outLabel.textContent = outputName;
      wrapper.appendChild(outLabel);
    }

    const canvas = document.createElement('canvas');
    canvas.className = 'tree-detail-canvas';
    canvas.width = 200;
    canvas.height = 200;
    wrapper.appendChild(canvas);

    body.appendChild(wrapper);

    // Fetch frame data and render heatmap.
    // Frame index is the rank by exec_order, NOT exec_order itself (gaps +
    // multi-output nodes shift the rank), so look it up by node name.
    // Non-activation nodes (shape/constant metadata) are filtered out of the
    // cube and have no frame — show a note instead of the wrong layer.
    const fidx = viewer.frameIndexForNode(node.name);
    if (fidx < 0) {
      const note = document.createElement('div');
      note.className = 'tree-detail-error';
      note.textContent = 'Not an activation layer (metadata/shape tensor — excluded from the 3D view).';
      wrapper.replaceChild(note, canvas);
      continue;
    }
    try {
      const resp = await fetch(`/api/frame/${fidx}`);
      if (!resp.ok) {
        const err = document.createElement('div');
        err.className = 'tree-detail-error';
        err.textContent = 'Failed to load';
        wrapper.replaceChild(err, canvas);
        continue;
      }
      const data = await resp.json();
      _renderHeatmap(canvas, data.grid, data.shape);
    } catch (e) {
      const err = document.createElement('div');
      err.className = 'tree-detail-error';
      err.textContent = 'Error loading frame';
      wrapper.replaceChild(err, canvas);
    }
  }
}

/** Close detail panel. */
function _closeDetailPanel() {
  if (_detailPanel) {
    _detailPanel.hidden = true;
  }
}

/** Render brightness grid as 2D heatmap on canvas. */
function _renderHeatmap(canvas, grid, shape) {
  if (!grid || !shape || shape.length < 2) return;

  const ctx = canvas.getContext('2d');
  const rows = shape[0];
  const cols = shape[1];
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;

  // /api/frame ships the grid as a 2D nested list ([[..],[..]], from
  // numpy.tolist() on a 2D array), so index it as grid[r][c]. Fall back to
  // flat indexing for any legacy 1D grid. Indexing a nested grid flat would
  // hand sampleColormap a whole row (array) → map[NaN] → undefined → throw,
  // which surfaced as "Error loading frame" for every valid node.
  const nested = Array.isArray(grid[0]);
  const at = (r, c) => (nested ? (grid[r] ? grid[r][c] : 0) : grid[r * cols + c]) || 0;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = at(r, c);
      const [rr, gg, bb] = sampleColormap('viridis', val);
      ctx.fillStyle = `rgb(${Math.round(rr * 255)}, ${Math.round(gg * 255)}, ${Math.round(bb * 255)})`;
      ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
    }
  }
}

/** After inference, mark tree leaves whose node has no activation frame in
 *  the 3D cube (metadata/shape tensors) as greyed-out, non-selectable, and
 *  give them a hover tooltip. Leaves with a frame are (re-)enabled.
 *  Call this once `viewer.loadFrames` has populated the cube. */
function refreshActivationStates() {
  if (!_nodeTree) return;
  const frames = viewer.frames || [];
  const leaves = els.container.querySelectorAll('.tree-leaf');

  // No frames yet (no inference / nothing captured) → leave everything
  // selectable and clear any stale non-activation markings.
  if (frames.length === 0) {
    _activeNames = null;
    for (const leaf of leaves) {
      leaf.classList.remove('tree-leaf-nonact');
      leaf.removeAttribute('data-activation');
      leaf.title = '';
    }
    return;
  }

  _activeNames = new Set(
    frames.map(f => f && f.node_name).filter(Boolean)
  );

  // Build the frame-meta index + energy/exec bounds and seed the filter
  // controls (op-type chips, energy slider, exec-order range) for this model.
  _buildMetaIndex();
  if (els.energy) {
    // Per-layer hotspot threshold: 0-1 fraction of each layer's own max. Fixed
    // range (independent of the model's absolute energy scale, so massive
    // outliers no longer push every useful value to one end of the slider).
    els.energy.min = '0';
    els.energy.max = '1';
    els.energy.step = '0.01';
    els.energy.value = '0';
    _energyThreshold = 0;
    _updateEnergyVal();
  }
  if (els.rangeLo && els.rangeHi) {
    els.rangeLo.min = els.rangeHi.min = String(_execBounds[0]);
    els.rangeLo.max = els.rangeHi.max = String(_execBounds[1]);
    els.rangeLo.step = els.rangeHi.step = '1';
    if (!els.rangeLo.value || els.rangeLo.value === '0') els.rangeLo.value = String(_execBounds[0]);
    if (!els.rangeHi.value || els.rangeHi.value === '0') els.rangeHi.value = String(_execBounds[1]);
    _updateRangeVal();
  }
  _buildOpTypeChips();

  let changed = false;
  for (const leaf of leaves) {
    const name = leaf.dataset.nodeName;
    const isAct = _activeNames.has(name);
    leaf.classList.toggle('tree-leaf-nonact', !isAct);
    leaf.setAttribute('data-activation', isAct ? 'true' : 'false');
    leaf.title = isAct
      ? ''
      : 'Not an activation layer — metadata/shape tensor, excluded from the 3D view.';
    if (!isAct && _selectedNames.has(name)) {
      _selectedNames.delete(name);
      changed = true;
    }
  }
  if (changed) {
    _syncRowHighlights();
    _dispatchFilter();
    _updateDetailPanel();
  }
}

// ── Public API ─────────────────────────────────────────────────

/**
 * Load node tree data and render into tree panel.
 * Called by inference.js after /api/load-model returns node_tree.
 * @param {Object} nodeTree - {type: "path"|"op_type", root: {...}}
 */
function loadNodeTree(nodeTree) {
  _nodeTree = nodeTree;
  // Flatten, then sort by exec order so the flat view follows the model's
  // order of appearance regardless of tree grouping.
  _allNodes = _flattenNodes(nodeTree.root).sort(
    (a, b) => (a.exec_order ?? 0) - (b.exec_order ?? 0)
  );
  _selectedNames.clear();
  _searchQuery = '';
  _activeNames = null;   // reset; repopulated after inference
  _metaByName = null;    // rebuilt when frames arrive
  _resetFilters();       // clear stale chips/sliders from a previous model
  viewer.setFilter(null);

  // Hide empty/loading/error states
  if (els.empty) els.empty.hidden = true;
  if (els.loading) els.loading.hidden = true;
  if (els.error) els.error.hidden = true;

  // Show presets + container (both live inside the collapsible Layers body).
  if (els.presets) els.presets.hidden = false;
  if (els.container) els.container.hidden = false;

  // Render tree
  renderTree(nodeTree, els.container);

  // Layer count badge on the collapsible section header.
  if (els.layersCount) {
    const n = _allNodes.length;
    els.layersCount.textContent = n ? `${n} layers` : '';
  }
  // Collapse the Layers section by default after a model loads (the full layer
  // list can be long; expand to browse/filter).
  _setLayersOpen(false);

  // Close any detail panel
  _closeDetailPanel();
}

/** Expand/collapse the collapsible Layers section. */
function _setLayersOpen(open) {
  if (!els.layersToggle || !els.layersBody) return;
  els.layersToggle.setAttribute('aria-expanded', String(open));
  els.layersBody.hidden = !open;
}

// ── Event Wiring ──────────────────────────────────────────────

function _init() {
  // Search input
  if (els.search) {
    els.search.addEventListener('input', () => {
      _searchQuery = els.search.value.trim();
      _applySearch();
      // Auto-expand the Layers section so search results are visible.
      if (_searchQuery) _setLayersOpen(true);
    });
  }

  // Collapsible Layers section toggle.
  if (els.layersToggle) {
    els.layersToggle.addEventListener('click', () => {
      const open = els.layersToggle.getAttribute('aria-expanded') === 'true';
      _setLayersOpen(!open);
    });
  }

  // Preset buttons
  if (els.presets) {
    const buttons = els.presets.querySelectorAll('.tree-preset-btn');
    for (const btn of buttons) {
      btn.addEventListener('click', () => {
        _applyPreset(btn.dataset.preset);
      });
    }
  }

  // Dead/dormant toggle.
  if (els.deadCheck) {
    els.deadCheck.addEventListener('change', () => {
      _filters.dead = els.deadCheck.checked;
      _recomputeFilter();
    });
  }

  // Energy threshold slider (0 = off / show all points). This is a per-layer,
  // per-point mask, not a layer filter — it hides low-activation points within
  // each layer (revealing hotspots) without changing which layers are visible.
  if (els.energy) {
    els.energy.addEventListener('input', () => {
      _energyThreshold = Math.max(0, Math.min(1, parseFloat(els.energy.value) || 0));
      _updateEnergyVal();
      if (typeof viewer.setEnergyThreshold === 'function') viewer.setEnergyThreshold(_energyThreshold);
    });
  }

  // Exec-order range: two sliders, clamped so lo ≤ hi. "all" (full extent) = off.
  if (els.rangeLo && els.rangeHi) {
    const onRange = () => {
      let lo = parseInt(els.rangeLo.value, 10);
      let hi = parseInt(els.rangeHi.value, 10);
      if (Number.isNaN(lo)) lo = _execBounds[0];
      if (Number.isNaN(hi)) hi = _execBounds[1];
      if (lo > hi) { [lo, hi] = [hi, lo]; els.rangeLo.value = String(lo); els.rangeHi.value = String(hi); }
      _filters.range = (lo === _execBounds[0] && hi === _execBounds[1]) ? null : [lo, hi];
      _updateRangeVal();
      _recomputeFilter();
    };
    els.rangeLo.addEventListener('input', onRange);
    els.rangeHi.addEventListener('input', onRange);
  }
}

// ── Auto-init when DOM ready ──────────────────────────────────

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _init);
} else {
  _init();
}

// ── Exports ───────────────────────────────────────────────────

export { loadNodeTree, refreshActivationStates };