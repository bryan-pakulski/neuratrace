/**
 * connections.js — Controller for the model connection visualization.
 *
 * Derives dataflow edges (incl. skip connections) between layers from the
 * graph nodes (inputs/outputs tensor names), and drives the 3D rendering +
 * highlight lived in viewer3d.js. When a layer is clicked with connections
 * enabled, it brightens the layer's 1-hop neighbourhood (viewer3d) and fills
 * the #conn-info panel with one row per connected layer, fetching the
 * activation-based similarity (/api/frames/pair-strength) for each incident
 * edge so the user can see how strongly connected each neighbour is.
 *
 * Exports: { setNodes, init, setEnabled }
 */

import { viewer, sampleColormap } from './viewer3d.js';

let _enabled = false;          // connections toggle (gates click-to-highlight)
let _wired = false;            // init() has run
let _panel = null;             // #conn-info element
let _rowFetches = new Map();   // rowId -> { controller, fill slot }

/** Store graph nodes from /api/load-model (called by inference.js). */
function setNodes(nodes) {
  viewer.setGraphNodes(nodes);
}

/** Show/hide connections (called by the View-panel checkbox). */
function setEnabled(on) {
  _enabled = !!on;
  viewer.setConnectionsEnabled(_enabled);
  if (!_enabled) clearPanel();
}

/** Wire the layer-click → highlight + panel behaviour (idempotent). */
function init() {
  if (_wired) return;
  _wired = true;
  _panel = document.getElementById('conn-info');

  document.addEventListener('viewer:layer-click', (e) => {
    if (!_enabled) return;
    const { frameIdx } = e.detail || {};
    if (frameIdx == null) return;
    // viewer.highlightConnections toggles off on re-click of the same layer.
    viewer.highlightConnections(frameIdx);
    if (viewer._highlightActive && viewer._highlightActive()) {
      renderPanel(viewer.getConnectionInfo(frameIdx));
    } else {
      clearPanel();
    }
  });

  // Esc clears the highlight + panel.
  document.addEventListener('keydown', (e) => {
    if (!_enabled) return;
    if (e.key === 'Escape') {
      viewer.clearConnectionHighlight();
      clearPanel();
    }
  });
}

// ── Panel rendering ──────────────────────────────────────────────────────

function clearPanel() {
  if (!_panel) return;
  _panel.hidden = true;
  _panel.innerHTML = '';
  _rowFetches.clear();
}

function renderPanel(info) {
  if (!_panel) return;
  _rowFetches.clear();
  const { frameIdx, meta, outEdges, inEdges } = info;
  const all = outEdges.concat(inEdges);
  if (!all.length) {
    _panel.hidden = false;
    _panel.innerHTML = `<div class="conn-empty">${_escape(meta && meta.node_name || '')} has no captured layer-to-layer connections.</div>`;
    return;
  }
  // Sort: outgoing first, then by span desc (most "skip" first).
  all.sort((a, b) => {
    if (a.dir !== b.dir) return a.dir === 'out' ? -1 : 1;
    return b.span - a.span;
  });
  const frag = document.createDocumentFragment();
  const header = document.createElement('div');
  header.className = 'conn-header';
  header.textContent = `${_escape(meta && meta.node_name || `frame ${frameIdx}`)} — ${all.length} connection${all.length > 1 ? 's' : ''}`;
  frag.appendChild(header);
  all.forEach((edge, i) => frag.appendChild(renderRow(frameIdx, edge, i)));
  _panel.hidden = false;
  _panel.innerHTML = '';
  _panel.appendChild(frag);
  // Fetch similarities for each incident edge (bounded by the node's degree).
  all.forEach((edge, i) => fetchStrength(frameIdx, edge, i));
}

function renderRow(clickedIdx, edge, i) {
  const other = viewer.getFrameMeta(edge.other) || {};
  const row = document.createElement('div');
  row.className = 'conn-row';
  row.dataset.rowId = String(i);
  row.dataset.other = String(edge.other);
  row.title = 'Click to select & centre this connected layer in the 3D view';
  // Navigate: re-select `edge.other` (goes through the normal layer-click path
  // so highlight + panel + heatmap inspector + frame info all update) and tween
  // the orbit pivot onto it.
  row.addEventListener('click', (ev) => {
    ev.stopPropagation();
    const meta = viewer.getFrameMeta(edge.other);
    document.dispatchEvent(new CustomEvent('viewer:layer-click', {
      detail: { frameIdx: edge.other, meta },
    }));
    viewer.centerOnLayer(edge.other);
  });
  const badge = document.createElement('span');
  badge.className = `conn-dir conn-dir-${edge.dir}`;
  badge.textContent = edge.dir === 'out' ? '→' : '←';
  badge.title = edge.dir === 'out' ? 'this layer feeds the target' : 'target feeds this layer';
  const name = document.createElement('span');
  name.className = 'conn-name';
  name.textContent = other.node_name || `frame ${edge.other}`;
  name.title = other.node_name || '';
  const op = document.createElement('span');
  op.className = 'conn-op';
  op.textContent = other.op_type || '';
  const span = document.createElement('span');
  span.className = 'conn-span';
  span.textContent = edge.span > 1 ? `skip +${edge.span}` : `+${edge.span}`;
  span.title = `Spans ${edge.span} layer${edge.span > 1 ? 's' : ''} along the execution order.`;
  const meter = document.createElement('span');
  meter.className = 'conn-meter';
  const bar = document.createElement('span');
  bar.className = 'conn-bar';
  bar.style.width = '0%';
  const score = document.createElement('span');
  score.className = 'conn-score';
  score.textContent = '…';
  meter.appendChild(bar);
  meter.appendChild(score);
  row.appendChild(badge);
  row.appendChild(name);
  row.appendChild(op);
  row.appendChild(span);
  row.appendChild(meter);
  return row;
}

function fetchStrength(clickedIdx, edge, i) {
  const row = _panel && _panel.querySelector(`.conn-row[data-row-id="${i}"]`);
  if (!row) return;
  const ctrl = new AbortController();
  _rowFetches.set(i, ctrl);
  const a = clickedIdx, b = edge.other;
  fetch(`/api/frames/pair-strength?a=${a}&b=${b}`, { signal: ctrl.signal })
    .then((r) => r.json())
    .then((d) => {
      if (_rowFetches.get(i) !== ctrl) return;     // stale (panel rebuilt)
      fillRow(row, d);
    })
    .catch((e) => {
      if (e && e.name === 'AbortError') return;
      const score = row.querySelector('.conn-score');
      if (score) score.textContent = '—';
    });
}

function fillRow(row, d) {
  const sim = Math.max(0, Math.min(1, Number.isFinite(d.similarity) ? d.similarity : 0));
  const bar = row.querySelector('.conn-bar');
  const score = row.querySelector('.conn-score');
  if (bar) {
    bar.style.width = `${Math.round(sim * 100)}%`;
    const [r, g, b] = sampleColormap('connections', sim);
    bar.style.background = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
  }
  if (score) {
    score.textContent = sim.toFixed(2);
    score.title = `cosine ${d.cosine != null ? d.cosine.toFixed(3) : '—'} · energy ${d.energy_a != null ? d.energy_a.toFixed(2) : '—'}→${d.energy_b != null ? d.energy_b.toFixed(2) : '—'}`;
  }
}

function _escape(s) {
  return String(s || '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

export { setNodes, init, setEnabled };