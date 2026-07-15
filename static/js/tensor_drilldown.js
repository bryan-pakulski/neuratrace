/**
 * tensor_drilldown.js — Multi-dimensional activation layer inspector.
 *
 * A drill-down opened from the View panel's per-layer inspector for genuinely
 * multi-dimensional layers (≥3 non-trivial axes after dropping a batch-1).
 * Shows the layer's internal structure the 3D cube collapses away:
 *
 *   1. Channel grid  — small multiples, one [rows × cols] heatmap per channel
 *      (VGG feature-map grids / NVIDIA FME "Channel View").
 *   2. PCA scatter   — 2D projection of the per-sample feature vectors
 *      (AttentionViz / Activation Atlas style), computed server-side via
 *      numpy SVD/eigh (no new dependency).
 *   3. Per-channel strip — one aggregate scalar per channel; a quick
 *      "which channels fire" summary (ActiVis).
 *
 * Each tensor axis is assigned a role (Spatial-rows / Spatial-cols / Channel /
 * Feature / reduce) via a heuristic on open, overridable by the user; changing
 * a role recomputes all three views. The full-resolution raw tensor is fetched
 * once (`/api/frame/{idx}/tensor`, binary float32, no downsampling) and cached
 * per frameIdx; the channel grid and strip are computed client-side from it
 * (pixel-exact at the layer's true dims), the scatter via `/pca`.
 *
 * Reuses the global `.modal-backdrop` / `.edit-modal*` CSS and the imperative
 * modal pattern from inference.js (the modal helpers there are module-local,
 * so small equivalents are reimplemented here).
 */

import { viewer, sampleColormap } from './viewer3d.js';

const ROLES = ['rows', 'cols', 'channel', 'feature', 'reduce'];
const ROLE_LABELS = {
  rows: 'Rows', cols: 'Cols', channel: 'Channel', feature: 'Feature', reduce: 'reduce',
};

const MAX_CHANNEL_CANVASES = 256;     // cap rendered small-multiples (note if more)
const CHANNEL_CANVAS_PX = 96;        // max CSS display side per channel thumbnail
const SCATTER_W = 660, SCATTER_H = 300;

// ── Per-frame tensor cache ────────────────────────────────────────────────
const _tensorCache = new Map();       // frameIdx → tdata

let _modal = null;                    // current .edit-modal element
let _backdrop = null;
let _state = null;                     // { frameIdx, meta, tdata, shape, roles, pcaToken }

// Exposed for headless verification probes (point/channel counts), not pixels.
window.__tensorDrilldown = null;

// Collapsible info blocks remember their open/closed state while the modal
// stays open (keyed by id). The roles explainer starts open; the three
// per-section infos start closed.
const _openInfo = new Set(['roles']);

// ── Info copy ──────────────────────────────────────────────────────────────

const ROLE_INFO = [
  ['Channel', 'the small-multiple index — each value along this axis becomes one channel tile (and one cell in the strip). Usually the smallest dim (e.g. 24, 256).'],
  ['Feature', 'the vector dimension D used for PCA — each sample point is projected from this axis. Usually a medium dim (e.g. 64, 128), a learned embedding per spatial location or token.'],
  ['Rows / Cols', 'the two spatial axes that define each channel’s [rows × cols] heatmap, and that PCA samples across. Usually the two largest dims (e.g. 320×320, 46×46).'],
  ['reduce', 'an axis averaged (mean) or maxed out of the way before rendering — e.g. an extra head or token dim not used as a spatial axis.'],
];

const SECTION_INFO = {
  'drilldown-channelgrid-wrap': 'One [rows × cols] heatmap per channel, after the Feature and reduce axes are averaged out. Bright = high activation. This is the feature-map view: it shows where across the rows×cols grid each individual channel fires. Click a tile to blow it up and pixel-peep (← → to flip, Save PNG to export).',
  'drilldown-scatter-wrap': 'Each point is one sample (a rows×cols location, or a token); its position is a 2-D PCA of that sample’s Feature vector. Points that land close together share a similar feature pattern. When a Channel axis is assigned, points are coloured by channel index. PCA collapses the high-D Feature axis down to the 2 axes of greatest variance — an overview of how samples/locations cluster.',
  'drilldown-strip-wrap': 'One scalar per channel — the channel’s aggregate activation (mean or max over every other axis), coloured on the global activation scale. The horizontal axis is channel index. A quick “which channels are firing” summary; use it to spot dead or dominant channels at a glance.',
};

// ── Public API ────────────────────────────────────────────────────────────

/** Is a layer multi-dimensional enough to warrant a drill-down?
 *  ≥3 axes of size >1 after dropping a leading batch-1 axis. */
function isMultiDim(meta) {
  const shape = meta && (meta.original_shape || meta.shape);
  if (!Array.isArray(shape) || shape.length < 3) return false;
  let s = shape.slice();
  if (s[0] === 1) s = s.slice(1);
  return s.filter((d) => d > 1).length >= 3;
}

/** Open the drill-down modal for a frame (layer-click must have supplied meta). */
async function openDrilldown(frameIdx, meta) {
  _closeExisting();
  _buildModal(frameIdx, meta);
  _state = {
    frameIdx, meta, tdata: null, shape: null, roles: null, pcaToken: 0,
    // PCA scatter state.
    pcaRes: null,       // last {x,y,label,n,d}
    pcaParams: null,    // _pcaParams() used for the current projection
    pcaBounds: null,    // {xmin,xmax,ymin,ymax} for screen↔data inversion
    pcaOutlierK: 2,     // sensitivity (dist > k·σ = outlier)
    pcaOutliersOnly: false,
    pcaOutliers: null,  // Uint8Array of outlier flags, indexed by point
    pcaSel: null,       // Set of selected point indices (brush), null = none
    hlChannels: null,   // Set of channel indices to highlight in grid + strip
  };
  _setLoading('Fetching tensor…');
  try {
    const tdata = await _fetchTensor(frameIdx);
    _state.tdata = tdata;
    _state.shape = tdata.shape;
    _state.roles = _autoRoles(tdata.shape, meta && meta.op_type);
    _renderAll();
  } catch (e) {
    _setError(`Failed to load tensor: ${e && e.message ? e.message : e}`);
  }
}

// ── Fetch ─────────────────────────────────────────────────────────────────

async function _fetchTensor(frameIdx) {
  if (_tensorCache.has(frameIdx)) return _tensorCache.get(frameIdx);
  const resp = await fetch(`/api/frame/${frameIdx}/tensor`);
  if (!resp.ok) {
    const j = await resp.json().catch(() => ({}));
    throw new Error((j && j.error) || `HTTP ${resp.status}`);
  }
  // The body is the FULL-resolution tensor as binary little-endian float32
  // (pixel-exact, no downsampling); shape/min/max/dtype travel in response
  // headers. Decode straight into a Float32Array view — host endianness
  // matches numpy's on x86/ARM, so no DataView swap is needed.
  const buf = await resp.arrayBuffer();
  const shapeHdr = resp.headers.get('X-Tensor-Shape');
  const shape = shapeHdr ? shapeHdr.split(',').map(Number) : null;
  if (!buf || !shape || shape.length === 0 || shape.some((d) => !Number.isFinite(d))) {
    throw new Error('malformed tensor response (missing X-Tensor-Shape or body)');
  }
  const flat = new Float32Array(buf);
  const expected = shape.reduce((a, b) => a * b, 1);
  if (flat.length !== expected) {
    throw new Error(`tensor byte length mismatch: got ${flat.length} floats, shape ${shape.join('×')} = ${expected}`);
  }
  const parseShape = (h) => (h ? h.split(',').filter(Boolean).map(Number) : []);
  const tdata = {
    flat,
    shape,
    originalShape: parseShape(resp.headers.get('X-Original-Shape')),
    dtype: resp.headers.get('X-Dtype') || 'float32',
    min: parseFloat(resp.headers.get('X-Tensor-Min')),
    max: parseFloat(resp.headers.get('X-Tensor-Max')),
  };
  _tensorCache.set(frameIdx, tdata);
  return tdata;
}

// ── Axis-role heuristic ──────────────────────────────────────────────────

function _autoRoles(shape, opType) {
  const nd = shape.length;
  const roles = new Array(nd).fill('reduce');
  const convish = /Conv|Pool|BatchNorm|Resize|Pad|Resize/i.test(opType || '');
  if (convish && nd === 3) {
    // [C, H, W] → channel = axis 0, rows = axis 1, cols = axis 2.
    roles[0] = 'channel'; roles[1] = 'rows'; roles[2] = 'cols';
    return roles;
  }
  // Generic: sort axes by size ascending (tie-break by index).
  const order = shape
    .map((s, i) => [s, i])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  // smallest → channel, largest → rows.
  roles[order[0][1]] = 'channel';
  roles[order[nd - 1][1]] = 'rows';
  const rest = order.slice(1, nd - 1).map((o) => o[1]); // middle axes, by size asc
  if (rest.length === 1) {
    roles[rest[0]] = 'feature';
  } else if (rest.length >= 2) {
    roles[rest[rest.length - 1]] = 'feature';   // largest middle → feature
    roles[rest[rest.length - 2]] = 'cols';     // next → cols
    for (let k = 0; k < rest.length - 2; k++) roles[rest[k]] = 'reduce';
  }
  // nd === 2 (channel+rows only): no feature/cols — leave as is.
  return roles;
}

function _axesWithRole(roles, role) {
  const out = [];
  for (let i = 0; i < roles.length; i++) if (roles[i] === role) out.push(i);
  return out;
}

// ── Modal scaffolding ─────────────────────────────────────────────────────

function _closeExisting() {
  _closePreview();
  if (_backdrop && _backdrop.parentNode) _backdrop.parentNode.removeChild(_backdrop);
  _modal = null;
  _backdrop = null;
}

function _buildModal(frameIdx, meta) {
  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';
  backdrop.addEventListener('mousedown', (e) => { if (e.target === backdrop) _closeExisting(); });

  const modal = document.createElement('div');
  modal.className = 'edit-modal drilldown-modal';
  modal.style.width = 'min(760px, calc(100vw - 24px))';

  const header = document.createElement('div');
  header.className = 'edit-modal-header';
  const title = document.createElement('span');
  title.className = 'edit-modal-title';
  const name = (meta && meta.node_name) || `frame ${frameIdx}`;
  const op = (meta && meta.op_type) || '';
  const shapeStr = (meta && (meta.original_shape || meta.shape) || []).join('×');
  title.textContent = `Drill-down: ${name}${op ? ` (${op})` : ''} [${shapeStr}]`;
  title.title = title.textContent;
  const closeBtn = document.createElement('button');
  closeBtn.className = 'btn btn-icon edit-modal-close';
  closeBtn.type = 'button';
  closeBtn.setAttribute('aria-label', 'Close');
  closeBtn.textContent = '✕';
  closeBtn.addEventListener('click', _closeExisting);
  header.appendChild(title);
  header.appendChild(closeBtn);
  modal.appendChild(header);

  const body = document.createElement('div');
  body.className = 'edit-modal-body drilldown-body';
  modal.appendChild(body);

  // Axis-role bar (populated once tensor shape is known).
  const axisBar = document.createElement('div');
  axisBar.className = 'drilldown-axisbar';
  const axisHint = document.createElement('div');
  axisHint.className = 'edit-modal-hint';
  axisHint.textContent = 'Assign each axis a role. Channel = small-multiple index, Feature = vector dim for PCA, Rows/Cols = spatial.';
  axisBar.appendChild(axisHint);

  // Collapsible “Roles explained” block: deeper per-role definitions.
  axisBar.appendChild(_infoToggle('roles', 'Roles explained',
    ROLE_INFO.map(([k, v]) => `<b>${k}</b> — ${v}`).join('<br>') +
    '<br><br>The inspector runs on the <i>full-resolution</i> tensor — no downsampling, so the per-channel heatmaps and strip are pixel-exact at the layer\'s true dimensions.',
    true));

  const axisRow = document.createElement('div');
  axisRow.className = 'drilldown-axisrow';
  const autoBtn = document.createElement('button');
  autoBtn.className = 'btn btn-sm btn-secondary drilldown-autobtn';
  autoBtn.type = 'button';
  autoBtn.textContent = 'Auto';
  autoBtn.title = 'Re-apply the axis-role heuristic from the shape and op type.';
  axisBar.appendChild(axisRow);
  axisBar.appendChild(autoBtn);
  body.appendChild(axisBar);

  // Sections.
  body.appendChild(_section('Channel grid', SECTION_INFO['drilldown-channelgrid-wrap'], 'drilldown-channelgrid-wrap', 'drilldown-channelgrid'));
  const scatterSection = _section('PCA projection (2D)', SECTION_INFO['drilldown-scatter-wrap'], 'drilldown-scatter-wrap', 'drilldown-scatter', 'canvas');
  _appendScatterControls(scatterSection.querySelector('.drilldown-section-head'));
  body.appendChild(scatterSection);
  body.appendChild(_section('Per-channel strip', SECTION_INFO['drilldown-strip-wrap'], 'drilldown-strip-wrap', 'drilldown-strip', 'canvas'));

  backdrop.appendChild(modal);
  document.body.appendChild(backdrop);
  _modal = modal;
  _backdrop = backdrop;

  // Wire the drag-brush on the scatter canvas once the modal is in the DOM.
  const scatterCanvas = modal.querySelector('.drilldown-scatter');
  if (scatterCanvas) _wireScatterBrush(scatterCanvas);

  autoBtn.addEventListener('click', () => {
    if (!_state || !_state.shape) return;
    _state.roles = _autoRoles(_state.shape, _state.meta && _state.meta.op_type);
    _renderAxisBar();
    _renderAll();
  });

  _centerModal(modal);
  _enableDrag(modal, header);
}

function _section(headingText, infoText, wrapClass, innerClass, innerTag) {
  const wrap = document.createElement('div');
  wrap.className = `drilldown-section ${wrapClass}`;
  // Header row: title + a `?` toggle for the info block (collapsed by default).
  const head = document.createElement('div');
  head.className = 'drilldown-section-head';
  const h = document.createElement('div');
  h.className = 'drilldown-section-title';
  h.textContent = headingText;
  head.appendChild(h);
  if (infoText) head.appendChild(_infoToggle(wrapClass, null, infoText, false));
  wrap.appendChild(head);
  // The channel grid is a <div> container holding many small canvases; the
  // strip and scatter are single <canvas> elements the renderers draw into
  // directly (they call getContext on this element), so they must be canvases.
  const inner = document.createElement(innerTag || 'div');
  inner.className = innerClass;
  wrap.appendChild(inner);
  return wrap;
}

/**
 * Build a label + `?` toggle + collapsible info paragraph.
 * `id` keys the open/closed state in `_openInfo` (so it persists while the
 * modal stays open). `defaultOpen` seeds the set on first sight.
 * `html` is set as innerHTML (supports <b>/<br>/<code>); pass plain text otherwise.
 */
function _infoToggle(id, label, html, defaultOpen) {
  if (defaultOpen && !_openInfo.has(id)) _openInfo.add(id);
  const wrap = document.createElement('div');
  wrap.className = 'drilldown-info-toggle';
  if (label) {
    const lab = document.createElement('span');
    lab.className = 'drilldown-info-label';
    lab.textContent = label;
    wrap.appendChild(lab);
  }
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'btn btn-icon drilldown-help-btn';
  btn.setAttribute('aria-label', 'Toggle explanation');
  btn.dataset.infoId = id;
  btn.textContent = '?';
  wrap.appendChild(btn);

  const body = document.createElement('div');
  body.className = 'drilldown-info';
  body.innerHTML = html;

  const apply = () => {
    const open = _openInfo.has(id);
    body.hidden = !open;
    btn.textContent = open ? '−' : '?';
    btn.setAttribute('aria-expanded', String(open));
  };
  btn.addEventListener('click', () => {
    if (_openInfo.has(id)) _openInfo.delete(id); else _openInfo.add(id);
    apply();
  });
  apply();
  wrap.appendChild(body);
  return wrap;
}

function _centerModal(modal) {
  const r = modal.getBoundingClientRect();
  modal.style.left = `${Math.max(8, (window.innerWidth - r.width) / 2)}px`;
  modal.style.top = `${Math.max(8, (window.innerHeight - r.height) / 2)}px`;
}

function _enableDrag(modal, handle) {
  handle.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.edit-modal-close')) return;
    handle.setPointerCapture(e.pointerId);
    const rect = modal.getBoundingClientRect();
    const sx = e.clientX, sy = e.clientY, sl = rect.left, st = rect.top;
    const onMove = (ev) => {
      const left = Math.max(0, Math.min(window.innerWidth - rect.width, sl + (ev.clientX - sx)));
      const top = Math.max(0, Math.min(window.innerHeight - rect.height, st + (ev.clientY - sy)));
      modal.style.left = `${left}px`;
      modal.style.top = `${top}px`;
    };
    const onUp = () => {
      handle.removeEventListener('pointermove', onMove);
      handle.removeEventListener('pointerup', onUp);
      handle.removeEventListener('pointercancel', onUp);
    };
    handle.addEventListener('pointermove', onMove);
    handle.addEventListener('pointerup', onUp);
    handle.addEventListener('pointercancel', onUp);
  });
}

function _setLoading(msg) {
  const body = _modal && _modal.querySelector('.drilldown-body');
  if (body) body.dataset.loading = msg;
  const grid = _modal && _modal.querySelector('.drilldown-channelgrid');
  if (grid) grid.innerHTML = `<div class="drilldown-loading">${msg}</div>`;
}

function _setError(msg) {
  const grid = _modal && _modal.querySelector('.drilldown-channelgrid');
  if (grid) grid.innerHTML = `<div class="drilldown-error">${msg}</div>`;
}

// ── Rendering ─────────────────────────────────────────────────────────────

function _brightnessMode() {
  const el = document.querySelector('input[name="brightness-mode"]:checked');
  return el && el.value === 'max' ? 'max' : 'mean';
}

function _colormap() {
  return (viewer && viewer._colormap) || 'viridis';
}

function _renderAll() {
  if (!_state || !_state.tdata) return;
  _renderAxisBar();
  _renderChannelGrid();
  _renderStrip();
  _renderScatter();
}

function _renderAxisBar() {
  const row = _modal.querySelector('.drilldown-axisrow');
  if (!row) return;
  row.innerHTML = '';
  const { shape, roles, meta } = _state;
  // The inspector works on the subsampled tensor (`shape`), but the user thinks
  // in the layer's true size. Align the original shape (which still carries the
  // leading batch-1) to the subsampled axes so each chip shows "orig → sampled".
  const orig = (meta && (meta.original_shape || meta.shape)) || null;
  const off = (orig && orig[0] === 1 && orig.length === shape.length + 1) ? 1 : 0;
  for (let i = 0; i < shape.length; i++) {
    const chip = document.createElement('div');
    chip.className = 'drilldown-axis';
    const lbl = document.createElement('span');
    lbl.className = 'drilldown-axis-name';
    const od = orig ? orig[i + off] : null;
    if (od != null && od !== shape[i]) {
      lbl.textContent = `axis${i}: ${od} → ${shape[i]}`;
      lbl.title = `original ${od} (inspector sees ${shape[i]})`;
    } else {
      lbl.textContent = `axis${i}: ${shape[i]}`;
    }
    const sel = document.createElement('select');
    sel.className = 'select drilldown-axis-role';
    sel.dataset.axis = String(i);
    for (const r of ROLES) {
      const opt = document.createElement('option');
      opt.value = r;
      opt.textContent = ROLE_LABELS[r];
      sel.appendChild(opt);
    }
    sel.value = roles[i];
    sel.addEventListener('change', () => {
      const ai = +sel.dataset.axis;
      _state.roles[ai] = sel.value;
      _renderChannelGrid();
      _renderStrip();
      _renderScatter();
    });
    chip.appendChild(lbl);
    chip.appendChild(sel);
    row.appendChild(chip);
  }
}

// ── Channel grid (small multiples) ────────────────────────────────────────

function _renderChannelGrid() {
  const grid = _modal.querySelector('.drilldown-channelgrid');
  if (!grid) return;
  grid.innerHTML = '';
  const { tdata, shape, roles } = _state;
  const chanAxis = roles.indexOf('channel');
  if (chanAxis < 0) {
    grid.innerHTML = '<div class="drilldown-hint">Assign a Channel axis to see per-channel maps.</div>';
    _publishDebug();
    return;
  }
  const mode = _brightnessMode();
  let maps;
  try {
    maps = _computeChannelMaps(tdata.flat, shape, roles, mode);
  } catch (e) {
    grid.innerHTML = `<div class="drilldown-error">${e.message}</div>`;
    _publishDebug();
    return;
  }
  const { channels, rows, cols, data, perChannel } = maps;
  const cmap = _colormap();
  // Stash for the click-to-preview lightbox (re-renders any channel at any zoom).
  _state.channelMaps = maps;

  const shown = Math.min(channels, MAX_CHANNEL_CANVASES);
  for (let c = 0; c < shown; c++) {
    const canvas = document.createElement('canvas');
    canvas.className = 'drilldown-canvas';
    canvas.width = Math.max(1, cols);
    canvas.height = Math.max(1, rows);
    canvas.title = `channel ${c} — click to blow up`;
    canvas.dataset.channel = String(c);
    _drawChannel(canvas, c, maps, cmap);
    const cap = document.createElement('div');
    cap.className = 'drilldown-canvas-cap';
    cap.textContent = String(c);
    const cell = document.createElement('div');
    cell.className = 'drilldown-channelcell';
    if (_state.hlChannels && _state.hlChannels.has(c)) cell.classList.add('drilldown-channelcell-hl');
    cell.appendChild(canvas);
    cell.appendChild(cap);
    // Click a tile → blown-up pixel-peep preview (with PNG export + channel flip).
    cell.addEventListener('click', () => _openChannelPreview(c, maps));
    grid.appendChild(cell);
  }
  if (channels > shown) {
    const note = document.createElement('div');
    note.className = 'drilldown-hint';
    note.textContent = `Showing ${shown} of ${channels} channels.`;
    grid.appendChild(note);
  }
  _publishDebug();
}

/** Render channel `c` of `maps` into `canvas` (per-channel min/max normalized). */
function _drawChannel(canvas, c, maps, cmap) {
  const { rows, cols, data, perChannel } = maps;
  const off = c * perChannel;
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < perChannel; i++) {
    const v = data[off + i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const range = mx - mn;
  canvas.width = Math.max(1, cols);
  canvas.height = Math.max(1, rows);
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(canvas.width, canvas.height);
  for (let i = 0; i < perChannel; i++) {
    const t = range > 1e-12 ? (data[off + i] - mn) / range : 0.5;
    const [r, g, b] = sampleColormap(cmap, t);
    const o = i * 4;
    img.data[o] = r * 255; img.data[o + 1] = g * 255; img.data[o + 2] = b * 255; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// ── Channel preview lightbox (blow-up + pixel-peep + PNG export) ───────────

let _preview = null;   // { backdrop, canvas, label, counter, prevBtn, nextBtn, zoomSel, c, channels, maps, name }

const PREVIEW_ZOOMS = ['Fit', '2×', '4×', '8×', '16×'];

function _openChannelPreview(c, maps) {
  _closePreview();
  if (!maps) maps = _state && _state.channelMaps;
  if (!maps) return;
  const channels = maps.channels;
  if (channels <= 0) return;
  const name = (_state && _state.meta && _state.meta.node_name) || `frame ${_state && _state.frameIdx}`;

  const backdrop = document.createElement('div');
  backdrop.className = 'channel-preview-backdrop';
  backdrop.addEventListener('mousedown', (e) => { if (e.target === backdrop) _closePreview(); });

  const panel = document.createElement('div');
  panel.className = 'channel-preview-panel';

  // Header
  const header = document.createElement('div');
  header.className = 'channel-preview-header';
  const label = document.createElement('span');
  label.className = 'channel-preview-label';
  const spacer = document.createElement('div');
  spacer.style.flex = '1';
  const zoomSel = document.createElement('select');
  zoomSel.className = 'select channel-preview-zoom';
  zoomSel.title = 'Zoom level (nearest-neighbour for pixel peeping)';
  for (const z of PREVIEW_ZOOMS) {
    const o = document.createElement('option');
    o.value = z; o.textContent = z; zoomSel.appendChild(o);
  }
  zoomSel.value = 'Fit';
  const saveBtn = document.createElement('button');
  saveBtn.className = 'btn btn-sm btn-secondary';
  saveBtn.type = 'button';
  saveBtn.textContent = 'Save PNG';
  saveBtn.title = 'Download this channel as a PNG (upscaled, crisp pixels).';
  const closeBtn = document.createElement('button');
  closeBtn.className = 'btn btn-icon edit-modal-close';
  closeBtn.type = 'button';
  closeBtn.setAttribute('aria-label', 'Close');
  closeBtn.textContent = '✕';
  header.appendChild(label);
  header.appendChild(spacer);
  header.appendChild(zoomSel);
  header.appendChild(saveBtn);
  header.appendChild(closeBtn);
  panel.appendChild(header);

  // Scrollable image stage
  const stage = document.createElement('div');
  stage.className = 'channel-preview-stage';
  const canvas = document.createElement('canvas');
  canvas.className = 'channel-preview-canvas';
  stage.appendChild(canvas);
  panel.appendChild(stage);

  // Nav row
  const nav = document.createElement('div');
  nav.className = 'channel-preview-nav';
  const prevBtn = document.createElement('button');
  prevBtn.className = 'btn btn-sm btn-secondary';
  prevBtn.type = 'button';
  prevBtn.textContent = '◀ Prev';
  const counter = document.createElement('span');
  counter.className = 'channel-preview-counter';
  const nextBtn = document.createElement('button');
  nextBtn.className = 'btn btn-sm btn-secondary';
  nextBtn.type = 'button';
  nextBtn.textContent = 'Next ▶';
  const hint = document.createElement('span');
  hint.className = 'channel-preview-hint';
  hint.textContent = '← → flip · Esc close';
  nav.appendChild(prevBtn);
  nav.appendChild(counter);
  nav.appendChild(nextBtn);
  nav.appendChild(hint);
  panel.appendChild(nav);

  backdrop.appendChild(panel);
  document.body.appendChild(backdrop);

  _preview = { backdrop, canvas, label, counter, prevBtn, nextBtn, zoomSel,
               c: Math.max(0, Math.min(c, channels - 1)), channels, maps, name };

  _renderPreviewChannel(_preview.c);
  _applyPreviewZoom();

  // Wire events.
  closeBtn.addEventListener('click', _closePreview);
  saveBtn.addEventListener('click', () => _saveChannelPNG(_preview.c));
  prevBtn.addEventListener('click', () => _renderPreviewChannel(_preview.c - 1));
  nextBtn.addEventListener('click', () => _renderPreviewChannel(_preview.c + 1));
  zoomSel.addEventListener('change', _applyPreviewZoom);
  backdrop.addEventListener('keydown', _previewKeydown);
  // Key handlers live on window so arrows work without focusing a button.
  window.addEventListener('keydown', _previewKeydown);
  // Stop the drilldown modal's pointer drag from starting when interacting here.
  panel.addEventListener('pointerdown', (e) => e.stopPropagation());

  // Focus the panel for Esc without scrolling the grid behind.
  backdrop.tabIndex = -1;
  backdrop.focus();
}

function _renderPreviewChannel(c) {
  if (!_preview) return;
  const p = _preview;
  p.c = Math.max(0, Math.min(c, p.channels - 1));
  _drawChannel(p.canvas, p.c, p.maps, _colormap());
  p.label.textContent = `Channel ${p.c} / ${p.channels - 1} — ${p.name}`;
  p.counter.textContent = `${p.c + 1} / ${p.channels}`;
  p.prevBtn.disabled = p.c <= 0;
  p.nextBtn.disabled = p.c >= p.channels - 1;
  _applyPreviewZoom();
}

function _applyPreviewZoom() {
  if (!_preview) return;
  const p = _preview;
  const z = p.zoomSel.value;
  const { rows, cols } = p.maps;
  const canvas = p.canvas;
  if (z === 'Fit') {
    canvas.style.width = '';
    canvas.style.height = '';
    canvas.style.maxWidth = '100%';
    canvas.style.maxHeight = '100%';
  } else {
    const n = parseInt(z, 10) || 2;
    canvas.style.maxWidth = 'none';
    canvas.style.maxHeight = 'none';
    canvas.style.width = `${Math.max(1, cols) * n}px`;
    canvas.style.height = `${Math.max(1, rows) * n}px`;
  }
}

function _previewKeydown(e) {
  if (!_preview) return;
  switch (e.key) {
    case 'ArrowLeft': e.preventDefault(); _renderPreviewChannel(_preview.c - 1); break;
    case 'ArrowRight': e.preventDefault(); _renderPreviewChannel(_preview.c + 1); break;
    case 'Escape': e.preventDefault(); _closePreview(); break;
  }
}

function _closePreview() {
  if (!_preview) return;
  window.removeEventListener('keydown', _previewKeydown);
  if (_preview.backdrop.parentNode) _preview.backdrop.parentNode.removeChild(_preview.backdrop);
  _preview = null;
}

/** Export channel `c` as an upscaled (nearest-neighbour) PNG download. */
function _saveChannelPNG(c) {
  if (!_preview) return;
  const p = _preview;
  const { rows, cols } = p.maps;
  // Render the channel at native res into a scratch canvas, then upscale with
  // smoothing disabled so the saved PNG preserves crisp blocky pixels.
  const src = document.createElement('canvas');
  _drawChannel(src, c, p.maps, _colormap());
  const MAX_SIDE = 1024;
  const side = Math.max(cols, rows) || 1;
  const scale = Math.max(1, Math.ceil(MAX_SIDE / side));
  const out = document.createElement('canvas');
  out.width = Math.max(1, cols) * scale;
  out.height = Math.max(1, rows) * scale;
  const ctx = out.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(src, 0, 0, out.width, out.height);
  const url = out.toDataURL('image/png');
  const a = document.createElement('a');
  const safeName = String(p.name).replace(/[^A-Za-z0-9._-]+/g, '_').slice(0, 48) || 'frame';
  a.href = url;
  a.download = `${safeName}_ch${c}.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// ── Per-channel strip ─────────────────────────────────────────────────────

function _renderStrip() {
  const canvas = _modal.querySelector('.drilldown-strip');
  if (!canvas) return;
  const { tdata, shape, roles } = _state;
  const chanAxis = roles.indexOf('channel');
  if (chanAxis < 0) {
    canvas.hidden = true; canvas.width = 0; canvas.height = 0;
    _publishDebug();
    return;
  }
  canvas.hidden = false;
  const strip = _reduceStrip(tdata.flat, shape, chanAxis, _brightnessMode());
  const C = strip.length;
  const hl = _state.hlChannels;
  // Two rows: row 0 = activation colour, row 1 = bright accent marker for any
  // highlighted channel (from the PCA selection/outliers). Displayed at 28px
  // tall with image-rendering:pixelated, so each row is a crisp band.
  canvas.width = C; canvas.height = 2;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(C, 2);
  const cmap = _colormap();
  const tmin = tdata.min, tmax = tdata.max, tr = tmax - tmin;
  for (let c = 0; c < C; c++) {
    const t = tr > 1e-12 ? (strip[c] - tmin) / tr : 0.5;
    const [r, g, b] = sampleColormap(cmap, Math.max(0, Math.min(1, t)));
    const o = c * 4;
    img.data[o] = r * 255; img.data[o + 1] = g * 255; img.data[o + 2] = b * 255; img.data[o + 3] = 255;
    if (hl && hl.has(c)) {
      const m = (C + c) * 4;   // row 1
      img.data[m] = 95; img.data[m + 1] = 212; img.data[m + 2] = 255; img.data[m + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  _publishDebug({ strip });
}

// ── PCA scatter ───────────────────────────────────────────────────────────

function _pcaParams(roles) {
  const featAxis = roles.indexOf('feature');
  const chanAxis = roles.indexOf('channel');
  const rowAxes = _axesWithRole(roles, 'rows');
  const colAxes = _axesWithRole(roles, 'cols');
  let featureAxis, channelParam, sampleAxes;
  if (featAxis >= 0) {
    featureAxis = featAxis;
    sampleAxes = [...rowAxes, ...colAxes];
    // Channel (if assigned) is passed separately: the backend folds it into
    // the sample rows and returns a per-point channel label for colouring.
    channelParam = chanAxis;             // -1 if no channel role
  } else if (chanAxis >= 0) {
    featureAxis = chanAxis;               // channel doubles as the vector dim
    sampleAxes = [...rowAxes, ...colAxes];
    channelParam = -1;
  } else {
    return null;                          // nothing to project
  }
  if (!sampleAxes.length) return null;
  sampleAxes = sampleAxes.slice().sort((a, b) => a - b);  // match backend sort
  return { featureAxis, channelParam, sampleAxes };
}

function _renderScatter() {
  const canvas = _modal.querySelector('.drilldown-scatter');
  if (!canvas) return;
  canvas.width = SCATTER_W; canvas.height = SCATTER_H;
  const params = _pcaParams(_state.roles);
  _state.pcaParams = params;
  if (!params) {
    _state.pcaRes = null; _state.pcaOutliers = null; _state.pcaSel = null;
    _updateHlChannels();
    _scatterMessage(canvas, 'Assign a Feature (or Channel) axis and ≥1 Rows/Cols axis to project.');
    _publishDebug();
    return;
  }
  _scatterMessage(canvas, 'Projecting…');
  const token = ++_state.pcaToken;
  const q = new URLSearchParams({
    feature_axis: String(params.featureAxis),
    sample_axes: params.sampleAxes.join(','),
    channel_axis: String(params.channelParam),
  });
  fetch(`/api/frame/${_state.frameIdx}/pca?${q}`)
    .then((r) => r.json())
    .then((res) => {
      if (_state.pcaToken !== token) return;     // a newer request superseded this
      _state.pcaRes = res;
      _state.pcaSel = null;          // a new projection invalidates the brush
      _computeOutliers();
      _drawScatter(canvas);
      _updateHlChannels();
      _renderChannelGrid();
      _renderStrip();
      _publishScatterSummary();
    })
    .catch((e) => {
      if (_state.pcaToken !== token) return;
      _scatterMessage(canvas, `Projection failed: ${e && e.message ? e.message : e}`);
      _publishDebug();
    });
}

/** Flag outliers: points whose distance from the PCA centroid exceeds k·σ. */
function _computeOutliers() {
  const res = _state.pcaRes;
  if (!res || !res.x || !res.x.length) { _state.pcaOutliers = null; return; }
  const n = res.x.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += res.x[i]; my += res.y[i]; }
  mx /= n; my /= n;
  const dist = new Float64Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) {
    const dx = res.x[i] - mx, dy = res.y[i] - my;
    const d = Math.sqrt(dx * dx + dy * dy);
    dist[i] = d; mean += d;
  }
  mean /= n;
  let varc = 0;
  for (let i = 0; i < n; i++) { const dd = dist[i] - mean; varc += dd * dd; }
  const sigma = Math.sqrt(varc / Math.max(1, n - 1));
  const k = _state.pcaOutlierK;
  const thr = mean + k * sigma;
  const flags = new Uint8Array(n);
  if (sigma > 1e-12) {
    for (let i = 0; i < n; i++) flags[i] = dist[i] > thr ? 1 : 0;
  }
  _state.pcaOutliers = flags;
}

/** Is point i currently in the foreground (selected or an outlier)? */
function _pointHi(i) {
  if (_state.pcaSel && _state.pcaSel.has(i)) return true;
  if (!_state.pcaSel && _state.pcaOutliers && _state.pcaOutliers[i]) return true;
  return false;
}

function _drawScatter(canvas) {
  if (!canvas) canvas = _modal.querySelector('.drilldown-scatter');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const res = _state.pcaRes;
  if (!res || !res.x || !res.x.length) {
    _scatterMessage(canvas, 'No projection (insufficient samples/feature dims).');
    return;
  }
  const params = _state.pcaParams || {};
  const n = res.x.length;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = res.x[i], y = res.y[i];
    if (x < xmin) xmin = x; if (x > xmax) xmax = x;
    if (y < ymin) ymin = y; if (y > ymax) ymax = y;
  }
  const pad = 16;
  const xr = (xmax - xmin) || 1, yr = (ymax - ymin) || 1;
  const W = canvas.width, H = canvas.height;
  const sx = (x) => pad + ((x - xmin) / xr) * (W - 2 * pad);
  const sy = (y) => H - (pad + ((y - ymin) / yr) * (H - 2 * pad));
  // Inverse (screen px → data) for brush hit-testing.
  _state.pcaBounds = { xmin, xmax, ymin, ymax, pad, W, H, xr, yr };
  const cmap = _colormap();
  const hasLabel = res.label != null;
  const nChan = hasLabel ? Math.max(1, params.channelParam >= 0
    ? _state.shape[params.channelParam] : 1) : 1;
  ctx.fillStyle = 'rgba(0,0,0,0.05)';
  ctx.fillRect(0, 0, W, H);

  const onlyOutliers = _state.pcaOutliersOnly;
  const anySel = !!_state.pcaSel;
  for (let i = 0; i < n; i++) {
    const hi = _pointHi(i);
    if (onlyOutliers && !hi && !(_state.pcaOutliers && _state.pcaOutliers[i])) continue;
    let r, g, b;
    if (hasLabel) {
      const t = nChan > 1 ? res.label[i] / (nChan - 1) : 0.5;
      [r, g, b] = sampleColormap(cmap, t);
    } else {
      [r, g, b] = sampleColormap(cmap, 0.7);   // fixed accent when no channel label
    }
    // Foreground (selected/outlier) bright + larger; inliers dim when a
    // selection exists or outlier-only mode is on.
    const fg = hi || (!onlyOutliers && !anySel);
    const alpha = fg ? 0.85 : 0.12;
    const rad = fg ? 2.6 : 1.6;
    ctx.fillStyle = `rgba(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0},${alpha})`;
    ctx.beginPath();
    ctx.arc(sx(res.x[i]), sy(res.y[i]), rad, 0, Math.PI * 2);
    ctx.fill();
  }

  // Redraw the current brush rectangle on top, if dragging.
  if (_brush && _brush.active && _brush.rect) _drawBrushRect(ctx, _brush.rect);
}

// ── Brush selection (drag a rectangle over the scatter) ────────────────────

let _brush = null;   // { active, x0, y0, rect } on the scatter canvas

function _wireScatterBrush(canvas) {
  _brush = { active: false, x0: 0, y0: 0, rect: null };
  const rectOf = (e) => {
    const r = canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) * (canvas.width / r.width);
    const y = (e.clientY - r.top) * (canvas.height / r.height);
    return { x, y };
  };
  canvas.addEventListener('pointerdown', (e) => {
    if (!_state.pcaRes) return;
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    const p = rectOf(e);
    _brush.active = true; _brush.x0 = p.x; _brush.y0 = p.y;
    _brush.rect = { x: p.x, y: p.y, w: 0, h: 0 };
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!_brush.active) return;
    const p = rectOf(e);
    _brush.rect = {
      x: Math.min(_brush.x0, p.x), y: Math.min(_brush.y0, p.y),
      w: Math.abs(p.x - _brush.x0), h: Math.abs(p.y - _brush.y0),
    };
    _drawScatter(canvas);
  });
  const end = (e) => {
    if (!_brush.active) return;
    _brush.active = false;
    const rect = _brush.rect;
    _brush.rect = null;
    if (rect && rect.w > 3 && rect.h > 3) _applyBrush(rect);
    else { _state.pcaSel = null; _drawScatter(canvas); _updateHlChannels(); _renderChannelGrid(); _renderStrip(); }
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  // Esc clears any brush selection while the drill-down is open.
  canvas.tabIndex = 0;
  canvas.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || !_state.pcaSel) return;
    _state.pcaSel = null;
    _drawScatter(canvas);
    _updateHlChannels();
    _renderChannelGrid();
    _renderStrip();
    _publishScatterSummary();
  });
}

/** Convert a screen-space rect to data space and select the points inside. */
function _applyBrush(rect) {
  const b = _state.pcaBounds;
  if (!b) return;
  const dToX = (px) => b.xmin + ((px - b.pad) / (b.W - 2 * b.pad)) * b.xr;
  // y is inverted on screen
  const dToY = (py) => b.ymin + ((b.H - b.pad - py) / (b.H - 2 * b.pad)) * b.yr;
  const x0 = Math.min(dToX(rect.x), dToX(rect.x + rect.w));
  const x1 = Math.max(dToX(rect.x), dToX(rect.x + rect.w));
  const y0 = Math.min(dToY(rect.y), dToY(rect.y + rect.h));
  const y1 = Math.max(dToY(rect.y), dToY(rect.y + rect.h));
  const res = _state.pcaRes;
  const sel = new Set();
  for (let i = 0; i < res.x.length; i++) {
    if (res.x[i] >= x0 && res.x[i] <= x1 && res.y[i] >= y0 && res.y[i] <= y1) sel.add(i);
  }
  _state.pcaSel = sel.size ? sel : null;
  _drawScatter();
  _updateHlChannels();
  _renderChannelGrid();
  _renderStrip();
  _publishScatterSummary();
}

function _drawBrushRect(ctx, rect) {
  ctx.save();
  ctx.strokeStyle = 'rgba(95,212,255,0.9)';
  ctx.fillStyle = 'rgba(95,212,255,0.08)';
  ctx.lineWidth = 1;
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
  ctx.restore();
}

/** Derive the highlighted-channel set from the current selection (or outliers). */
function _updateHlChannels() {
  const res = _state.pcaRes;
  if (!res || res.label == null) { _state.hlChannels = null; return; }
  const src = _state.pcaSel
    ? _state.pcaSel
    : (_state.pcaOutliers ? Array.from(_state.pcaOutliers.keys()).filter((i) => _state.pcaOutliers[i]) : []);
  if (!src.size && !(src.length)) { _state.hlChannels = null; return; }
  const chans = new Set();
  for (const i of src) { const c = res.label[i]; if (Number.isFinite(c)) chans.add(c); }
  _state.hlChannels = chans.size ? chans : null;
}

function _scatterMessage(canvas, msg) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'var(--text-tertiary, #888)';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(msg, canvas.width / 2, canvas.height / 2);
}

/** Append outlier/brush controls to the scatter section header. */
function _appendScatterControls(head) {
  const row = document.createElement('div');
  row.className = 'drilldown-scatter-ctrl';

  const k0 = (_state && _state.pcaOutlierK) || 2;
  const only0 = !!(_state && _state.pcaOutliersOnly);
  const kLab = document.createElement('label');
  kLab.className = 'drilldown-scatter-k';
  kLab.textContent = 'outlier k';
  const kVal = document.createElement('span');
  kVal.className = 'drilldown-scatter-k-val';
  kVal.textContent = k0.toFixed(1);
  const kSlider = document.createElement('input');
  kSlider.type = 'range'; kSlider.min = '1'; kSlider.max = '4';
  kSlider.step = '0.1'; kSlider.value = String(k0);
  kSlider.title = 'Outlier sensitivity: flag points farther than k·σ from the centroid.';
  kSlider.addEventListener('input', () => {
    if (!_state) return;
    _state.pcaOutlierK = +kSlider.value;
    kVal.textContent = (+kSlider.value).toFixed(1);
    _computeOutliers();
    _drawScatter();
    _updateHlChannels();
    _renderChannelGrid();
    _renderStrip();
    _publishScatterSummary();
  });
  kLab.appendChild(kSlider);
  kLab.appendChild(kVal);
  row.appendChild(kLab);

  const only = document.createElement('label');
  only.className = 'drilldown-scatter-only';
  const onlyChk = document.createElement('input');
  onlyChk.type = 'checkbox';
  onlyChk.checked = only0;
  onlyChk.addEventListener('change', () => {
    if (!_state) return;
    _state.pcaOutliersOnly = onlyChk.checked;
    _drawScatter();
    _publishScatterSummary();
  });
  only.appendChild(onlyChk);
  only.appendChild(document.createTextNode('outliers only'));
  row.appendChild(only);

  const clr = document.createElement('button');
  clr.type = 'button'; clr.className = 'btn btn-sm btn-secondary drilldown-scatter-clear';
  clr.textContent = 'Clear';
  clr.title = 'Clear the brush selection.';
  clr.addEventListener('click', () => {
    _state.pcaSel = null;
    _drawScatter();
    _updateHlChannels();
    _renderChannelGrid();
    _renderStrip();
    _publishScatterSummary();
  });
  row.appendChild(clr);

  head.appendChild(row);
}

/** Publish a short selection/outlier count next to the scatter. */
function _publishScatterSummary() {
  const res = _state.pcaRes;
  const nOut = _state.pcaOutliers ? Array.from(_state.pcaOutliers.values()).filter(Boolean).length : 0;
  const nSel = _state.pcaSel ? _state.pcaSel.size : 0;
  const nHl = _state.hlChannels ? _state.hlChannels.size : 0;
  _publishDebug({
    points: res && res.x ? res.x.length : 0,
    outliers: nOut,
    selected: nSel,
    hlChannels: nHl,
    hlChannelList: _state.hlChannels ? Array.from(_state.hlChannels).sort((a, b) => a - b) : [],
    pcaOutliersOnly: !!_state.pcaOutliersOnly,
    pcaOutlierK: _state.pcaOutlierK,
    hasLabels: !!(res && res.label != null),
  });
}

// ── Tensor math helpers (flat Float32Array + row-major strides) ────────────

function _buildStrides(shape) {
  const st = new Array(shape.length);
  let s = 1;
  for (let i = shape.length - 1; i >= 0; i--) { st[i] = s; s *= shape[i]; }
  return st;
}

/** Reduce a flat tensor over `redAxes` (mean or max); keep the rest in original order. */
function _reduceAxes(data, shape, redAxes, mode) {
  const nd = shape.length;
  if (!redAxes.length) return { data, shape: shape.slice() };
  const red = redAxes.slice().sort((a, b) => a - b);
  const keep = [];
  for (let i = 0; i < nd; i++) if (!red.includes(i)) keep.push(i);
  const newShape = keep.map((i) => shape[i]);
  const oldStr = _buildStrides(shape);
  const newStr = _buildStrides(newShape);
  const redShape = red.map((i) => shape[i]);
  const redStr = _buildStrides(redShape);
  const nOut = newShape.reduce((a, b) => a * b, 1);
  const nRed = redShape.reduce((a, b) => a * b, 1) || 1;
  const out = new Float32Array(nOut);
  const isMax = mode === 'max';
  for (let o = 0; o < nOut; o++) {
    let base = 0, oo = o;
    for (let k = 0; k < keep.length; k++) {
      base += (Math.floor(oo / newStr[k]) % newShape[k]) * oldStr[keep[k]];
      oo %= newStr[k];
    }
    let acc = isMax ? -Infinity : 0;
    for (let r = 0; r < nRed; r++) {
      let oi = base, rr = r;
      for (let a = 0; a < red.length; a++) {
        oi += (Math.floor(rr / redStr[a]) % redShape[a]) * oldStr[red[a]];
        rr %= redStr[a];
      }
      const v = data[oi];
      if (isMax) { if (v > acc) acc = v; } else acc += v;
    }
    if (!isMax) acc /= nRed;
    out[o] = acc;
  }
  return { data: out, shape: newShape };
}

/** Transpose a flat tensor by `perm` (axis indices of the input shape). */
function _transpose(data, shape, perm) {
  const nd = shape.length;
  const oldStr = _buildStrides(shape);
  const newShape = perm.map((i) => shape[i]);
  const newStr = _buildStrides(newShape);
  const n = data.length;
  const out = new Float32Array(n);
  for (let idx = 0; idx < n; idx++) {
    let oldIdx = 0, c = idx;
    for (let k = 0; k < nd; k++) {
      const coord = Math.floor(c / newStr[k]) % newShape[k];
      oldIdx += coord * oldStr[perm[k]];
      c %= newStr[k];
    }
    out[idx] = data[oldIdx];
  }
  return { data: out, shape: newShape };
}

/** Reduce over Feature+reduce axes, transpose to [channel, rows…, cols…], return maps. */
function _computeChannelMaps(flat, shape, roles, mode) {
  const chanAxis = roles.indexOf('channel');
  const rowAxes = _axesWithRole(roles, 'rows');
  const colAxes = _axesWithRole(roles, 'cols');
  const redAxes = _axesWithRole(roles, 'feature').concat(_axesWithRole(roles, 'reduce'));
  if (chanAxis < 0) throw new Error('no channel axis');
  const reduced = _reduceAxes(flat, shape, redAxes, mode);
  // Kept axes (in original order) → positions in the reduced shape.
  const kept = [];
  for (let i = 0; i < shape.length; i++) if (!redAxes.includes(i)) kept.push(i);
  const order = [chanAxis, ...rowAxes, ...colAxes].map((ax) => kept.indexOf(ax));
  const t = _transpose(reduced.data, reduced.shape, order);
  const C = t.shape[0];
  const rowDims = t.shape.slice(1, 1 + rowAxes.length);
  const colDims = t.shape.slice(1 + rowAxes.length);
  const R = rowDims.reduce((a, b) => a * b, 1) || 1;
  const C2 = colDims.reduce((a, b) => a * b, 1) || 1;
  return { channels: C, rows: R, cols: C2, data: t.data, perChannel: R * C2 };
}

/** One aggregate scalar per channel (reduce over all other axes). */
function _reduceStrip(flat, shape, chanAxis, mode) {
  const redAxes = [];
  for (let i = 0; i < shape.length; i++) if (i !== chanAxis) redAxes.push(i);
  const r = _reduceAxes(flat, shape, redAxes, mode);
  return r.data; // length = shape[chanAxis]
}

// ── Debug exposure for headless probes ────────────────────────────────────

function _publishDebug(extra = {}) {
  if (!window.__tensorDrilldown) window.__tensorDrilldown = {};
  const chanAxis = _state && _state.roles ? _state.roles.indexOf('channel') : -1;
  let channels = 0;
  if (chanAxis >= 0 && _state && _state.shape) channels = _state.shape[chanAxis];
  const grid = _modal && _modal.querySelector('.drilldown-channelgrid');
  Object.assign(window.__tensorDrilldown, {
    frameIdx: _state && _state.frameIdx,
    shape: _state && _state.shape,
    roles: _state && _state.roles ? _state.roles.slice() : null,
    channels,
    channelCanvases: grid ? grid.querySelectorAll('canvas.drilldown-canvas').length : 0,
    flatLength: _state && _state.tdata && _state.tdata.flat ? _state.tdata.flat.length : 0,
    ...extra,
  });
}

export { isMultiDim, openDrilldown };