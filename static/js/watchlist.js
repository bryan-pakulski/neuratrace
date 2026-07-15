/**
 * watchlist.js — Pinned-layer watch list.
 *
 * Lets the user bookmark layers (by node name) and quickly jump back to them,
 * seeing each layer's metadata. The list is keyed by node name and persisted in
 * localStorage, so it survives page reloads AND new inference passes on the
 * same model (frame indices are re-resolved from the name whenever frames
 * load). Entries whose name isn't in the currently-loaded model are shown
 * greyed ("not in this model") but kept, so switching models back restores them.
 *
 * Exports: { init, add, remove, contains, toggle, refresh }
 *
 * Adding is driven from the heatmap inspector's Watch button (heatmap_popup.js,
 * which owns the "currently inspected layer"); this module owns the list panel
 * + navigation. Clicking a row selects that layer (via the normal layer-click
 * path) and centres the camera on it.
 */

import { viewer } from './viewer3d.js';

const LS_KEY = 'onnx_viewer.watchlist';

let _entries = [];          // ordered node names (canonical store)
let _body = null;            // #watchlist-body
let _root = null;            // #watchlist
let _count = null;           // #watchlist-count
let _empty = null;           // #watchlist-empty
let _clearBtn = null;        // #watchlist-clear
let _wired = false;

// ── Public API ─────────────────────────────────────────────────────────────

/** Wire DOM + load persisted entries. Idempotent. */
function init() {
  if (_wired) return;
  _wired = true;
  _root = document.getElementById('watchlist');
  _body = document.getElementById('watchlist-body');
  _count = document.getElementById('watchlist-count');
  _empty = document.getElementById('watchlist-empty');
  _clearBtn = document.getElementById('watchlist-clear');
  _load();
  if (_clearBtn) {
    _clearBtn.addEventListener('click', () => {
      _entries = [];
      _persist();
      render();
    });
  }
  render();
}

/** Is `name` currently watched? */
function contains(name) {
  return _entries.indexOf(name) >= 0;
}

/** Add a layer by node name (no-op if already present). */
function add(name) {
  if (!name || _entries.indexOf(name) >= 0) return;
  _entries.unshift(name);     // newest first
  _persist();
  render();
}

/** Remove a layer by node name. */
function remove(name) {
  const i = _entries.indexOf(name);
  if (i < 0) return;
  _entries.splice(i, 1);
  _persist();
  render();
}

/** Toggle a layer's watched state. Returns the new watched state. */
function toggle(name, meta) {
  if (!name) return false;
  if (contains(name)) { remove(name); return false; }
  add(name);
  return true;
}

/** Re-resolve names → frame indices and re-render (call after frames load /
 *  after a new inference pass so metadata + navigation stay current). */
function refresh() {
  render();
}

// ── Rendering ──────────────────────────────────────────────────────────────

function render() {
  if (!_root) return;
  const n = _entries.length;
  _root.hidden = n === 0;          // hide the whole section when empty
  if (_count) _count.textContent = n ? `${n}` : '';
  if (_clearBtn) _clearBtn.hidden = n === 0;
  if (_empty) _empty.hidden = n !== 0;
  if (!_body) return;
  _body.innerHTML = '';
  const framesLoaded = viewer.frames && viewer.frames.length > 0;
  for (const name of _entries) {
    _body.appendChild(_renderRow(name, framesLoaded));
  }
}

function _renderRow(name, framesLoaded) {
  const idx = viewer.frameIndexForNode(name);
  const present = idx != null && idx >= 0;
  const meta = present ? viewer.getFrameMeta(idx) : null;
  const row = document.createElement('div');
  row.className = 'watchlist-row' + (present ? '' : ' watchlist-row-missing');

  // Remove star.
  const star = document.createElement('button');
  star.type = 'button';
  star.className = 'watchlist-star';
  star.title = 'Remove from watch list';
  star.textContent = '✕';
  star.addEventListener('click', (e) => { e.stopPropagation(); remove(name); });
  row.appendChild(star);

  // Name + metadata block.
  const info = document.createElement('div');
  info.className = 'watchlist-info';
  const nameEl = document.createElement('div');
  nameEl.className = 'watchlist-name';
  nameEl.textContent = name;
  nameEl.title = name;
  info.appendChild(nameEl);

  const det = document.createElement('div');
  det.className = 'watchlist-detail';
  if (present && meta) {
    const shape = meta.original_shape || meta.shape || [];
    const parts = [];
    if (meta.op_type) parts.push(meta.op_type);
    parts.push(`[${shape.join('×')}]`);
    if (meta.exec_order != null) parts.push(`#${meta.exec_order}`);
    if (meta.sampled) parts.push('sampled');
    det.textContent = parts.join(' · ');
  } else {
    det.textContent = framesLoaded ? 'not in this model' : 'pending…';
  }
  info.appendChild(det);
  row.appendChild(info);

  if (present) {
    row.title = 'Click to select & centre this layer in the 3D view';
    row.classList.add('watchlist-row-nav');
    row.addEventListener('click', () => _navigate(idx, meta));
  }
  return row;
}

/** Select + centre a watched layer (same path as a connector-row click). */
function _navigate(frameIdx, meta) {
  document.dispatchEvent(new CustomEvent('viewer:layer-click', {
    detail: { frameIdx, meta: meta || viewer.getFrameMeta(frameIdx) },
  }));
  viewer.centerOnLayer(frameIdx);
}

// ── Persistence ─────────────────────────────────────────────────────────────

function _load() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return;
    const arr = JSON.parse(raw);
    if (Array.isArray(arr)) _entries = arr.filter((x) => typeof x === 'string');
  } catch (e) { /* localStorage unavailable / corrupt */ }
}

function _persist() {
  try { localStorage.setItem(LS_KEY, JSON.stringify(_entries)); } catch (e) {}
}

export { init, add, remove, contains, toggle, refresh };