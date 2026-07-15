/**
 * heatmap_popup.js — Per-layer heatmap inspector that lives inside the View
 * controls panel.
 *
 * Listens for `viewer:layer-click` CustomEvents dispatched by viewer3d.js
 * when the user clicks a 3D layer slab. Renders that layer's brightness grid
 * to the <canvas> below the view options using the viewer's exact
 * colormap/colour-scale mapping, and offers Save-as-PNG. Clicking another
 * layer re-draws the same canvas.
 */

import { viewer } from './viewer3d.js';
import { isMultiDim, openDrilldown } from './tensor_drilldown.js';
import { contains as isWatched, toggle as toggleWatch } from './watchlist.js';

const CANVAS_MAX_W = 272;       // CSS display width cap (px) — fits the 300px panel
const CANVAS_MAX_H = 320;       // CSS display height cap (px)

let _els = null;                // { canvas, title, saveBtn, drillBtn, hint }
let _frameIdx = null;
let _meta = null;

/** Resolve the fixed DOM elements inside the View panel on first use. */
function _elements() {
  if (_els) return _els;
  _els = {
    canvas:  document.getElementById('viz-heatmap-canvas'),
    title:   document.getElementById('viz-heatmap-title'),
    saveBtn: document.getElementById('viz-heatmap-save'),
    drillBtn: document.getElementById('viz-heatmap-drilldown'),
    watchBtn: document.getElementById('viz-heatmap-watch'),
    hint:    document.getElementById('viz-heatmap-hint'),
  };
  if (_els.saveBtn) {
    _els.saveBtn.addEventListener('click', () => {
      if (_els.canvas && _els.canvas.width > 0) _savePng(_els.canvas, _els.title.textContent);
    });
  }
  if (_els.drillBtn) {
    _els.drillBtn.addEventListener('click', () => {
      if (_frameIdx != null) openDrilldown(_frameIdx, _meta);
    });
  }
  if (_els.watchBtn) {
    _els.watchBtn.addEventListener('click', () => {
      if (_meta && _meta.node_name) {
        toggleWatch(_meta.node_name, _meta);
        _updateWatchBtn();
      }
    });
  }
  return _els;
}

/** Reflect whether the inspected layer is in the watch list (☆ Watch / ★ Watched). */
function _updateWatchBtn() {
  const btn = _els && _els.watchBtn;
  if (!btn) return;
  if (!_meta || !_meta.node_name) {
    btn.disabled = true;
    btn.textContent = '☆ Watch';
    btn.title = 'Add this layer to the watch list (kept across inference passes).';
    return;
  }
  btn.disabled = false;
  if (isWatched(_meta.node_name)) {
    btn.textContent = '★ Watched';
    btn.title = 'Remove this layer from the watch list.';
  } else {
    btn.textContent = '☆ Watch';
    btn.title = 'Add this layer to the watch list (kept across inference passes).';
  }
}

/** Render (or re-render) the heatmap for a frame index into the View panel. */
async function openHeatmapPopup(frameIdx, meta) {
  const els = _elements();
  if (!els.canvas) return;
  const grid = await viewer.ensureGrid(frameIdx);
  if (!grid || !grid.length || !grid[0] || !grid[0].length) return;

  _frameIdx = frameIdx;
  _meta = meta || null;
  const rows = grid.length;
  const cols = grid[0].length;
  const canvas = els.canvas;
  canvas.width = cols;
  canvas.height = rows;

  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) {
    const row = grid[r];
    for (let c = 0; c < cols; c++) {
      const [cr, cg, cb] = viewer.heatColor(row[c], frameIdx);
      const o = (r * cols + c) * 4;
      img.data[o] = cr;
      img.data[o + 1] = cg;
      img.data[o + 2] = cb;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  // CSS display size: scale small grids up, cap to both the panel width and a
  // max height. Set only the width and leave height:auto so the canvas keeps
  // its grid aspect ratio (max-width:100% can then shrink it without skewing).
  const scale = Math.max(1, Math.min(CANVAS_MAX_W / cols, CANVAS_MAX_H / rows, 8));
  canvas.style.width = `${Math.round(cols * scale)}px`;
  canvas.style.height = 'auto';

  const shape = (meta && (meta.original_shape || meta.shape)) || [];
  const name = (meta && meta.node_name) || `frame ${frameIdx}`;
  const op = (meta && meta.op_type) || '';
  els.title.textContent = `${name} [${shape.join(', ')}]${op ? ` · ${op}` : ''}`;
  els.title.title = els.title.textContent;

  els.canvas.hidden = false;
  if (els.hint) els.hint.hidden = true;
  els.saveBtn.disabled = false;
  // Drill-down only makes sense for genuinely multi-dimensional layers (≥3
  // non-trivial axes after dropping a batch-1). 2D layers already show their
  // whole grid in this inspector canvas.
  if (els.drillBtn) {
    const multi = isMultiDim(meta);
    els.drillBtn.disabled = !multi;
    els.drillBtn.title = multi
      ? 'Open the multi-dimensional drill-down for this layer (channel grid, PCA projection, per-channel strip).'
      : 'Drill-down is only available for multi-dimensional layers (3+ axes).';
  }
  _updateWatchBtn();
}

/** Save the heatmap canvas to a PNG download (pure client-side). */
function _savePng(canvas, titleText) {
  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const safe = (titleText || 'heatmap').replace(/[^a-zA-Z0-9_-]+/g, '_').slice(0, 60);
    a.href = url;
    a.download = `${safe}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, 'image/png');
}

// Wire up the viewer's layer-click events.
document.addEventListener('viewer:layer-click', (e) => {
  const { frameIdx, meta } = e.detail || {};
  if (frameIdx == null) return;
  openHeatmapPopup(frameIdx, meta);
});

export { openHeatmapPopup };