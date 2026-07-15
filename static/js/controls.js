/**
 * controls.js — Transport + visualization controls for the 3D activation cube.
 *
 * Wires DOM transport controls to the viewer3d singleton.
 * Exports: { loadFrames, enableControls, disableControls }
 *
 * The cube is static by default (all layers visible). Play / scrub / step
 * engage the viewer's focus sweep (dims layers far from the playhead); stop
 * returns to the full static cube. Visualization controls: colormap, point
 * fill, density (points per layer), size metric, and colour scale
 * (per-layer / global). Layer gap and size-scale are fixed. Preferences
 * persist via localStorage.
 */

import { viewer } from './viewer3d.js';
import { setEnabled as setConnectionsEnabled, init as initConnections } from './connections.js';
import { init as initWatchlist, refresh as refreshWatchlist } from './watchlist.js';

// ── DOM References ──────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const els = {
  // Transport buttons
  btnRewind:     $('btn-rewind'),
  btnPlayPause:  $('btn-play-pause'),
  btnStop:       $('btn-stop'),
  btnStepBack:   $('btn-step-back'),
  btnStepFwd:    $('btn-step-fwd'),
  btnLoop:       $('btn-loop'),
  // Icons inside play/pause button
  iconPlay:      $('icon-play'),
  iconPause:     $('icon-pause'),
  // Scrubber + counter
  scrubber:      $('scrubber'),
  frameCounter:  $('frame-counter'),
  // Speed
  speedSelect:   $('speed-select'),
  // Visualization controls
  colormapSelect:    $('colormap-select'),
  pointSizeSlider:   $('point-size-slider'),
  densitySlider:     $('density-slider'),
  colorScaleSelect:  $('color-scale-select'),
  sizeMetricSelect:  $('size-metric-select'),
  hideUpcomingCheck: $('hide-upcoming-check'),
  showConnectionsCheck: $('show-connections-check'),
  // Floating view-controls window
  vizFloating:       $('viz-floating'),
  vizFloatingHeader: $('viz-floating-header'),
  vizFloatingCollapse: $('viz-floating-collapse'),
  // Frame info bar
  fiLabel:       $('frame-info-label'),
  fiOp:          $('frame-info-op'),
  fiShape:       $('frame-info-shape'),
  fiExec:        $('frame-info-exec'),
  fiSampled:     $('frame-info-sampled'),
  // Canvas states
  canvasEmpty:   $('canvas-empty'),
  // Status
  statusLabel:   document.querySelector('.status-label'),
};

// ── State ───────────────────────────────────────────────────────

let _framesLoaded = false;
let _isScrubbing = false;

// Persisted visualization preference keys.
const _LS = {
  colormap: 'onnx_viewer.colormap',
  pointSize: 'onnx_viewer.point_fill',  // fill fraction (renamed from point_size)
  density: 'onnx_viewer.density',
  colorScale: 'onnx_viewer.color_scale',
  sizeMetric: 'onnx_viewer.size_metric',
  hideUpcoming: 'onnx_viewer.hide_upcoming',
  showConnections: 'onnx_viewer.show_connections',
  vizPos: 'onnx_viewer.viz_pos',
  vizCollapsed: 'onnx_viewer.viz_collapsed',
};

/** Apply persisted visualization prefs to the viewer (after frames load). */
function _applyPersistedVizPrefs() {
  try {
    const cm = localStorage.getItem(_LS.colormap);
    const ps = localStorage.getItem(_LS.pointSize);
    const dens = localStorage.getItem(_LS.density);
    const scale = localStorage.getItem(_LS.colorScale);
    if (cm && els.colormapSelect) { els.colormapSelect.value = cm; viewer.setColormap(cm); }
    if (ps && els.pointSizeSlider) {
      const v = parseFloat(ps);
      if (!isNaN(v)) { els.pointSizeSlider.value = ps; viewer.setPointSize(v); }
    }
    if (dens && els.densitySlider) {
      const v = parseFloat(dens);
      if (!isNaN(v)) { els.densitySlider.value = dens; viewer.setDensity(v); }
    }
    if (scale && els.colorScaleSelect) {
      els.colorScaleSelect.value = scale;
      viewer.setColorScale(scale);
    }
    const smetric = localStorage.getItem(_LS.sizeMetric);
    if (smetric && els.sizeMetricSelect) {
      els.sizeMetricSelect.value = smetric;
      viewer.setSizeMetric(smetric);
    }
    const hideUp = localStorage.getItem(_LS.hideUpcoming);
    if (hideUp !== null && els.hideUpcomingCheck) {
      const on = hideUp === '1';
      els.hideUpcomingCheck.checked = on;
      viewer.setHideUpcoming(on);
    }
    const showConn = localStorage.getItem(_LS.showConnections);
    if (showConn === '1' && els.showConnectionsCheck) {
      els.showConnectionsCheck.checked = true;
      setConnectionsEnabled(true);
    }
  } catch (e) { /* localStorage unavailable */ }
}

// ── Helpers ─────────────────────────────────────────────────────

/** Toggle play/pause icon visibility. */
function _updatePlayIcon() {
  if (!els.iconPlay || !els.iconPause) return;
  const playing = viewer.isPlaying;
  els.iconPlay.hidden = playing;
  els.iconPause.hidden = !playing;
  if (els.btnPlayPause) {
    els.btnPlayPause.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    els.btnPlayPause.title = playing ? 'Pause' : 'Play';
  }
}

/** Update frame counter text: "current / total". */
function _updateFrameCounter() {
  if (!els.frameCounter) return;
  const cur = viewer.currentFrame;
  const total = viewer.frameCount;
  els.frameCounter.textContent = `${cur + 1} / ${total}`;
}

/** Update scrubber position (without triggering input event). */
function _updateScrubber() {
  if (!els.scrubber || _isScrubbing) return;
  const cur = viewer.currentFrame;
  els.scrubber.value = String(cur);
}

/** Update frame info bar with metadata. */
function _updateFrameInfo(frameIdx, meta) {
  if (els.fiLabel) els.fiLabel.textContent = meta.node_name || '—';
  if (els.fiOp) els.fiOp.textContent = meta.op_type || '';
  if (els.fiShape) {
    const shape = meta.original_shape || meta.shape || [];
    els.fiShape.textContent = `[${shape.join(', ')}]`;
  }
  if (els.fiExec) {
    els.fiExec.textContent = meta.exec_order != null ? `#${meta.exec_order}` : '';
  }
  if (els.fiSampled) {
    els.fiSampled.hidden = !meta.sampled;
  }
  _updateFrameCounter();
  _updateScrubber();
}

/** Enable all transport controls. */
function enableControls() {
  const buttons = [els.btnRewind, els.btnPlayPause, els.btnStop,
                   els.btnStepBack, els.btnStepFwd, els.btnLoop];
  for (const btn of buttons) {
    if (btn) btn.disabled = false;
  }
  if (els.scrubber) {
    els.scrubber.disabled = false;
    els.scrubber.max = String(viewer.frameCount - 1);
  }
  if (els.speedSelect) els.speedSelect.disabled = false;
  if (els.colormapSelect) els.colormapSelect.disabled = false;
  if (els.pointSizeSlider) els.pointSizeSlider.disabled = false;
  if (els.densitySlider) els.densitySlider.disabled = false;
  if (els.colorScaleSelect) els.colorScaleSelect.disabled = false;
  if (els.sizeMetricSelect) els.sizeMetricSelect.disabled = false;
  if (els.hideUpcomingCheck) els.hideUpcomingCheck.disabled = false;
  if (els.showConnectionsCheck) els.showConnectionsCheck.disabled = false;
  if (els.vizFloating) els.vizFloating.hidden = false;
  _framesLoaded = true;
}

/** Disable all transport controls. */
function disableControls() {
  const buttons = [els.btnRewind, els.btnPlayPause, els.btnStop,
                   els.btnStepBack, els.btnStepFwd, els.btnLoop];
  for (const btn of buttons) {
    if (btn) btn.disabled = true;
  }
  if (els.scrubber) {
    els.scrubber.disabled = true;
    els.scrubber.max = '0';
    els.scrubber.value = '0';
  }
  if (els.speedSelect) els.speedSelect.disabled = true;
  if (els.colormapSelect) els.colormapSelect.disabled = true;
  if (els.pointSizeSlider) els.pointSizeSlider.disabled = true;
  if (els.densitySlider) els.densitySlider.disabled = true;
  if (els.colorScaleSelect) els.colorScaleSelect.disabled = true;
  if (els.sizeMetricSelect) els.sizeMetricSelect.disabled = true;
  if (els.hideUpcomingCheck) els.hideUpcomingCheck.disabled = true;
  if (els.showConnectionsCheck) els.showConnectionsCheck.disabled = true;
  if (els.vizFloating) els.vizFloating.hidden = true;
  if (els.frameCounter) els.frameCounter.textContent = '0 / 0';
  _framesLoaded = false;
}

/** Update status pill label. */
function _updateStatus(msg) {
  if (els.statusLabel) {
    els.statusLabel.textContent = msg;
  }
}

// ── Frame Loading (called by inference.js) ──────────────────────

/**
 * Load frame metadata into the viewer and enable controls.
 * @param {Array} frameMetadata - array of frame metadata from /api/inference
 */
async function loadFrames(frameMetadata) {
  if (!frameMetadata || frameMetadata.length === 0) {
    console.warn('controls.js: no frames to load');
    return;
  }

  // Hide empty state
  if (els.canvasEmpty) els.canvasEmpty.hidden = true;

  // Load into viewer
  await viewer.loadFrames(frameMetadata);

  // Enable controls
  enableControls();

  // Set initial loop state from button
  if (els.btnLoop) {
    const loopActive = els.btnLoop.dataset.active === 'true';
    viewer.setLoop(loopActive);
  }

  // Set initial speed from select
  if (els.speedSelect) {
    viewer.setSpeed(parseFloat(els.speedSelect.value) || 1.0);
  }

  // Apply persisted visualization prefs (now that frames exist).
  _applyPersistedVizPrefs();

  // Re-resolve the watch list against the freshly loaded frames (names →
  // frame indices + metadata), so it stays accurate across inference passes.
  refreshWatchlist();

  // Update frame info for frame 0
  if (frameMetadata[0]) {
    _updateFrameInfo(0, frameMetadata[0]);
  }

  _updatePlayIcon();
  _updateStatus(`${frameMetadata.length} frames loaded`);
}

// ── Event Wiring ────────────────────────────────────────────────

function _init() {
  // Register viewer callbacks
  viewer.onFrameChange((frameIdx, meta) => {
    _updateFrameInfo(frameIdx, meta);
  });

  viewer.onStatusChange((msg) => {
    _updateStatus(msg);
  });

  // Wire the connection-visualization click handler + info panel.
  initConnections();

  // Wire the pinned-layer watch list.
  initWatchlist();

  // Play/Pause
  if (els.btnPlayPause) {
    els.btnPlayPause.addEventListener('click', () => {
      if (viewer.isPlaying) {
        viewer.pause();
      } else {
        viewer.play();
      }
      _updatePlayIcon();
    });
  }

  // Stop
  if (els.btnStop) {
    els.btnStop.addEventListener('click', () => {
      viewer.stop();
      _updatePlayIcon();
    });
  }

  // Rewind
  if (els.btnRewind) {
    els.btnRewind.addEventListener('click', () => {
      viewer.rewind();
      _updatePlayIcon();
    });
  }

  // Step Forward
  if (els.btnStepFwd) {
    els.btnStepFwd.addEventListener('click', () => {
      viewer.pause();
      viewer.stepForward();
      _updatePlayIcon();
    });
  }

  // Step Backward
  if (els.btnStepBack) {
    els.btnStepBack.addEventListener('click', () => {
      viewer.pause();
      viewer.stepBackward();
      _updatePlayIcon();
    });
  }

  // Loop toggle
  if (els.btnLoop) {
    els.btnLoop.addEventListener('click', () => {
      const active = els.btnLoop.dataset.active === 'true';
      els.btnLoop.dataset.active = String(!active);
      viewer.setLoop(!active);
    });
  }

  // Scrubber
  if (els.scrubber) {
    els.scrubber.addEventListener('input', () => {
      _isScrubbing = true;
      const idx = parseInt(els.scrubber.value, 10);
      viewer.pause();
      viewer.setFrame(idx);
      _updatePlayIcon();
    });

    els.scrubber.addEventListener('change', () => {
      _isScrubbing = false;
    });
  }

  // Speed select
  if (els.speedSelect) {
    els.speedSelect.addEventListener('change', () => {
      const speed = parseFloat(els.speedSelect.value) || 1.0;
      viewer.setSpeed(speed);
    });
  }

  // ── Keyboard shortcuts ──────────────────────────────────────
  // ── Visualization controls (colormap, point size, layer gap, density, scale) ──
  // Persisted prefs are applied to the viewer after frames load via
  // _applyPersistedVizPrefs(); listeners below keep the viewer in sync on change.

  if (els.colormapSelect) {
    els.colormapSelect.addEventListener('change', () => {
      viewer.setColormap(els.colormapSelect.value);
      try { localStorage.setItem(_LS.colormap, els.colormapSelect.value); } catch (e) {}
    });
  }

  if (els.pointSizeSlider) {
    els.pointSizeSlider.addEventListener('input', () => {
      const v = parseFloat(els.pointSizeSlider.value);
      if (!isNaN(v)) {
        viewer.setPointSize(v);
        try { localStorage.setItem(_LS.pointSize, String(v)); } catch (e) {}
      }
    });
  }

  // Density rebuilds every layer's geometry — debounce so dragging stays smooth.
  let _densityDebounce = null;
  if (els.densitySlider) {
    els.densitySlider.addEventListener('input', () => {
      const v = parseFloat(els.densitySlider.value);
      if (isNaN(v)) return;
      try { localStorage.setItem(_LS.density, String(v)); } catch (e) {}
      if (_densityDebounce) clearTimeout(_densityDebounce);
      _densityDebounce = setTimeout(() => {
        viewer.setDensity(v);
        _densityDebounce = null;
      }, 120);
    });
  }

  if (els.colorScaleSelect) {
    els.colorScaleSelect.addEventListener('change', () => {
      viewer.setColorScale(els.colorScaleSelect.value);
      try { localStorage.setItem(_LS.colorScale, els.colorScaleSelect.value); } catch (e) {}
    });
  }

  if (els.sizeMetricSelect) {
    els.sizeMetricSelect.addEventListener('change', () => {
      viewer.setSizeMetric(els.sizeMetricSelect.value);
      try { localStorage.setItem(_LS.sizeMetric, els.sizeMetricSelect.value); } catch (e) {}
    });
  }

  if (els.hideUpcomingCheck) {
    els.hideUpcomingCheck.addEventListener('change', () => {
      const on = els.hideUpcomingCheck.checked;
      viewer.setHideUpcoming(on);
      try { localStorage.setItem(_LS.hideUpcoming, on ? '1' : '0'); } catch (e) {}
    });
  }

  if (els.showConnectionsCheck) {
    els.showConnectionsCheck.addEventListener('change', () => {
      const on = els.showConnectionsCheck.checked;
      setConnectionsEnabled(on);
      try { localStorage.setItem(_LS.showConnections, on ? '1' : '0'); } catch (e) {}
    });
  }

  document.addEventListener('keydown', (e) => {
    // Don't intercept if focus is in an input/textarea/select
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (!_framesLoaded) return;

    switch (e.key) {
      case ' ':
        e.preventDefault();
        if (viewer.isPlaying) {
          viewer.pause();
        } else {
          viewer.play();
        }
        _updatePlayIcon();
        break;
      case 'ArrowRight':
        e.preventDefault();
        viewer.pause();
        viewer.stepForward();
        _updatePlayIcon();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        viewer.pause();
        viewer.stepBackward();
        _updatePlayIcon();
        break;
      case 'Home':
        e.preventDefault();
        viewer.rewind();
        _updatePlayIcon();
        break;
      case 'End':
        e.preventDefault();
        viewer.setFrame(viewer.frameCount - 1);
        _updatePlayIcon();
        break;
    }
  });

  _initVizFloating();
}

// ── Floating View-controls window: drag, collapse, persist ────────

/** Clamp the floating panel inside its canvas container. */
function _clampVizPos(left, top, container, panel) {
  const cw = container.clientWidth, ch = container.clientHeight;
  const pw = panel.offsetWidth, ph = panel.offsetHeight;
  return {
    left: Math.max(0, Math.min(left, cw - pw)),
    top: Math.max(0, Math.min(top, ch - ph)),
  };
}

function _initVizFloating() {
  const panel = els.vizFloating;
  const header = els.vizFloatingHeader;
  const collapseBtn = els.vizFloatingCollapse;
  if (!panel || !header) return;
  const container = panel.parentElement; // .canvas-container

  // Restore persisted position + collapse state.
  try {
    const pos = localStorage.getItem(_LS.vizPos);
    if (pos) {
      const { left, top } = JSON.parse(pos);
      if (Number.isFinite(left) && Number.isFinite(top)) {
        const c = _clampVizPos(left, top, container, panel);
        panel.style.left = `${c.left}px`;
        panel.style.top = `${c.top}px`;
        panel.style.right = 'auto';
      }
    }
    if (localStorage.getItem(_LS.vizCollapsed) === '1') {
      panel.classList.add('collapsed');
    }
  } catch (e) { /* localStorage unavailable */ }

  // Collapse / expand toggle.
  if (collapseBtn) {
    collapseBtn.addEventListener('click', () => {
      const collapsed = panel.classList.toggle('collapsed');
      try { localStorage.setItem(_LS.vizCollapsed, collapsed ? '1' : '0'); } catch (e) {}
      // Re-clamp in case height changed.
      if (panel.style.left || panel.style.top) {
        const c = _clampVizPos(parseFloat(panel.style.left) || 0,
                               parseFloat(panel.style.top) || 0, container, panel);
        panel.style.left = `${c.left}px`;
        panel.style.top = `${c.top}px`;
      }
    });
  }

  // Drag by the header.
  let dragging = false;
  let startPx = 0, startPy = 0, startLeft = 0, startTop = 0;

  header.addEventListener('pointerdown', (e) => {
    if (e.target === collapseBtn) return; // don't drag from the button
    dragging = true;
    header.classList.add('dragging');
    const rect = panel.getBoundingClientRect();
    // Switch from right-anchored to explicit left/top.
    startLeft = rect.left - container.getBoundingClientRect().left;
    startTop = rect.top - container.getBoundingClientRect().top;
    panel.style.left = `${startLeft}px`;
    panel.style.top = `${startTop}px`;
    panel.style.right = 'auto';
    startPx = e.clientX;
    startPy = e.clientY;
    header.setPointerCapture(e.pointerId);
    e.preventDefault();
  });

  header.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const left = startLeft + (e.clientX - startPx);
    const top = startTop + (e.clientY - startPy);
    const c = _clampVizPos(left, top, container, panel);
    panel.style.left = `${c.left}px`;
    panel.style.top = `${c.top}px`;
  });

  const endDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    header.classList.remove('dragging');
    try {
      localStorage.setItem(_LS.vizPos, JSON.stringify({
        left: parseFloat(panel.style.left) || 0,
        top: parseFloat(panel.style.top) || 0,
      }));
    } catch (e) { /* localStorage unavailable */ }
  };
  header.addEventListener('pointerup', endDrag);
  header.addEventListener('pointercancel', endDrag);

  // Keep the panel inside the container when the canvas resizes.
  window.addEventListener('resize', () => {
    if (panel.hidden) return;
    if (panel.style.left || panel.style.top) {
      const c = _clampVizPos(parseFloat(panel.style.left) || 0,
                             parseFloat(panel.style.top) || 0, container, panel);
      panel.style.left = `${c.left}px`;
      panel.style.top = `${c.top}px`;
    }
  });
}

// ── Auto-init when DOM ready ────────────────────────────────────

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _init);
} else {
  _init();
}

// ── Exports ─────────────────────────────────────────────────────

export { loadFrames, enableControls, disableControls };