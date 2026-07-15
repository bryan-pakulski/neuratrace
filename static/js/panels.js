/**
 * panels.js — Resizable & collapsible side panels.
 *
 * Drag the handles between panels to resize. Click chevron buttons
 * in panel headers to collapse/expand. Dispatches window 'resize'
 * events so viewer3d.js recalculates canvas dimensions.
 */

// ── Config ──────────────────────────────────────────────────────────

const MIN_TREE_W = 150;
const MAX_TREE_W = 500;
const MIN_INF_W  = 200;
const MAX_INF_W  = 600;
const COLLAPSED_W = 36;
const ANIM_FADE_MS = 160; // matches .panel-header/.panel-body opacity transition

// ── State ────────────────────────────────────────────────────────────

let _treeW = 240;
let _infW  = 320;
let _treeCollapsed = false;
let _infCollapsed  = false;

// ── Helpers ─────────────────────────────────────────────────────────

function _setTreeWidth(w) {
  _treeW = w;
  document.documentElement.style.setProperty('--tree-w', w + 'px');
}

function _setInfWidth(w) {
  _infW = w;
  document.documentElement.style.setProperty('--inf-w', w + 'px');
}

let _rafId = null;
function _dispatchResize() {
  if (_rafId) cancelAnimationFrame(_rafId);
  _rafId = requestAnimationFrame(() => {
    window.dispatchEvent(new Event('resize'));
  });
}

// ── Collapse / Expand animation ──────────────────────────────────────
// Two-phase: the grid track width slides via the .app-main transition,
// while header/body cross-fade. display:none can't transition, so the
// .collapsed class (which hides content) is applied only after the
// fade-out completes, and removed before the fade-in starts.

function _collapsePanel(panel, collapsedBar, applyNarrow) {
  if (!panel) return;
  if (panel._animTimer) { clearTimeout(panel._animTimer); panel._animTimer = null; }
  // Cancel an in-flight expand.
  panel.classList.remove('starting');
  if (panel.classList.contains('collapsed')) return; // already collapsed

  panel.classList.add('fading');        // fade content out
  applyNarrow();                        // slide width shut
  if (collapsedBar) collapsedBar.hidden = false;
  _dispatchResize();

  panel._animTimer = setTimeout(() => {
    panel.classList.remove('fading');
    panel.classList.add('collapsed');   // now safe to display:none
    panel._animTimer = null;
    _dispatchResize();
  }, ANIM_FADE_MS);
}

function _expandPanel(panel, collapsedBar, applyWide) {
  if (!panel) return;
  if (panel._animTimer) { clearTimeout(panel._animTimer); panel._animTimer = null; }

  const wasFading   = panel.classList.contains('fading');
  const wasCollapsed = panel.classList.contains('collapsed');
  panel.classList.remove('fading');

  if (wasFading) {
    // Mid-collapse: just reverse the opacity transition back to 1.
  } else if (wasCollapsed) {
    // Fully hidden: reveal faded-out, then fade in. Force a reflow so
    // the opacity:0 starting state is committed before we remove it —
    // rAF would work too but is throttled in background tabs.
    panel.classList.add('starting');
    panel.classList.remove('collapsed');
    void panel.offsetWidth;
    panel.classList.remove('starting');
  }

  applyWide();                          // slide width open
  if (collapsedBar) collapsedBar.hidden = true;
  _dispatchResize();
}

// ── Resize handles ───────────────────────────────────────────────────

function _initResize() {
  const handleLeft  = document.getElementById('handle-left');
  const handleRight = document.getElementById('handle-right');

  function _wireHandle(handle, opts) {
    if (!handle) return;
    handle.addEventListener('mousedown', (e) => {
      if (opts.collapsedRef()) return;
      e.preventDefault();
      document.body.classList.add('panel-resizing');

      const startX  = e.clientX;
      const startW  = opts.getStartW();
      const onMove  = opts.makeOnMove(startX, startW);
      const onUp    = () => {
        document.body.classList.remove('panel-resizing');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        _dispatchResize();
      };
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }

  _wireHandle(handleLeft, {
    collapsedRef: () => _treeCollapsed,
    getStartW:    () => _treeW,
    makeOnMove: (startX, startW) => (e) => {
      const dx = e.clientX - startX;
      _setTreeWidth(Math.max(MIN_TREE_W, Math.min(MAX_TREE_W, startW + dx)));
      _dispatchResize();
    },
  });

  _wireHandle(handleRight, {
    collapsedRef: () => _infCollapsed,
    getStartW:    () => _infW,
    makeOnMove: (startX, startW) => (e) => {
      const dx = e.clientX - startX;
      _setInfWidth(Math.max(MIN_INF_W, Math.min(MAX_INF_W, startW - dx)));
      _dispatchResize();
    },
  });
}

// ── Collapse / Expand ────────────────────────────────────────────────

function _initCollapse() {
  // ── Tree panel ──
  const treePanel  = document.getElementById('panel-tree');
  const treeToggle = document.getElementById('btn-collapse-tree');
  const treeExpand = document.getElementById('btn-expand-tree');
  const treeHandle = document.getElementById('handle-left');

  function toggleTree() {
    _treeCollapsed = !_treeCollapsed;
    const collapsedBar = document.getElementById('collapsed-tree-bar');
    if (_treeCollapsed) {
      _collapsePanel(treePanel, collapsedBar, () => {
        document.documentElement.style.setProperty('--tree-w', COLLAPSED_W + 'px');
        document.documentElement.style.setProperty('--handle-left-w', '0px');
      });
    } else {
      _expandPanel(treePanel, collapsedBar, () => {
        _setTreeWidth(_treeW);
        document.documentElement.style.setProperty('--handle-left-w', '4px');
      });
    }
  }

  if (treeToggle) treeToggle.addEventListener('click', toggleTree);
  if (treeExpand) treeExpand.addEventListener('click', toggleTree);

  // ── Inference panel ──
  const infPanel  = document.getElementById('panel-inference');
  const infToggle = document.getElementById('btn-collapse-inference');
  const infExpand = document.getElementById('btn-expand-inference');
  const infHandle = document.getElementById('handle-right');

  function toggleInf() {
    _infCollapsed = !_infCollapsed;
    const collapsedBar = document.getElementById('collapsed-inference-bar');
    if (_infCollapsed) {
      _collapsePanel(infPanel, collapsedBar, () => {
        document.documentElement.style.setProperty('--inf-w', COLLAPSED_W + 'px');
        document.documentElement.style.setProperty('--handle-right-w', '0px');
      });
    } else {
      _expandPanel(infPanel, collapsedBar, () => {
        _setInfWidth(_infW);
        document.documentElement.style.setProperty('--handle-right-w', '4px');
      });
    }
  }

  if (infToggle) infToggle.addEventListener('click', toggleInf);
  if (infExpand) infExpand.addEventListener('click', toggleInf);
}

// ── Init ─────────────────────────────────────────────────────────────

function _init() {
  const main = document.querySelector('.app-main');

  // Start with narrower side panels on small viewports. Subsequent
  // drags clamp to MIN/MAX_TREE_W / MIN/MAX_INF_W regardless.
  if (window.innerWidth <= 900) {
    _treeW = Math.min(_treeW, 200);
    _infW  = Math.min(_infW, 280);
  }

  // Set initial widths without animating (avoid a slide on load).
  if (main) main.classList.add('no-anim');
  _setTreeWidth(_treeW);
  _setInfWidth(_infW);
  _initResize();
  _initCollapse();
  requestAnimationFrame(() => requestAnimationFrame(() => {
    if (main) main.classList.remove('no-anim');
  }));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _init);
} else {
  _init();
}