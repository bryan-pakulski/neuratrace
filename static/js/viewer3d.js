/**
 * viewer3d.js — Three.js 3D Activation Cube
 *
 * The network is rendered as a single 3D volume ("cube"): each captured
 * layer (ONNX node output) is a square slab of points stacked along Z.
 * Every point is one activation ("neuron" = one cell of the layer's
 * brightness grid), coloured by activation strength via a colormap.
 *
 * - Square footprint: every layer fills the same XY square regardless of
 *   its neuron count, so stacked slabs form a cube.
 * - Density control: client-side subsampling caps points per layer (and a
 *   global cap) so the browser stays responsive.
 * - Colour scale: per-layer (default) or global across the network
 *   (reconstructed from each layer's pre-normalization raw_min/raw_max).
 * - Focus sweep: by default all layers are shown at full opacity (static
 *   cube). The transport controls (play / scrubber / step) engage a focus
 *   plane that dims layers far from the playhead; stop() returns to the
 *   full static cube.
 *
 * Exports: { viewer, Viewer3D, COLORMAPS, sampleColormap }
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Colormaps ────────────────────────────────────────────────────────────

const COLORMAPS = {
  viridis: [
    [0.267, 0.005, 0.329],
    [0.282, 0.140, 0.457],
    [0.254, 0.265, 0.530],
    [0.207, 0.372, 0.553],
    [0.164, 0.471, 0.558],
    [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518],
    [0.267, 0.749, 0.441],
    [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.150],
    [0.993, 0.906, 0.144],
  ],
  plasma: [
    [0.051, 0.033, 0.526],
    [0.163, 0.058, 0.526],
    [0.280, 0.075, 0.529],
    [0.395, 0.083, 0.518],
    [0.505, 0.090, 0.494],
    [0.608, 0.115, 0.451],
    [0.706, 0.165, 0.387],
    [0.796, 0.237, 0.301],
    [0.870, 0.330, 0.200],
    [0.925, 0.440, 0.096],
    [0.963, 0.571, 0.043],
    [0.984, 0.711, 0.078],
    [0.988, 0.845, 0.157],
    [0.973, 0.949, 0.255],
  ],
  inferno: [
    [0.001, 0.000, 0.014],
    [0.062, 0.031, 0.110],
    [0.131, 0.055, 0.220],
    [0.208, 0.068, 0.329],
    [0.287, 0.073, 0.441],
    [0.363, 0.078, 0.557],
    [0.434, 0.095, 0.664],
    [0.500, 0.132, 0.750],
    [0.563, 0.182, 0.809],
    [0.625, 0.240, 0.844],
    [0.687, 0.303, 0.855],
    [0.749, 0.368, 0.841],
    [0.808, 0.434, 0.805],
    [0.862, 0.500, 0.751],
    [0.910, 0.566, 0.683],
    [0.949, 0.631, 0.604],
    [0.976, 0.695, 0.520],
    [0.993, 0.761, 0.434],
    [0.998, 0.827, 0.342],
    [0.995, 0.891, 0.243],
    [0.983, 0.955, 0.139],
  ],
  led: [
    [0.02, 0.02, 0.02],
    [0.10, 0.01, 0.01],
    [0.22, 0.02, 0.01],
    [0.38, 0.03, 0.01],
    [0.55, 0.06, 0.01],
    [0.70, 0.12, 0.02],
    [0.82, 0.22, 0.03],
    [0.90, 0.35, 0.05],
    [0.95, 0.50, 0.08],
    [0.98, 0.65, 0.12],
    [1.00, 0.80, 0.20],
    [1.00, 0.90, 0.35],
    [1.00, 0.97, 0.55],
  ],
  // Cool grey → cyan ramp for connection lines, deliberately distinct from the
  // activation colormaps so connections read as a separate visual layer.
  connections: [
    [0.34, 0.38, 0.46],
    [0.30, 0.42, 0.52],
    [0.26, 0.46, 0.58],
    [0.22, 0.52, 0.64],
    [0.20, 0.58, 0.72],
    [0.22, 0.66, 0.82],
    [0.30, 0.76, 0.90],
    [0.46, 0.86, 0.96],
  ],
};

/**
 * Sample a colormap at t∈[0,1].
 * @param {string} name - colormap name
 * @param {number} t - value 0-1
 * @returns {[number, number, number]} RGB tuple 0-1
 */
function sampleColormap(name, t) {
  const map = COLORMAPS[name] || COLORMAPS.viridis;
  t = Math.max(0, Math.min(1, t));
  const idx = t * (map.length - 1);
  const i = Math.floor(idx);
  const f = idx - i;
  if (i >= map.length - 1) return map[map.length - 1];
  const a = map[i];
  const b = map[i + 1];
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];
}

/** Linear-interpolated percentile (p in 0-1) of an unsorted numeric array. */
function _percentile(values, p) {
  if (!values.length) return NaN;
  const a = values.slice().sort((x, y) => x - y);
  const idx = p * (a.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return a[lo];
  return a[lo] + (a[hi] - a[lo]) * (idx - lo);
}

// ── Constants ────────────────────────────────────────────────────────────

// Global colour-scale range is taken as the interquartile range of the
// per-layer raw min/max rather than the absolute min/max. A handful of
// extreme intermediate layers (variance x² `Pow`, mean-subtracted `Sub`)
// otherwise stretch the global range so far that no normal activation layer
// ever reads as bright. The IQR matches the *typical* activation scale; the
// few outlier layers simply saturate at the colormap ends.
const GLOBAL_RANGE_LO_PCT = 0.25;
const GLOBAL_RANGE_HI_PCT = 0.75;

const SLAB_SIZE = 20.0;               // XY footprint of the LARGEST layer (cube side)
const SLAB_MIN_SIDE = 0.4;            // floor so the tiniest layer stays visible
const SIZE_EXP = 0.2;                 // power-law compression of layer size → slab side
const POINT_FILL_DEFAULT = 0.85;      // point diameter as a fraction of cell spacing
const POINT_FILL_MIN = 0.02;
const POINT_FILL_MAX = 2.0;
const SLICE_SPACING = 0.2;            // fixed Z gap between layers
const MAX_POINTS_PER_LAYER_DEFAULT = 6000;
const GLOBAL_POINT_CAP = 400_000;     // floor on total rendered points (low-density safety)
const GLOBAL_CAP_PER_DENSITY = 100;   // cap scales with density so max → raw cloud
const PLAYHEAD_HIGHLIGHT_RANGE = 2.0; // focus window radius (in layers)
const FOCUS_DIM_OPACITY = 0.06;       // opacity for layers outside the focus window
const HOVER_THRESHOLD = 0.45;         // world-space raycast radius for point hover
const PIVOT_TWEEN_MS = 600;           // smooth pivot transition duration
// ── Connection visualization ──
const CONN_SEGMENTS = 12;             // line segments per connection arc
const CONN_R_BASE = 1.5;              // min bow radius (world units) for an arc
const CONN_R_PER_SPAN = 0.06;         // extra bow per layer span (skip = wide arc)
const CONN_R_MAX = 9.0;               // cap so long skips stay inside the cube
const CONN_BASE_OPACITY = 0.32;       // default line opacity (cube stays readable)
const CONN_DIM_OPACITY = 0.06;        // non-incident lines during highlight
const CONN_HIGHLIGHT_OPACITY = 0.95;  // incident lines during highlight
const CONN_OUTLINE_SELECTED = 0x5fd4ff; // accent for the clicked slab
const CONN_OUTLINE_CONNECTED = 0x2f6f8f; // secondary outline for connected slabs

// ── Viewer3D Class ───────────────────────────────────────────────────────

class Viewer3D {
  constructor() {
    this._initialized = false;
    this._frames = [];            // frame metadata array
    this._grids = new Map();      // frame_idx → { grid, rawMin, rawMax, eligible }
    this._pointsObjects = [];     // THREE.Points per frame
    this._frameGroup = null;
    this._playhead = 0;           // float position (frame index)
    this._isPlaying = false;
    this._playSpeed = 1.0;
    this._loopEnabled = true;
    this._colormap = 'viridis';
    this._pointSize = POINT_FILL_DEFAULT;   // point diameter = fill × cell spacing
    this._sliceSpacing = SLICE_SPACING;
    this._maxPointsPerLayer = MAX_POINTS_PER_LAYER_DEFAULT;
    this._sizeMetric = 'neurons';     // 'neurons' | 'spatial' | 'channels'
    this._sizeExp = SIZE_EXP;         // power-law compression of layer size
    this._sizeMax = 1;                // max layer-size value (for normalization)
    this._colorScale = 'per-layer';   // 'per-layer' | 'global'
    this._globalMin = 0;
    this._globalMax = 1;
    this._focusActive = false;        // when false → static cube (all bright)
    this._hideUpcoming = false;       // hide layers ahead of the playhead (no occlusion)
    this._activeFilter = null;
    this._visibleIndices = [];
    // Per-layer hotspot threshold (0-1): each layer's grid already stores a
    // per-layer-normalized brightness (grid[r][c], 0-1). When > 0, points whose
    // grid value is below this fraction are sent off-screen (see
    // _applyAllEnergyMasks), revealing each layer's hotspots independently of
    // absolute magnitude. Independent of layer visibility (setFilter).
    this._energyThreshold = 0;
    this._energyMaskRaf = null;
    this._lastFrameIdx = -1;
    this._onFrameChangeCallback = null;
    this._onStatusChangeCallback = null;
    // Hover tooltip (layer name under the cursor)
    this._raycaster = null;
    this._pointer = null;             // NDC coords
    this._pointerPx = { x: 0, y: 0 }; // pixel coords (relative to canvas)
    this._tooltipEl = null;
    this._hoverPending = false;
    // Smooth pivot transition (animated orbit target)
    this._pivotTween = null;
    // ── Connection visualization (dataflow edges between layers) ──
    this._graphNodes = null;          // nodes[] from /api/load-model (inputs/outputs)
    this._edges = null;               // [{ from, to, tensor, span }] (built lazily)
    this._outEdges = null;            // frameIdx -> [edge]
    this._inEdges = null;             // frameIdx -> [edge]
    this._connLines = null;           // THREE.LineSegments (arcs)
    this._connEnabled = false;        // show connections toggle
    this._connEnergyMin = 0;          // for energy normalization
    this._connEnergyMax = 1;
    this._highlightFrame = null;      // frameIdx of the clicked layer (or null)
    this._highlightSet = null;        // Set of frameIdx kept bright (selected + 1-hop)
    this._highlightGroup = null;      // THREE.Group of slab outlines
  }

  // ── Initialization ─────────────────────────────────────────────────────

  init(containerId = 'canvas-container') {
    if (this._initialized) return;

    const container = document.getElementById(containerId);
    if (!container) {
      console.error('Viewer3D: container element not found:', containerId);
      return;
    }

    this._container = container;

    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0x0a0a0a);

    const w = container.clientWidth || 800;
    const h = container.clientHeight || 600;
    this._camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 5000);
    this._camera.position.set(0, 0, 80);

    this._renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this._renderer.setSize(w, h);
    this._renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this._renderer.domElement);
    this._renderer.domElement.style.display = 'block';

    this._controls = new OrbitControls(this._camera, this._renderer.domElement);
    this._controls.enableDamping = true;
    this._controls.dampingFactor = 0.08;
    this._controls.target.set(0, 0, 0);
    // Allow translating the camera (pan), not just orbiting the center.
    this._controls.enablePan = true;
    this._controls.screenSpacePanning = true;
    this._controls.minDistance = 2;
    this._controls.maxDistance = 400;

    this._frameGroup = new THREE.Group();
    this._scene.add(this._frameGroup);

    // Hover tooltip: a plain absolutely-positioned div (no scene labels).
    this._tooltipEl = document.createElement('div');
    this._tooltipEl.className = 'layer-tooltip';
    this._tooltipEl.style.display = 'none';
    container.appendChild(this._tooltipEl);

    // Raycasting for point hover.
    this._raycaster = new THREE.Raycaster();
    this._raycaster.params.Points.threshold = HOVER_THRESHOLD;
    this._pointer = new THREE.Vector2();

    this._pointerMoveHandler = (e) => this._onPointerMove(e);
    this._pointerLeaveHandler = () => this._hideTooltip();
    this._renderer.domElement.addEventListener('pointermove', this._pointerMoveHandler);
    this._renderer.domElement.addEventListener('pointerleave', this._pointerLeaveHandler);

    // Click-to-inspect a layer: distinguish a click from an orbit drag by
    // tracking the pointer-down position and requiring near-zero movement.
    this._pointerDownPx = null;
    this._pointerDownHandler = (e) => { this._pointerDownPx = { x: e.clientX, y: e.clientY }; };
    this._pointerUpHandler = (e) => this._onClick(e);
    this._renderer.domElement.addEventListener('pointerdown', this._pointerDownHandler);
    this._renderer.domElement.addEventListener('pointerup', this._pointerUpHandler);

    this._resizeHandler = () => this._onResize();
    window.addEventListener('resize', this._resizeHandler);

    this._initialized = true;
    this._animate();

    this._setStatus('Ready');
  }

  _onResize() {
    if (!this._container) return;
    const w = this._container.clientWidth;
    const h = this._container.clientHeight;
    if (w === 0 || h === 0) return;
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
    this._renderer.setSize(w, h);
  }

  // ── Frame Loading ──────────────────────────────────────────────────────

  /**
   * Load all layer grids and build the activation cube.
   * @param {Array} frameMetadata - from /api/inference (includes raw_min/raw_max)
   */
  async loadFrames(frameMetadata) {
    if (!this._initialized) this.init();

    this.clearFrames();
    this._frames = frameMetadata;
    this._grids.clear();
    this._visibleIndices = frameMetadata.map((_, i) => i);

    // Frames changed → edges (which map node names → frame indices) are stale.
    this._edges = null;
    this._outEdges = null;
    this._inEdges = null;
    this.clearConnectionHighlight();

    // Cube-friendly Z spacing: keep total depth ≈ slab side.
    const n = frameMetadata.length;
    this._sliceSpacing = SLICE_SPACING;   // fixed layer gap (control removed)

    this._computeGlobalRange();
    this._computeSizeMax();

    // Placeholder point shells (geometry built once grids arrive).
    for (let i = 0; i < n; i++) {
      const z = i * this._sliceSpacing;
      const points = this._createEmptyPoints(z, frameMetadata[i]);
      this._pointsObjects.push(points);
      this._frameGroup.add(points);
    }

    // Fetch every layer's grid in parallel.
    let done = 0;
    await Promise.all(frameMetadata.map((_, i) =>
      this._loadFrameGrid(i).then(() => {
        done += 1;
        this._setStatus(`Loading activations… ${done}/${n}`);
      }),
    ));

    this._buildAllGeometries();
    this._recomputeAllColors();
    this._applyAllOpacities();
    this._autoFrameCamera();

    // If connections were already on, rebuild them against the new frame set.
    if (this._connEnabled) this._rebuildConnections();

    this._setStatus(`${n} layers loaded`);
  }

  /**
   * Fetch grid data for a frame, cache it. Geometry is built separately
   * (after all grids are available, so a global point cap can be enforced).
   */
  async _loadFrameGrid(frameIdx) {
    if (this._grids.has(frameIdx)) return this._grids.get(frameIdx);
    try {
      const resp = await fetch(`/api/frame/${frameIdx}`);
      if (!resp.ok) {
        console.warn(`Failed to fetch frame ${frameIdx}: ${resp.status}`);
        return null;
      }
      const data = await resp.json();
      const rawMin = data.raw_min ?? data.rawMin ?? 0;
      const rawMax = data.raw_max ?? data.rawMax ?? 1;
      const entry = {
        grid: data.grid,
        rawMin,
        rawMax,
        dtype: data.dtype,
        // Mirrors _eligibleForGlobal (computed from metadata at load time);
        // stored per-entry so _cellT can fall back to per-layer for
        // integer / extreme-valued layers in global mode.
        eligible: this._eligibleForGlobal({ dtype: data.dtype, raw_min: rawMin, raw_max: rawMax }),
      };
      this._grids.set(frameIdx, entry);
      return entry;
    } catch (e) {
      console.warn(`Error fetching frame ${frameIdx}:`, e);
      return null;
    }
  }

  /**
   * A layer contributes to the global colour scale only if it is a
   * floating-point activation tensor with finite, in-band raw min/max.
   * Integer/shape tensors (dtype int64, etc.) carry sentinel extremes
   * (e.g. ±9.2e18) that would otherwise swamp the global range and map
   * every real activation to the colormap midpoint. Such layers fall back
   * to per-layer normalization in global mode (see _cellT).
   */
  _eligibleForGlobal(m) {
    if (!m) return false;
    const dt = String(m.dtype || '').toLowerCase();
    if (!/float|bfloat/.test(dt)) return false;
    const lo = m.raw_min;
    const hi = m.raw_max;
    if (lo == null || hi == null) return false;
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return false;
    if (Math.abs(lo) > 1e9 || Math.abs(hi) > 1e9) return false;
    return true;
  }

  /**
   * Compute the global colour-scale range over the currently *visible*
   * (non-hidden) eligible layers only, using the interquartile range of
   * their raw min/max rather than the absolute extremes.
   *
   * Two problems this fixes:
   *  - Filtered-out ("hidden") layers must not count toward the range —
   *    otherwise enabling global scale while a filter is active still lets
   *    invisible outlier layers dim the whole cube.
   *  - A few extreme intermediate layers (variance x² `Pow`, mean-subtracted
   *    `Sub`) sit orders of magnitude above normal activations and, via the
   *    absolute min/max, stretched the range so far that no normal layer ever
   *    read as bright. The IQR matches the typical activation scale; those
   *    outliers simply saturate at the colormap ends (sampleColormap clamps t).
   *
   * With too few eligible layers to form a stable IQR, fall back to the
   * absolute min/max of what is available.
   */
  _computeGlobalRange() {
    const idx = (this._visibleIndices && this._visibleIndices.length)
      ? this._visibleIndices
      : this._frames.map((_, i) => i);
    const mins = [];
    const maxs = [];
    for (const i of idx) {
      const m = this._frames[i];
      if (!m || !this._eligibleForGlobal(m)) continue;
      mins.push(m.raw_min);
      maxs.push(m.raw_max);
    }
    let gmin, gmax;
    if (mins.length >= 4) {
      gmin = _percentile(mins, GLOBAL_RANGE_LO_PCT);
      gmax = _percentile(maxs, GLOBAL_RANGE_HI_PCT);
    } else {
      gmin = mins.length ? Math.min(...mins) : 0;
      gmax = maxs.length ? Math.max(...maxs) : 1;
    }
    if (!isFinite(gmin) || !isFinite(gmax) || gmax - gmin < 1e-8) {
      gmin = 0;
      gmax = 1;
    }
    this._globalMin = gmin;
    this._globalMax = gmax;
  }

  /** Map a per-layer-normalized brightness (0-1) to the colour-scale t. */
  _cellT(b, entry) {
    if (this._colorScale === 'global' && entry && entry.eligible) {
      const raw = entry.rawMin + b * (entry.rawMax - entry.rawMin);
      return (raw - this._globalMin) / (this._globalMax - this._globalMin + 1e-12);
    }
    // Per-layer mode, or a non-eligible layer in global mode → use the
    // layer's own normalized value directly.
    return b;
  }

  // ── Layer size → slab footprint ────────────────────────────────────────
  //
  // Each slab's XY side scales with a size metric so the cube shows the
  // network's shape — e.g. the hourglass pinch of an encoder/decoder or the
  // hidden-dim bottleneck of an LSTM. The largest layer fills SLAB_SIZE;
  // others scale down under a power-law compression (_sizeExp) so the dynamic
  // range (often 10^3–10^7) stays visible.

  /** A layer's size on the selected metric, from its original tensor shape. */
  _layerSizeValue(meta) {
    const osh = meta.original_shape || meta.shape || [];
    // Drop a leading batch dim (N) when present.
    const dims = (osh.length >= 2 && osh[0] <= 4) ? osh.slice(1) : osh.slice();
    if (dims.length === 0) return 1;
    if (this._sizeMetric === 'channels') return dims[0];
    if (this._sizeMetric === 'spatial') {
      if (dims.length >= 3) return dims[dims.length - 2] * dims[dims.length - 1];
      if (dims.length === 2) return dims[1];
      return 1; // 1D has no spatial extent
    }
    // 'neurons' → total activation count.
    let p = 1;
    for (const d of dims) p *= d;
    return p;
  }

  _computeSizeMax() {
    let mx = 0;
    for (const m of this._frames) mx = Math.max(mx, this._layerSizeValue(m));
    this._sizeMax = mx > 0 ? mx : 1;
  }

  /** XY side (world units) for a layer's slab. */
  _slabSideFor(meta) {
    const v = this._layerSizeValue(meta);
    const r = this._sizeMax > 0 ? v / this._sizeMax : 0;
    return Math.max(SLAB_MIN_SIDE, SLAB_SIZE * Math.pow(r, this._sizeExp));
  }

  /** Recompute every layer's slab side (after metric/exponent changes). */
  _recomputeSlabSides() {
    for (const points of this._pointsObjects) {
      if (!points || !points.userData) continue;
      points.userData.slabSide = this._slabSideFor(points.userData.meta);
    }
  }

  _createEmptyPoints(z, meta) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, z]), 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array([0.3, 0.3, 0.3]), 3));

    const material = new THREE.PointsMaterial({
      size: this._pointSize,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 1,
      depthWrite: true,
    });

    const points = new THREE.Points(geometry, material);
    points.userData = {
      frameIdx: meta.frame_idx, meta, z, loaded: false,
      slabSide: this._slabSideFor(meta),
    };
    // Hidden until _applyAllOpacities() runs at the end of loadFrames —
    // avoids a grid of blank placeholder planes while grids stream in.
    points.visible = false;
    return points;
  }

  /**
   * Build all layer geometries, enforcing a global point cap. The cap scales
   * with the density setting so a low density stays smooth (floor cap) while
   * max density can render the full raw cloud (cap = density × N layers).
   */
  _buildAllGeometries() {
    const sizes = this._frames.map((_, i) => {
      const e = this._grids.get(i);
      return e && e.grid && e.grid.length ? e.grid.length * e.grid[0].length : 0;
    });

    const globalCap = Math.max(GLOBAL_POINT_CAP, this._maxPointsPerLayer * GLOBAL_CAP_PER_DENSITY);
    let budget = this._maxPointsPerLayer;
    const totalAtBudget = sizes.reduce((a, s) => a + Math.min(s, budget), 0);
    if (totalAtBudget > globalCap) {
      budget = Math.max(50, Math.floor((budget * globalCap) / totalAtBudget));
    }

    for (let i = 0; i < this._frames.length; i++) {
      this._buildPointsGeometry(i, budget);
    }
    // Geometry rebuild (e.g. density change) discards userData.basePositions/
    // energyValues, which _buildPointsGeometry re-stores; re-apply the current
    // hotspot mask so it survives a rebuild.
    this._applyAllEnergyMasks();
  }

  /**
   * Build a layer's point geometry as a square slab.
   * @param {number} frameIdx
   * @param {number} budget - max points for this layer
   */
  _buildPointsGeometry(frameIdx, budget) {
    const entry = this._grids.get(frameIdx);
    const points = this._pointsObjects[frameIdx];
    if (!points || !entry || !entry.grid || !entry.grid.length) return;

    const grid = entry.grid;
    const rows = grid.length;
    const cols = grid[0].length;
    const total = rows * cols;

    let stride = 1;
    if (total > budget) stride = Math.ceil(Math.sqrt(total / budget));

    const positions = [];
    // Per-vertex per-layer-normalized brightness (grid[r][c], 0-1) captured in
    // the same vertex order as positions — drives per-point energy masking
    // (_applyAllEnergyMasks) so each layer can hide its low-activation points
    // and reveal its own hotspots.
    const energy = [];
    const z = points.userData.z;
    const side = points.userData.slabSide;
    const sx = cols > 1 ? side / (cols - 1) : 0;
    const sy = rows > 1 ? side / (rows - 1) : 0;
    const ox = -side / 2;
    // Row 0 is the TOP of the feature map; place it at +Y (top) and walk
    // downward so slabs aren't rendered upside-down. The colour loop in
    // _recomputeLayerColors iterates (r,c) in the same vertex order, so
    // flipping only the Y here keeps each vertex's colour matched to its cell.
    const oy = side / 2;

    for (let r = 0; r < rows; r += stride) {
      for (let c = 0; c < cols; c += stride) {
        positions.push(ox + c * sx, oy - r * sy, z);
        energy.push(grid[r][c]);
      }
    }

    points.geometry.dispose();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    // Colour attribute filled by _recomputeLayerColors (same vertex order).
    geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(positions.length), 3));
    points.geometry = geometry;
    // Effective point spacing (accounts for stride) → drives fill sizing.
    const cellSpacing = Math.min(sx * stride, sy * stride) || side / 8;
    points.userData.loaded = true;
    points.userData.sampled = stride > 1;
    points.userData.gridDims = { rows, cols, stride };
    points.userData.cellSpacing = cellSpacing;
    // Original (unmasked) positions + per-vertex energy for hotspot masking.
    points.userData.basePositions = new Float32Array(positions);
    points.userData.energyValues = new Float32Array(energy);
    points.material.size = this._pointSizeForCell(cellSpacing);
    // Pre-compute the bounding sphere from the unmasked positions so frustum
    // culling uses the real (small) extent; masking only moves points outward
    // to a sentinel, so this sphere still bounds the surviving hotspots.
    geometry.computeBoundingSphere();
  }

  /**
   * World-space point diameter for a layer = fill fraction × cell spacing,
   * but never wider than the layer gap. Without that cap, large or coarsely
   * subsampled slabs produce huge billboards that visually smear across
   * several Z-layers so the cube reads as a blob instead of distinct slabs.
   * Raising the Layer Gap relaxes the cap, letting slabs fill more.
   */
  _pointSizeForCell(cellSpacing) {
    const cap = Math.max(0.02, this._sliceSpacing * 0.9);
    return Math.max(0.02, Math.min(this._pointSize * cellSpacing, cap));
  }

  /** Reapply per-layer point sizes after the fill fraction changes. */
  _applyPointSizes() {
    for (const points of this._pointsObjects) {
      if (!points || !points.userData) continue;
      const cell = points.userData.cellSpacing;
      if (cell != null) points.material.size = this._pointSizeForCell(cell);
    }
  }

  // ── Colour & Opacity ───────────────────────────────────────────────────

  /** Recompute the colour buffer for one layer from its grid. */
  _recomputeLayerColors(frameIdx) {
    const points = this._pointsObjects[frameIdx];
    if (!points || !points.userData.loaded) return;
    const entry = this._grids.get(frameIdx);
    if (!entry) return;
    const { rows, cols, stride } = points.userData.gridDims;
    const grid = entry.grid;
    const colorAttr = points.geometry.attributes.color;
    if (!colorAttr) return;
    const arr = colorAttr.array;
    let ci = 0;
    for (let r = 0; r < rows; r += stride) {
      for (let c = 0; c < cols; c += stride) {
        const t = this._cellT(grid[r][c], entry);
        const [cr, cg, cb] = sampleColormap(this._colormap, t);
        arr[ci] = cr; arr[ci + 1] = cg; arr[ci + 2] = cb;
        ci += 3;
      }
    }
    colorAttr.needsUpdate = true;
  }

  _recomputeAllColors() {
    for (let i = 0; i < this._pointsObjects.length; i++) {
      this._recomputeLayerColors(i);
    }
  }

  /** Apply visibility + focus opacity to one layer. Cheap — runs each frame. */
  _applyLayerOpacity(frameIdx) {
    const points = this._pointsObjects[frameIdx];
    if (!points) return;
    const visible = this._visibleIndices.includes(frameIdx);
    if (!visible) { points.visible = false; return; }

    // During playback/scrub, hide layers ahead of the playhead so the
    // current layer isn't occluded by the (closer, higher-Z) upcoming slabs.
    // Past layers stay visible — they sit behind the current one and don't
    // occlude it; the focus sweep dims them as context. Gated on
    // _focusActive so Stop returns the full static cube.
    if (this._hideUpcoming && this._focusActive && frameIdx > this._playhead) {
      points.visible = false;
      return;
    }
    points.visible = true;

    let opacity = 1;
    if (this._focusActive) {
      const dist = Math.abs(frameIdx - this._playhead);
      const hi = Math.max(0, 1 - dist / PLAYHEAD_HIGHLIGHT_RANGE);
      opacity = FOCUS_DIM_OPACITY + (1 - FOCUS_DIM_OPACITY) * hi;
    }
    // Connection highlight takes precedence: keep the selected + 1-hop slabs at
    // full opacity, dim the rest as context.
    if (this._highlightActive()) {
      opacity = this._highlightSet && this._highlightSet.has(frameIdx)
        ? 1
        : Math.min(opacity, FOCUS_DIM_OPACITY);
    }
    points.material.opacity = opacity;
    points.material.depthWrite = opacity > 0.5;   // dimmed layers shouldn't occlude
  }

  _applyAllOpacities() {
    for (let i = 0; i < this._pointsObjects.length; i++) {
      this._applyLayerOpacity(i);
    }
  }

  // ── Per-layer energy (hotspot) masking ──────────────────────────────────
  //
  // PointsMaterial honours a uniform per-layer opacity, not per-vertex alpha,
  // so hiding individual points means moving them off-screen. Each layer's
  // grid already stores a per-layer-normalized brightness (energyValues ==
  // grid[r][c], 0-1); points whose value is below the threshold are sent to a
  // far off-screen sentinel (1e9,1e9,1e9) where the GPU clips them before
  // rasterization. The survivors are each layer's hotspots, independent of
  // absolute magnitude and of layer visibility (setFilter). This is orthogonal
  // to _applyLayerOpacity: a layer can be visible-but-dimmed while its
  // low-energy points are masked.

  /**
   * Set the per-layer hotspot threshold (0-1). 0 = no masking (show all
   * points). Debounced via requestAnimationFrame so continuous slider drags
   * coalesce into one position-buffer pass per frame.
   * @param {number} t threshold in [0,1]
   */
  setEnergyThreshold(t) {
    const v = Math.max(0, Math.min(1, Number(t) || 0));
    if (v === this._energyThreshold) return;
    this._energyThreshold = v;
    if (this._energyMaskRaf) cancelAnimationFrame(this._energyMaskRaf);
    this._energyMaskRaf = requestAnimationFrame(() => {
      this._energyMaskRaf = null;
      this._applyAllEnergyMasks();
    });
  }

  /** Rewrite every loaded layer's positions to apply the current threshold. */
  _applyAllEnergyMasks() {
    const t = this._energyThreshold;
    const SENT = 1e9;
    for (const points of this._pointsObjects) {
      if (!points || !points.userData || !points.userData.loaded) continue;
      const base = points.userData.basePositions;
      const ev = points.userData.energyValues;
      const posAttr = points.geometry && points.geometry.attributes.position;
      if (!base || !ev || !posAttr) continue;
      const pos = posAttr.array;
      const n = ev.length;
      if (t <= 0) {
        // Restore every point to its base position.
        if (pos !== base) pos.set(base);
      } else {
        for (let i = 0; i < n; i++) {
          const k = i * 3;
          if (ev[i] >= t) {
            pos[k] = base[k];
            pos[k + 1] = base[k + 1];
            pos[k + 2] = base[k + 2];
          } else {
            pos[k] = SENT;
            pos[k + 1] = SENT;
            pos[k + 2] = SENT;
          }
        }
      }
      posAttr.needsUpdate = true;
      // NOTE: deliberately do NOT null geometry.boundingSphere. The sphere is
      // computed from basePositions at build time (the unmasked extent); masked
      // points only move *outward* to a sentinel, so the surviving points stay
      // within that sphere and frustum culling stays correct. Nulling it would
      // force a recompute that includes the 1e9 sentinels → a giant sphere and
      // no culling.
    }
  }

  /**
   * Total count of on-screen (non-sentinel) vertices across loaded layers.
   * Debug/verification helper for the headless probe.
   * @returns {number}
   */
  countVisiblePoints() {
    let count = 0;
    for (const points of this._pointsObjects) {
      if (!points || !points.userData || !points.userData.loaded) continue;
      const posAttr = points.geometry && points.geometry.attributes.position;
      if (!posAttr) continue;
      const pos = posAttr.array;
      for (let i = 0; i < pos.length; i += 3) {
        if (Math.abs(pos[i]) < 1e6) count++;
      }
    }
    return count;
  }

  // ── Playback ───────────────────────────────────────────────────────────

  play() {
    if (this._frames.length === 0) return;
    this._isPlaying = true;
    this._focusActive = true;
    this._applyAllOpacities();
    this._setStatus('Playing');
  }

  pause() {
    this._isPlaying = false;
    this._setStatus('Paused');
  }

  stop() {
    this._isPlaying = false;
    this._playhead = 0;
    this._lastFrameIdx = -1;
    this._focusActive = false;        // back to the full static cube
    this._applyAllOpacities();
    this._notifyFrameChange(0);
    this._setStatus('Stopped');
  }

  rewind() {
    this._playhead = 0;
    this._lastFrameIdx = -1;
    this._applyAllOpacities();
    this._notifyFrameChange(0);
  }

  setFrame(frameIdx) {
    frameIdx = Math.max(0, Math.min(this._frames.length - 1, frameIdx));
    this._playhead = frameIdx;
    this._lastFrameIdx = -1;
    this._focusActive = true;
    this._loadFrameGrid(frameIdx).then(() => this._applyAllOpacities());
    this._notifyFrameChange(frameIdx);
  }

  stepForward() {
    const next = Math.floor(this._playhead) + 1;
    if (next < this._frames.length) this.setFrame(next);
  }

  stepBackward() {
    const prev = Math.floor(this._playhead) - 1;
    if (prev >= 0) this.setFrame(prev);
  }

  setSpeed(speed) { this._playSpeed = speed; }
  setLoop(enabled) { this._loopEnabled = enabled; }

  /** Engage/disengage the focus sweep (called by transport controls). */
  setFocusActive(active) {
    this._focusActive = !!active;
    this._applyAllOpacities();
  }

  /** Toggle hiding of layers ahead of the playhead during playback/scrub. */
  setHideUpcoming(enabled) {
    this._hideUpcoming = !!enabled;
    this._applyAllOpacities();
  }

  // ── Connection visualization ─────────────────────────────────────────
  // Dataflow edges (incl. skip connections) between layer slabs, with an
  // activation-energy color and a click-to-highlight mode that brightens the
  // clicked layer's 1-hop neighbourhood and dims the rest.

  /** Store the graph nodes (from /api/load-model) so edges can be derived. */
  setGraphNodes(nodes) {
    this._graphNodes = Array.isArray(nodes) ? nodes : null;
    this._edges = null;       // force rebuild against the new graph
    this._outEdges = null;
    this._inEdges = null;
  }

  _highlightActive() {
    return this._highlightFrame != null;
  }

  /** Derive dataflow edges from node inputs/outputs, mapped to frame indices. */
  _buildEdges() {
    this._edges = [];
    this._outEdges = {};
    this._inEdges = {};
    if (!this._graphNodes || !this._frames.length) return;
    // tensor name → producing node name (first producer wins; ONNX tensor
    // names are effectively unique). Inputs with no producer are graph
    // inputs / initializer weights — not layer-to-layer edges.
    const producer = new Map();
    for (const n of this._graphNodes) {
      if (!n || !n.outputs) continue;
      for (const out of n.outputs) {
        if (!producer.has(out)) producer.set(out, n.name);
      }
    }
    for (const b of this._graphNodes) {
      if (!b || !b.inputs) continue;
      const to = this.frameIndexForNode(b.name);
      if (to < 0) continue;
      for (const t of b.inputs) {
        const aName = producer.get(t);
        if (!aName || aName === b.name) continue;
        const from = this.frameIndexForNode(aName);
        if (from < 0) continue;
        const span = to - from;
        if (span < 1) continue;          // DAG: forward edges only
        const edge = { from, to, tensor: t, span };
        this._edges.push(edge);
        (this._outEdges[from] ||= []).push(edge);
        (this._inEdges[to] ||= []).push(edge);
      }
    }
    // Energy normalization range across all source frames.
    let minE = Infinity, maxE = -Infinity;
    for (const e of this._edges) {
      const en = this._frameEnergy(e.from);
      if (en < minE) minE = en;
      if (en > maxE) maxE = en;
    }
    this._connEnergyMin = isFinite(minE) ? minE : 0;
    this._connEnergyMax = isFinite(maxE) && maxE > minE ? maxE : (isFinite(maxE) ? maxE : 1);
  }

  /** Peak activation magnitude of a frame (the "signal energy" on its edges). */
  _frameEnergy(frameIdx) {
    const m = this._frames[frameIdx];
    if (!m) return 0;
    const mn = m.raw_min, mx = m.raw_max;
    const a = mn == null ? 0 : Math.abs(mn);
    const b = mx == null ? 0 : Math.abs(mx);
    return Math.max(a, b);
  }

  /** Ensure edges are built; return them (empty if no graph/fames). */
  _ensureEdges() {
    if (this._edges === null) this._buildEdges();
    return this._edges;
  }

  /** Show or hide the connection arcs. When on, nothing is drawn until a layer
   *  is clicked — only that layer's incident edges are shown (see
   *  highlightConnections), so the cube stays readable instead of filling with
   *  the full 800+ edge graph. */
  setConnectionsEnabled(on) {
    this._connEnabled = !!on;
    if (!this._connEnabled) {
      this._disposeConnLines();
      this.clearConnectionHighlight();
      return;
    }
    this._ensureEdges();                 // build adjacency, draw nothing yet
    if (this._highlightActive()) this._buildIncidentLines(this._highlightFrame);
  }

  /** (Re)build adjacency after a frame reload; draw nothing unless a layer is
   *  already selected, in which case refresh its incident edges. */
  _rebuildConnections() {
    this._disposeConnLines();
    this._ensureEdges();
    if (this._highlightActive()) this._buildIncidentLines(this._highlightFrame);
  }

  /** Draw only the edges incident to `frameIdx` (1-hop in + out), bright. */
  _buildIncidentLines(frameIdx) {
    this._disposeConnLines();
    if (!this._frameGroup) return;
    this._ensureEdges();
    const incident = (this._outEdges[frameIdx] || []).concat(this._inEdges[frameIdx] || []);
    if (!incident.length) return;
    const seg = CONN_SEGMENTS;
    const positions = new Float32Array(incident.length * seg * 2 * 3);
    const colors = new Float32Array(incident.length * seg * 2 * 3);
    let pi = 0, ci = 0;
    for (const e of incident) {
      const [r, g, b] = this._edgeColor(e, true);
      const pts = this._arcPoints(e, seg + 1);
      for (let s = 0; s < seg; s++) {
        const p0 = pts[s], p1 = pts[s + 1];
        positions[pi] = p0.x; positions[pi + 1] = p0.y; positions[pi + 2] = p0.z; pi += 3;
        positions[pi] = p1.x; positions[pi + 1] = p1.y; positions[pi + 2] = p1.z; pi += 3;
        for (let k = 0; k < 2; k++) {
          colors[ci] = r; colors[ci + 1] = g; colors[ci + 2] = b; ci += 3;
        }
      }
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: CONN_HIGHLIGHT_OPACITY,
      depthWrite: false,
    });
    const lines = new THREE.LineSegments(geom, mat);
    lines.renderOrder = 2;
    this._connLines = lines;
    this._frameGroup.add(lines);
  }

  _disposeConnLines() {
    if (this._connLines) {
      this._frameGroup.remove(this._connLines);
      this._connLines.geometry.dispose();
      this._connLines.material.dispose();
      this._connLines = null;
    }
  }

  /** Quadratic Bézier arc points for an edge, bowing out in XY (R ∝ span). */
  _arcPoints(e, count) {
    const zA = e.from * this._sliceSpacing;
    const zB = e.to * this._sliceSpacing;
    const midZ = (zA + zB) / 2;
    const r = Math.min(CONN_R_MAX, CONN_R_BASE + CONN_R_PER_SPAN * e.span);
    const theta = this._hashAngle(e.tensor);
    const cx = Math.cos(theta) * r, cy = Math.sin(theta) * r;
    const p0 = new THREE.Vector3(0, 0, zA);
    const p1 = new THREE.Vector3(cx, cy, midZ);
    const p2 = new THREE.Vector3(0, 0, zB);
    const pts = new Array(count);
    for (let i = 0; i < count; i++) {
      const t = i / (count - 1);
      const u = 1 - t;
      pts[i] = new THREE.Vector3(
        u * u * p0.x + 2 * u * t * p1.x + t * t * p2.x,
        u * u * p0.y + 2 * u * t * p1.y + t * t * p2.y,
        u * u * p0.z + 2 * u * t * p1.z + t * t * p2.z,
      );
    }
    return pts;
  }

  /** Deterministic [0, 2π) angle from a tensor name so each skip bows fixed. */
  _hashAngle(s) {
    let h = 2166136261 >>> 0;
    const str = String(s || '');
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return (h % 1000) / 1000 * Math.PI * 2;
  }

  /** Energy-normalized color for an edge (bright = strong source signal). */
  _edgeColor(e, bright) {
    const en = this._frameEnergy(e.from);
    const range = this._connEnergyMax - this._connEnergyMin;
    const t = range > 1e-12 ? (en - this._connEnergyMin) / range : 0.5;
    const [r, g, b] = sampleColormap('connections', t);
    if (bright) {
      // Push toward white-cyan for highlighted incident edges.
      return [0.5 + 0.5 * r, 0.7 + 0.3 * g, 1.0];
    }
    return [r, g, b];
  }

  /** Highlight a layer + its 1-hop neighbourhood; dim the rest and draw only
   *  the incident edges. */
  highlightConnections(frameIdx) {
    if (this._highlightFrame === frameIdx) {        // toggle off on re-click
      this.clearConnectionHighlight();
      return;
    }
    this._ensureEdges();
    this._highlightFrame = frameIdx;
    const set = new Set([frameIdx]);
    const incident = (this._outEdges[frameIdx] || []).concat(this._inEdges[frameIdx] || []);
    for (const e of incident) { set.add(e.from); set.add(e.to); }
    this._highlightSet = set;
    this._buildHighlightOutlines();
    this._buildIncidentLines(frameIdx);
    this._applyAllOpacities();
  }

  /** Wireframe square outlines around the selected + connected slabs. */
  _buildHighlightOutlines() {
    this._disposeHighlightOutlines();
    if (!this._frameGroup || !this._highlightSet) return;
    const group = new THREE.Group();
    for (const idx of this._highlightSet) {
      const points = this._pointsObjects[idx];
      if (!points) continue;
      const side = points.userData.slabSide || SLAB_MIN_SIDE;
      const z = points.userData.z;
      const h = side / 2;
      const verts = new Float32Array([
        -h, -h, z,  h, -h, z,  h, h, z, -h, h, z,
      ]);
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(verts, 3));
      const color = idx === this._highlightFrame ? CONN_OUTLINE_SELECTED : CONN_OUTLINE_CONNECTED;
      const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false });
      const loop = new THREE.LineLoop(g, m);
      loop.renderOrder = 3;
      group.add(loop);
    }
    this._highlightGroup = group;
    this._frameGroup.add(group);
  }

  _disposeHighlightOutlines() {
    if (this._highlightGroup) {
      this._frameGroup.remove(this._highlightGroup);
      this._highlightGroup.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
      });
      this._highlightGroup = null;
    }
  }

  /** Clear the connection highlight (outlines, dimming, incident edges). */
  clearConnectionHighlight() {
    const wasActive = this._highlightActive();
    this._highlightFrame = null;
    this._highlightSet = null;
    this._disposeHighlightOutlines();
    this._disposeConnLines();          // remove incident edges → clean cube
    if (wasActive) this._applyAllOpacities();
  }

  /** Edges incident to a frame, for the info panel (+ pair-strength fetches). */
  getConnectionInfo(frameIdx) {
    this._ensureEdges();
    const meta = this._frames[frameIdx] || null;
    const out = (this._outEdges[frameIdx] || []).map((e) => ({ ...e, dir: 'out', other: e.to }));
    const inn = (this._inEdges[frameIdx] || []).map((e) => ({ ...e, dir: 'in', other: e.from }));
    return { frameIdx, meta, outEdges: out, inEdges: inn };
  }

  // ── Animation Loop ────────────────────────────────────────────────────

  _animate() {
    requestAnimationFrame(() => this._animate());
    if (!this._initialized) return;

    // Smoothly tween the orbit target to a new pivot (selected layer).
    if (this._pivotTween) {
      const t = Math.min(1, (performance.now() - this._pivotTween.start) / this._pivotTween.duration);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; // easeInOutQuad
      this._controls.target.lerpVectors(this._pivotTween.from, this._pivotTween.to, e);
      if (t >= 1) this._pivotTween = null;
    }

    this._controls.update();

    if (this._isPlaying && this._frames.length > 0) {
      const delta = this._playSpeed * (1 / 30);
      this._playhead += delta;

      if (this._playhead >= this._frames.length) {
        if (this._loopEnabled) {
          this._playhead = 0;
        } else {
          this._playhead = this._frames.length - 1;
          this.pause();
        }
      }

      const currentIdx = Math.floor(this._playhead);
      if (currentIdx !== this._lastFrameIdx) {
        if (!this._grids.has(currentIdx)) this._loadFrameGrid(currentIdx);
        this._notifyFrameChange(currentIdx);
        this._lastFrameIdx = currentIdx;
      }

      this._applyAllOpacities();
    }

    this._renderer.render(this._scene, this._camera);
  }

  _notifyFrameChange(frameIdx) {
    if (this._onFrameChangeCallback) {
      const meta = this._frames[frameIdx];
      if (meta) this._onFrameChangeCallback(frameIdx, meta);
    }
    this._updateFrameInfoBar(frameIdx);
  }

  _updateFrameInfoBar(frameIdx) {
    const meta = this._frames[frameIdx];
    if (!meta) return;
    const $ = (id) => document.getElementById(id);
    const labelEl = $('frame-info-label');
    const opEl = $('frame-info-op');
    const shapeEl = $('frame-info-shape');
    const execEl = $('frame-info-exec');
    const sampledEl = $('frame-info-sampled');
    const uniformEl = $('frame-info-uniform');
    if (labelEl) labelEl.textContent = meta.node_name || '—';
    if (opEl) opEl.textContent = meta.op_type || '';
    if (shapeEl) {
      const shape = meta.original_shape || meta.shape || [];
      shapeEl.textContent = `[${shape.join(', ')}]`;
    }
    if (execEl) execEl.textContent = `#${meta.exec_order ?? frameIdx}`;
    if (sampledEl) sampledEl.hidden = !meta.sampled;
    if (uniformEl) uniformEl.hidden = !meta.is_uniform;
  }

  // ── Filtering ──────────────────────────────────────────────────────────

  setFilter(filter) {
    this._activeFilter = filter;
    if (!filter) {
      this._visibleIndices = this._frames.map((_, i) => i);
    } else if (filter.type === 'op_type') {
      this._visibleIndices = this._frames
        .map((m, i) => (m.op_type === filter.value ? i : -1)).filter((i) => i >= 0);
    } else if (filter.type === 'node') {
      this._visibleIndices = this._frames
        .map((m, i) => (m.node_name === filter.value ? i : -1)).filter((i) => i >= 0);
    } else if (filter.type === 'range') {
      const [start, end] = filter.value;
      this._visibleIndices = [];
      for (let i = Math.max(0, start); i <= Math.min(this._frames.length - 1, end); i++) {
        this._visibleIndices.push(i);
      }
    } else if (filter.type === 'multi_node') {
      const names = new Set(filter.value);
      this._visibleIndices = this._frames
        .map((m, i) => (names.has(m.node_name) ? i : -1)).filter((i) => i >= 0);
    }
    this._applyAllOpacities();
    // Hidden layers must not count toward the global colour range, so
    // recompute it over the now-visible set and refresh global-mode colours.
    this._computeGlobalRange();
    if (this._colorScale === 'global') this._recomputeAllColors();
    // No status-pill update: the filter is already obvious from the cube and
    // the tree checkboxes, and the "Filter: multi_node=…" string was ugly in
    // the header. Leave the status pill on its prior message.
  }

  getOpTypes() {
    const types = new Set();
    this._frames.forEach((m) => types.add(m.op_type));
    return [...types].sort();
  }

  // ── Configuration ─────────────────────────────────────────────────────

  setColormap(name) {
    if (!COLORMAPS[name]) { console.warn(`Unknown colormap: ${name}`); return; }
    this._colormap = name;
    this._recomputeAllColors();
  }

  setPointSize(fill) {
    this._pointSize = Math.max(POINT_FILL_MIN, Math.min(POINT_FILL_MAX, fill));
    this._applyPointSizes();
  }

  /** Set max points per layer; rebuilds geometries (debounce upstream). */
  setDensity(maxPointsPerLayer) {
    this._maxPointsPerLayer = Math.max(50, Math.floor(maxPointsPerLayer));
    this._buildAllGeometries();
    this._recomputeAllColors();
    this._applyAllOpacities();
  }

  /** Switch colour scale: 'per-layer' or 'global'. */
  setColorScale(mode) {
    if (mode !== 'per-layer' && mode !== 'global') return;
    this._colorScale = mode;
    // Range reflects the current (possibly filtered) visible set.
    this._computeGlobalRange();
    this._recomputeAllColors();
  }

  /**
   * Switch the metric that sets each slab's XY footprint:
   *   'neurons'  – total element count (default; shows encoder→decoder volume pinch)
   *   'spatial'  – H×W spatial footprint (channel collapse)
   *   'channels' – leading channel/feature count
   * Recomputes slab sides then rebuilds every layer's geometry.
   */
  setSizeMetric(metric) {
    if (metric !== 'neurons' && metric !== 'spatial' && metric !== 'channels') return;
    if (this._sizeMetric === metric) return;
    this._sizeMetric = metric;
    this._computeSizeMax();
    this._recomputeSlabSides();
    this._buildAllGeometries();
    this._recomputeAllColors();
    this._applyAllOpacities();
  }

  // ── Callbacks ─────────────────────────────────────────────────────────

  onFrameChange(callback) { this._onFrameChangeCallback = callback; }
  onStatusChange(callback) { this._onStatusChangeCallback = callback; }
  _setStatus(msg) { if (this._onStatusChangeCallback) this._onStatusChangeCallback(msg); }

  // ── Cleanup ───────────────────────────────────────────────────────────

  clearFrames() {
    for (const points of this._pointsObjects) {
      this._frameGroup.remove(points);
      points.geometry.dispose();
      points.material.dispose();
    }
    this._pointsObjects = [];
    this._frames = [];
    this._grids.clear();
    this._playhead = 0;
    this._lastFrameIdx = -1;
    this._focusActive = false;
    this._visibleIndices = [];
    this._pivotTween = null;
    // Reset the hotspot mask so a freshly loaded model starts unmasked (the
    // tree UI also resets its slider, but guard against a bare reload too).
    if (this._energyMaskRaf) { cancelAnimationFrame(this._energyMaskRaf); this._energyMaskRaf = null; }
    this._energyThreshold = 0;
    this._hideTooltip();
  }

  dispose() {
    this.clearFrames();
    if (this._resizeHandler) window.removeEventListener('resize', this._resizeHandler);
    if (this._pointerMoveHandler && this._renderer) {
      this._renderer.domElement.removeEventListener('pointermove', this._pointerMoveHandler);
      this._renderer.domElement.removeEventListener('pointerleave', this._pointerLeaveHandler);
    }
    if (this._tooltipEl && this._tooltipEl.parentNode) {
      this._tooltipEl.parentNode.removeChild(this._tooltipEl);
    }
    if (this._renderer) {
      this._renderer.dispose();
      if (this._renderer.domElement.parentNode) {
        this._renderer.domElement.parentNode.removeChild(this._renderer.domElement);
      }
    }
    this._initialized = false;
  }

  // ── Pivot & Hover ──────────────────────────────────────────────────────

  /**
   * Find the frame index (rank) for a graph node name. Frames are ranked by
   * exec_order, which is NOT the same as the node's exec_order (non-
   * intermediate nodes and multi-output nodes shift the rank), so look up by
   * node_name. Returns the first match or -1.
   */
  frameIndexForNode(name) {
    for (let i = 0; i < this._frames.length; i++) {
      if (this._frames[i].node_name === name) return i;
    }
    return -1;
  }

  /**
   * Smoothly move the orbit pivot to a selected layer so the camera orbits
   * around it. Called when a tree node is selected.
   * @returns {boolean} true if the layer was found and a tween started.
   */
  focusPivot(nodeName) {
    if (!this._controls || !this._frames.length) return false;
    const idx = this.frameIndexForNode(nodeName);
    if (idx < 0) return false;
    const z = idx * this._sliceSpacing;
    this._pivotTween = {
      from: this._controls.target.clone(),
      to: new THREE.Vector3(0, 0, z),
      start: performance.now(),
      duration: PIVOT_TWEEN_MS,
    };
    return true;
  }

  /**
   * Smoothly move the orbit pivot to a layer by frame index (used when
   * navigating from a connector row in the info panel). Returns true if a
   * tween was started.
   */
  centerOnLayer(frameIdx) {
    if (!this._controls || !this._frames.length) return false;
    if (frameIdx == null || frameIdx < 0 || frameIdx >= this._frames.length) return false;
    const z = frameIdx * this._sliceSpacing;
    this._pivotTween = {
      from: this._controls.target.clone(),
      to: new THREE.Vector3(0, 0, z),
      start: performance.now(),
      duration: PIVOT_TWEEN_MS,
    };
    return true;
  }

  _onPointerMove(e) {
    if (!this._frames.length || !this._raycaster) return;
    const rect = this._renderer.domElement.getBoundingClientRect();
    this._pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this._pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this._pointerPx.x = e.clientX - rect.left;
    this._pointerPx.y = e.clientY - rect.top;
    // Throttle to one raycast per animation frame.
    if (!this._hoverPending) {
      this._hoverPending = true;
      requestAnimationFrame(() => { this._updateHover(); this._hoverPending = false; });
    }
  }

  _updateHover() {
    if (!this._frames.length || !this._raycaster || !this._camera) {
      this._hideTooltip();
      return;
    }
    this._raycaster.setFromCamera(this._pointer, this._camera);
    const objs = this._pointsObjects.filter((p) => p.visible && p.userData.loaded);
    const hits = this._raycaster.intersectObjects(objs, false);
    if (hits.length) {
      const meta = hits[0].object.userData.meta;
      if (meta) { this._showTooltip(meta, this._pointerPx.x, this._pointerPx.y); return; }
    }
    this._hideTooltip();
  }

  _showTooltip(meta, x, y) {
    if (!this._tooltipEl) return;
    const shape = meta.original_shape || meta.shape || [];
    this._tooltipEl.innerHTML =
      `<span class="lt-name">${meta.node_name || '—'}</span>` +
      `<span class="lt-op">${meta.op_type || ''}</span>` +
      `<span class="lt-shape">[${shape.join(', ')}]</span>`;
    // Keep the tooltip inside the container.
    const w = this._container.clientWidth;
    const h = this._container.clientHeight;
    const tw = this._tooltipEl.offsetWidth || 160;
    const th = this._tooltipEl.offsetHeight || 24;
    const px = Math.min(x + 14, w - tw - 4);
    const py = Math.min(y + 14, h - th - 4);
    this._tooltipEl.style.left = Math.max(4, px) + 'px';
    this._tooltipEl.style.top = Math.max(4, py) + 'px';
    this._tooltipEl.style.display = 'block';
  }

  _hideTooltip() {
    if (this._tooltipEl) this._tooltipEl.style.display = 'none';
  }

  /** Handle a canvas click (not a drag): raycast the topmost visible layer and
   *  dispatch a `viewer:layer-click` event with its frame index + metadata.
   *  Does not move the playhead — purely an inspector hook for the heatmap
   *  popup. */
  _onClick(e) {
    if (!this._frames.length || !this._raycaster || !this._camera) return;
    if (!this._pointerDownPx) return;
    const dx = e.clientX - this._pointerDownPx.x;
    const dy = e.clientY - this._pointerDownPx.y;
    this._pointerDownPx = null;
    // Ignore drags (orbit/pan) — only treat near-stationary taps as clicks.
    if (dx * dx + dy * dy > 25) return;

    const rect = this._renderer.domElement.getBoundingClientRect();
    this._pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this._pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this._raycaster.setFromCamera(this._pointer, this._camera);
    const objs = this._pointsObjects.filter((p) => p.visible && p.userData.loaded);
    const hits = this._raycaster.intersectObjects(objs, false);
    if (!hits.length) return;
    const obj = hits[0].object;
    const frameIdx = obj.userData.frameIdx;
    const meta = obj.userData.meta;
    if (frameIdx == null) return;
    document.dispatchEvent(new CustomEvent('viewer:layer-click', {
      detail: { frameIdx, meta: meta || this._frames[frameIdx] || null },
    }));
  }

  /** Cube-matching colour for a per-layer-normalized brightness value (0-1).
   *  Returns [r, g, b] in 0-255, applying the same global/per-layer mapping
   *  and colormap the 3D points use. Requires the frame's grid entry to be
   *  cached (call ensureGrid first); falls back to per-layer if not. */
  heatColor(b, frameIdx) {
    const entry = this._grids.get(frameIdx);
    const t = this._cellT(b, entry);
    const [r, g, bl] = sampleColormap(this._colormap, t);
    return [Math.round(r * 255), Math.round(g * 255), Math.round(bl * 255)];
  }

  /** Frame metadata (node name, op type, shape) for a frame index. */
  getFrameMeta(frameIdx) {
    return this._frames[frameIdx] || null;
  }

  _autoFrameCamera() {
    if (this._frames.length === 0) return;
    const totalZ = (this._frames.length - 1) * this._sliceSpacing;
    const centerZ = totalZ / 2;
    this._controls.target.set(0, 0, centerZ);
    const dist = Math.max(SLAB_SIZE, totalZ) * 1.7;
    this._camera.position.set(0, -dist * 0.55, centerZ + dist * 0.85);
    this._controls.update();
  }

  // ── Getters ───────────────────────────────────────────────────────────

  get frameCount() { return this._frames.length; }
  get currentFrame() { return Math.floor(this._playhead); }
  get isPlaying() { return this._isPlaying; }
  get colormapNames() { return Object.keys(COLORMAPS); }
  get frames() { return this._frames; }
  get visibleIndices() { return this._visibleIndices; }
  get sliceSpacing() { return this._sliceSpacing; }
  get colorScale() { return this._colorScale; }
  get density() { return this._maxPointsPerLayer; }

  getGrid(frameIdx) {
    const e = this._grids.get(frameIdx);
    return e ? e.grid : undefined;
  }

  async ensureGrid(frameIdx) {
    if (this._grids.has(frameIdx)) return this.getGrid(frameIdx);
    const e = await this._loadFrameGrid(frameIdx);
    return e ? e.grid : undefined;
  }
}

// ── Singleton ────────────────────────────────────────────────────────────

const viewer = new Viewer3D();

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => viewer.init());
} else {
  viewer.init();
}

export { viewer, Viewer3D, COLORMAPS, sampleColormap };