/**
 * inference.js — Orchestrates model loading, input resolution, inference execution,
 * and output display. Wires DOM elements to backend REST + WebSocket endpoints.
 *
 * Imports from sibling modules:
 *   controls.js → { loadFrames, enableControls, disableControls }
 *   tree.js     → { loadNodeTree }
 */

import { loadFrames, enableControls, disableControls } from './controls.js';
import { loadNodeTree, refreshActivationStates } from './tree.js';
import { setNodes as setConnectionNodes } from './connections.js';

// ── DOM references ──────────────────────────────────────────────────────

const els = {
  // Status
  statusLabel: document.querySelector('.status-label'),
  statusPill:  document.getElementById('status-pill'),

  // Model section
  modelFileInput:  document.getElementById('model-file-input'),
  btnLoadModel:    document.getElementById('btn-load-model'),
  modelInfo:       document.getElementById('model-info'),
  infoModelName:   document.getElementById('info-model-name'),
  infoNumNodes:    document.getElementById('info-num-nodes'),
  infoNumInputs:   document.getElementById('info-num-inputs'),
  infoNumOutputs:  document.getElementById('info-num-outputs'),
  infoNumInter:    document.getElementById('info-num-intermediate'),

  // Input resolution
  inputSection:      document.getElementById('input-section'),
  inputShapesCont:   document.getElementById('input-shapes-container'),
  btnResolveInputs:  document.getElementById('btn-resolve-inputs'),

  // Inference controls
  inferenceSection: document.getElementById('inference-section'),
  optRandom:       document.getElementById('opt-random-input'),
  optPreserve:     document.getElementById('opt-preserve-nodes'),
  btnRunInference: document.getElementById('btn-run-inference'),

  // Output
  outputSection: document.getElementById('output-section'),
  outputList:   document.getElementById('output-list'),

  // Canvas states
  canvasEmpty:        document.getElementById('canvas-empty'),
  canvasLoading:      document.getElementById('canvas-loading'),
  canvasError:        document.getElementById('canvas-error'),
  canvasErrorText:    document.getElementById('canvas-error-text'),
  canvasErrorRetry:   document.getElementById('canvas-error-retry'),

  // Tree states
  treeEmpty:   document.getElementById('tree-empty'),
  treeLoading: document.getElementById('tree-loading'),
  treeError:   document.getElementById('tree-error'),
  treeErrorText: document.getElementById('tree-error-text'),
  treeContainer: document.getElementById('tree-container'),
  treePresets: document.getElementById('tree-presets'),
};

// ── State ───────────────────────────────────────────────────────────────

let _modelData = null;       // response from /api/load-model
let _resolvedShapes = null;  // response from /api/resolve-inputs
let _hasDynamicDims = false; // whether model has any dynamic dims

// ── Helpers ─────────────────────────────────────────────────────────────

function _setStatus(msg, state) {
  if (els.statusLabel) els.statusLabel.textContent = msg;

  // Infer state from message if not explicitly provided
  if (!state) {
    const lower = (msg || '').toLowerCase();
    if (lower.includes('error') || lower.includes('fail') || lower.includes('invalid')) {
      state = 'error';
    } else if (lower.includes('loading') || lower.includes('running') || lower.includes('resolving') || lower.includes('…') || lower.includes('...')) {
      state = 'loading';
    } else if (lower.includes('ready') || lower.includes('loaded') || lower.includes('resolved') || lower.includes('captured') || lower.includes('frames')) {
      state = 'active';
    } else {
      state = 'idle';
    }
  }

  if (els.statusPill) els.statusPill.setAttribute('data-state', state);
}

function _showCanvasError(msg) {
  if (els.canvasErrorText) els.canvasErrorText.textContent = msg;
  if (els.canvasError) els.canvasError.hidden = false;
  if (els.canvasLoading) els.canvasLoading.hidden = true;
  if (els.canvasEmpty) els.canvasEmpty.hidden = true;
}

function _hideCanvasStates() {
  if (els.canvasError) els.canvasError.hidden = true;
  if (els.canvasLoading) els.canvasLoading.hidden = true;
  if (els.canvasEmpty) els.canvasEmpty.hidden = true;
}

function _showCanvasLoading() {
  if (els.canvasLoading) els.canvasLoading.hidden = false;
  if (els.canvasError) els.canvasError.hidden = true;
  if (els.canvasEmpty) els.canvasEmpty.hidden = true;
}

/** Update the loading overlay caption (e.g. inference → streaming grids). */
function _setCanvasLoadingText(msg) {
  if (!els.canvasLoading) return;
  const p = els.canvasLoading.querySelector('p');
  if (p) p.textContent = msg;
}

function _showTreeError(msg) {
  if (els.treeErrorText) els.treeErrorText.textContent = msg;
  if (els.treeError) els.treeError.hidden = false;
  if (els.treeLoading) els.treeLoading.hidden = true;
  if (els.treeEmpty) els.treeEmpty.hidden = true;
}

function _escape(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

/** Map ONNX dtype string (as returned by the backend) to a readable name. */
const _DTYPE_NAMES = {
  FLOAT: 'float32', UINT8: 'uint8', INT8: 'int8', INT32: 'int32',
  INT64: 'int64', BOOL: 'bool', FLOAT16: 'float16', DOUBLE: 'float64',
  UINT32: 'uint32', UINT64: 'uint64', BFLOAT16: 'bfloat16',
};
function _dtypeName(t) {
  if (t == null) return '';
  const key = String(t).toUpperCase();
  return _DTYPE_NAMES[key] || `dtype_${t}`;
}

/** Check if a shape contains dynamic dimensions (null or symbolic string). */
function _hasDynamic(shape) {
  return shape.some(d => d === null || d === undefined || typeof d === 'string');
}

// ── Model Loading ──────────────────────────────────────────────────────

async function _loadModel() {
  const file = els.modelFileInput?.files?.[0];
  if (!file) {
    _setStatus('Select a model file');
    return;
  }

  _setStatus('Loading model…');

  // Disable load button
  if (els.btnLoadModel) els.btnLoadModel.disabled = true;

  // Show tree loading skeleton
  if (els.treeEmpty) els.treeEmpty.hidden = true;
  if (els.treeError) els.treeError.hidden = true;
  if (els.treeLoading) els.treeLoading.hidden = false;

  // Reset hidden sections
  if (els.inputSection) els.inputSection.hidden = true;
  if (els.inferenceSection) els.inferenceSection.hidden = true;
  if (els.outputSection) els.outputSection.hidden = true;
  if (els.modelInfo) els.modelInfo.hidden = true;

  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('brightness_mode', _getBrightnessMode());
    formData.append('preserve_nodes', String(els.optPreserve?.checked ?? false));

    const resp = await fetch('/api/load-model', {
      method: 'POST',
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }

    _modelData = await resp.json();

    // Update model info
    if (els.infoModelName) {
      // model_path is the uploaded filename (e.g. "rtdetrv2-det-best-640.onnx").
      const name = (_modelData.model_path || '').split(/[\\/]/).pop() || '—';
      els.infoModelName.textContent = name;
      els.infoModelName.title = name;
    }
    if (els.infoNumNodes) els.infoNumNodes.textContent = _modelData.num_nodes;
    if (els.infoNumInputs) els.infoNumInputs.textContent = _modelData.inputs.length;
    if (els.infoNumOutputs) els.infoNumOutputs.textContent = _modelData.outputs.length;
    if (els.infoNumInter) els.infoNumInter.textContent = _modelData.num_intermediate;
    if (els.modelInfo) els.modelInfo.hidden = false;

    // Build node tree
    loadNodeTree(_modelData.node_tree);

    // Hand graph nodes (inputs/outputs per node) to the connection visualizer
    // so it can derive dataflow edges once frames are loaded.
    setConnectionNodes(_modelData.nodes);

    // Show tree
    if (els.treeLoading) els.treeLoading.hidden = true;
    if (els.treeContainer) els.treeContainer.hidden = false;
    if (els.treePresets) els.treePresets.hidden = false;

    // Build input shape resolution UI
    _buildInputShapes(_modelData.inputs);

    // Show inference section
    if (els.inferenceSection) els.inferenceSection.hidden = false;
    if (els.btnRunInference) els.btnRunInference.disabled = false;

    _setStatus(`Model loaded: ${_modelData.num_nodes} nodes`);
  } catch (err) {
    console.error('Model load failed:', err);
    _showTreeError(err.message || 'Failed to load model');
    _setStatus('Model load failed');
  } finally {
    if (els.btnLoadModel) els.btnLoadModel.disabled = false;
  }
}

// ── Input Shape Resolution ─────────────────────────────────────────────

// Per-input file-picker elements, keyed by input name.
const _inputFileEls = {};        // { inputName: HTMLInputElement }
const _inputFileNameEls = {};    // { inputName: HTMLSpanElement }
const _inputSourceBadges = {};   // { inputName: HTMLSpanElement }

// Per-input data-source config (source of truth for _runInference).
//   { inputName: { mode, value, file, layout, channel_order } }
// mode: 'random' | 'file' | 'zeros' | 'ones' | 'constant' | 'inline'
// value: number (constant) | Array (inline) | null
// file:  File object (file mode) | null
// layout: 'auto' | 'NCHW' | 'NHWC'  (file mode, 4D image inputs only)
// channel_order: 'RGB' | 'BGR'      (file mode, 4D image inputs only)
const _inputConfig = {};
const _inputSpecs = {};          // { inputName: input spec from /api/load-model }

function _shapeDisplay(shape) {
  return `[${shape.map(d => (d === null || d === undefined) ? '?' : d).join(', ')}]`;
}

/** Concrete resolved shape for an input, reading its dynamic-dim fields. */
function _resolvedShapeFor(inp, wrap) {
  const fields = wrap ? wrap.querySelectorAll('.input-dim') : [];
  const byIdx = {};
  fields.forEach((f) => { byIdx[f.dataset.dimIndex] = parseInt(f.value, 10); });
  return inp.shape.map((d, i) => {
    // Concrete int dim — use as-is. Dynamic (null or symbolic string) → field.
    if (d !== null && d !== undefined && typeof d !== 'string') return d;
    const v = byIdx[String(i)];
    return (v && !isNaN(v) && v > 0) ? v : '?';
  });
}

/** A dim is "channel-like" if it's a concrete small number (1/2/3/4). */
function _isChannelDim(d) {
  return typeof d === 'number' && [1, 2, 3, 4].includes(d);
}

/** Infer NCHW vs NHWC for a 4D image input. Mirrors the backend's
 *  _detect_layout so UI defaults and the backend agree. */
function _detectLayout(shape) {
  if (shape.length !== 4) return 'NCHW';
  const d1 = shape[1], d2 = shape[2], d3 = shape[3];
  const chFirst = _isChannelDim(d1) && !_isChannelDim(d2);
  const chLast = _isChannelDim(d3) && !_isChannelDim(d2);
  if (chFirst && !chLast) return 'NCHW';
  if (chLast && !chFirst) return 'NHWC';
  if (chLast) return 'NHWC';
  return 'NCHW';
}

/** Default value for a dynamic dim, mirroring the backend's _dim_default so
 *  the UI and backend agree when the user doesn't override. Layout-aware for
 *  4D image inputs (NHWC puts H/W at indices 1/2, not 2/3). */
function _defaultDimValue(shape, i) {
  if (i === 0) return '1';                         // batch / N
  if (shape.length === 4) {
    const lay = _detectLayout(shape);
    if (lay === 'NHWC') {
      if (i === 1 || i === 2) return '224';        // H, W
      if (i === 3) return '3';                     // C (rarely dynamic)
      return '1';
    }
    if (i === 1) return '3';                       // C
    if (i === 2 || i === 3) return '224';          // H, W
    return '1';
  }
  if (shape.length === 2 && i === 1) return '128'; // [batch, seq]
  return '128';
}

/** H/W dim indices for a 4D layout (first = H, second = W). */
function _hwIndicesFor(layout) {
  return layout === 'NHWC' ? [1, 2] : [2, 3];
}

/** Refresh the "→ feed [shape]" hint on an input's label. */
function _refreshFeedShape(wrap, inp, label, dtypeStr) {
  const dtypePart = dtypeStr ? ` (${dtypeStr})` : '';
  label.textContent =
    `${inp.name} ${_shapeDisplay(inp.shape)}${dtypePart}  → feed ${_shapeDisplay(_resolvedShapeFor(inp, wrap))}`;
}

/** Build one row per model input: shape (+ editable dynamic dims) and a
 *  file picker so each input can be fed explicit data. */
function _buildInputShapes(inputs) {
  if (!els.inputShapesCont) return;
  els.inputShapesCont.innerHTML = '';
  for (const k of Object.keys(_inputFileEls)) delete _inputFileEls[k];
  for (const k of Object.keys(_inputFileNameEls)) delete _inputFileNameEls[k];
  for (const k of Object.keys(_inputSourceBadges)) delete _inputSourceBadges[k];

  _hasDynamicDims = false;
  for (const k of Object.keys(_inputConfig)) delete _inputConfig[k];
  for (const k of Object.keys(_inputSpecs)) delete _inputSpecs[k];

  for (const inp of inputs) {
    _inputSpecs[inp.name] = inp;
    _inputConfig[inp.name] = { mode: 'random', value: null, file: null };

    const wrap = document.createElement('div');
    wrap.className = 'input-shape-row input-feed-row';
    wrap.dataset.inputName = inp.name;

    const dtypeStr = _dtypeName(inp.dtype);
    const label = document.createElement('div');
    label.className = 'input-shape-label';
    wrap.appendChild(label);

    // Editable fields for dynamic dims.
    const dimsWrap = document.createElement('div');
    dimsWrap.className = 'input-shape-dims';
    for (let i = 0; i < inp.shape.length; i++) {
      const dim = inp.shape[i];
      // Dynamic dims are null OR symbolic strings (e.g. "N", "H", "W").
      if (dim === null || dim === undefined || typeof dim === 'string') {
        _hasDynamicDims = true;
        const field = document.createElement('input');
        field.type = 'number';
        field.className = 'input input-dim';
        field.dataset.inputName = inp.name;
        field.dataset.dimIndex = String(i);
        field.placeholder = typeof dim === 'string' ? dim : `dim ${i}`;
        field.value = _defaultDimValue(inp.shape, i);
        field.min = '1';
        field.title = 'Concrete value for this dynamic (unknown) dimension of the input shape.';
        dimsWrap.appendChild(field);
      }
    }
    if (dimsWrap.children.length > 0) {
      wrap.appendChild(dimsWrap);
      dimsWrap.addEventListener('input', () => _refreshFeedShape(wrap, inp, label, dtypeStr));
    }

    // Per-input file picker.
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.className = 'file-input';
    fileInput.accept = '.jpg,.jpeg,.png,.bmp,.npy,.npz';
    fileInput.dataset.inputName = inp.name;

    const fileLabel = document.createElement('label');
    fileLabel.className = 'file-upload-label input-file-label';
    fileLabel.textContent = 'Choose file';
    fileLabel.title = 'Choose an image (.jpg/.png/.bmp) or numpy (.npy/.npz) file to feed this input. Use Edit for layout and channel-order options.';
    fileLabel.appendChild(fileInput);

    const fileName = document.createElement('span');
    fileName.className = 'input-file-name';
    fileName.textContent = 'no file';
    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) {
        fileName.textContent = fileInput.files[0].name;
        fileName.classList.add('loaded');
        _inputConfig[inp.name] = { mode: 'file', value: null, file: fileInput.files[0] };
        _setRowBadge(inp.name, 'file');
      } else {
        fileName.textContent = 'no file';
        fileName.classList.remove('loaded');
        _inputConfig[inp.name] = { mode: 'random', value: null, file: null };
        _setRowBadge(inp.name, 'random');
      }
    });

    // Source badge, populated after inference from result.input_sources.
    const badge = document.createElement('span');
    badge.className = 'input-source-badge';
    badge.hidden = true;

    // Edit button — opens the per-input configuration modal (file / fill
    // generators / inline values / image resize).
    const editBtn = document.createElement('button');
    editBtn.className = 'btn btn-sm btn-secondary input-edit-btn';
    editBtn.type = 'button';
    editBtn.textContent = 'Edit';
    editBtn.title = 'Configure how this input is fed: file, random, zeros/ones/constant, or typed inline values; resize images and set channel layout/order.';
    editBtn.addEventListener('click', () => _openEditModal(inp.name));

    wrap.appendChild(fileLabel);
    wrap.appendChild(fileName);
    wrap.appendChild(editBtn);
    wrap.appendChild(badge);
    els.inputShapesCont.appendChild(wrap);

    _inputFileEls[inp.name] = fileInput;
    _inputFileNameEls[inp.name] = fileName;
    _inputSourceBadges[inp.name] = badge;

    _refreshFeedShape(wrap, inp, label, dtypeStr);
  }

  // Always show the Inputs section after a model loads; "Resolve Shapes"
  // only matters when there are dynamic dims.
  if (els.inputSection) els.inputSection.hidden = false;
  if (els.btnResolveInputs) els.btnResolveInputs.hidden = !_hasDynamicDims;
}

// ── Per-input Edit modal ───────────────────────────────────────────────

const _FILL_MODES = ['random', 'file', 'zeros', 'ones', 'constant', 'inline'];

/** Set a row's source badge to a mode label (pre-inference, from config). */
function _setRowBadge(name, mode) {
  const badge = _inputSourceBadges[name];
  if (!badge) return;
  badge.hidden = false;
  badge.textContent = mode;
  badge.classList.toggle('file', mode === 'file');
  badge.classList.toggle('random', mode === 'random');
  badge.classList.toggle('config', ['zeros', 'ones', 'constant', 'inline'].includes(mode));
}

/** Update the row's file-name display to mirror a File chosen in the modal. */
function _syncRowFileDisplay(name, file) {
  const fileName = _inputFileNameEls[name];
  if (!fileName) return;
  if (file) {
    fileName.textContent = file.name;
    fileName.classList.add('loaded');
  } else {
    fileName.textContent = 'no file';
    fileName.classList.remove('loaded');
  }
}

/** Build the resolved shape for an input (concrete + dynamic-dim fields). */
function _resolvedShapeForName(name) {
  const inp = _inputSpecs[name];
  if (!inp) return [];
  const wrap = els.inputShapesCont?.querySelector(
    `.input-shape-row[data-input-name="${CSS.escape(name)}"]`);
  return _resolvedShapeFor(inp, wrap);
}

let _activeModal = null;

/** Open the per-input configuration modal. */
function _openEditModal(inputName) {
  const inp = _inputSpecs[inputName];
  if (!inp) return;
  const cfg = _inputConfig[inputName] || { mode: 'random', value: null, file: null };

  // Close any existing modal first.
  if (_activeModal) _closeModal(_activeModal);

  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';

  const modal = document.createElement('div');
  modal.className = 'edit-modal';
  _activeModal = modal;

  // Header (drag handle)
  const header = document.createElement('div');
  header.className = 'edit-modal-header';
  const title = document.createElement('span');
  title.className = 'edit-modal-title';
  title.textContent = `Edit input: ${inputName}`;
  const closeBtn = document.createElement('button');
  closeBtn.className = 'btn btn-icon edit-modal-close';
  closeBtn.type = 'button';
  closeBtn.setAttribute('aria-label', 'Close');
  closeBtn.textContent = '✕';
  closeBtn.addEventListener('click', () => _closeModal(modal));
  header.appendChild(title);
  header.appendChild(closeBtn);
  modal.appendChild(header);

  // Body
  const body = document.createElement('div');
  body.className = 'edit-modal-body';

  // Shape + dtype summary
  const dtypeStr = _dtypeName(inp.dtype);
  const summary = document.createElement('div');
  summary.className = 'edit-modal-field edit-modal-summary';
  summary.textContent = `${_shapeDisplay(inp.shape)} (${dtypeStr}) → feed ${_shapeDisplay(_resolvedShapeForName(inputName))}`;
  body.appendChild(summary);

  // Source mode select
  const modeRow = document.createElement('div');
  modeRow.className = 'edit-modal-field';
  const modeLabel = document.createElement('label');
  modeLabel.className = 'edit-modal-label';
  modeLabel.textContent = 'Source';
  modeLabel.title = 'How this input is fed to the model: a file, random noise, or a constant/inline value.';
  const modeSelect = document.createElement('select');
  modeSelect.className = 'select';
  modeSelect.title = 'How this input is fed to the model: a file, random noise, or a constant/inline value.';
  for (const m of _FILL_MODES) {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    modeSelect.appendChild(opt);
  }
  modeSelect.value = cfg.mode;
  modeRow.appendChild(modeLabel);
  modeRow.appendChild(modeSelect);
  body.appendChild(modeRow);

  const isImage = inp.shape.length === 4;   // NCHW image-capable

  // File picker (file mode)
  const fileRow = document.createElement('div');
  fileRow.className = 'edit-modal-field edit-modal-file';
  const fileLabel = document.createElement('label');
  fileLabel.className = 'file-upload-label input-file-label';
  fileLabel.textContent = cfg.file ? `Choose file (${cfg.file.name})` : 'Choose file';
  fileLabel.title = 'Choose an image or numpy file to feed this input. Routed by input name — no filename matching required.';
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.className = 'file-input';
  fileInput.accept = '.jpg,.jpeg,.png,.bmp,.npy,.npz';
  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      fileLabel.textContent = `Choose file (${fileInput.files[0].name})`;
      pendingFile = fileInput.files[0];
    } else {
      pendingFile = null;
    }
  });
  fileLabel.appendChild(fileInput);
  fileRow.appendChild(fileLabel);
  body.appendChild(fileRow);

  // Layout + channel-order controls (file mode + 4D image). These let the
  // user feed models with non-NCHW image inputs (e.g. TensorFlow NHWC exports
  // with shapes like [unk, unk, unk, 3]) and remap RGB↔BGR channels.
  const detectedLayout = isImage ? _detectLayout(inp.shape) : 'NCHW';
  const imageOptsRow = document.createElement('div');
  imageOptsRow.className = 'edit-modal-field edit-modal-imageopts';
  if (isImage) {
    const layoutLabel = document.createElement('label');
    layoutLabel.className = 'edit-modal-label';
    layoutLabel.textContent = 'Layout';
    layoutLabel.title = 'Tensor layout of this image input. Auto infers from the shape; NCHW = [N,C,H,W], NHWC = [N,H,W,C] (common for TensorFlow exports).';
    const layoutSelect = document.createElement('select');
    layoutSelect.className = 'select';
    layoutSelect.title = layoutLabel.title;
    for (const v of ['auto', 'NCHW', 'NHWC']) {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v === 'auto' ? `auto (${detectedLayout})` : v;
      layoutSelect.appendChild(opt);
    }
    layoutSelect.value = cfg.layout || 'auto';

    const chLabel = document.createElement('label');
    chLabel.className = 'edit-modal-label';
    chLabel.textContent = 'Channel order';
    chLabel.title = 'Colour channel order the model expects. RGB matches PIL/JPEG; BGR swaps red and blue (OpenCV-trained models).';
    const chSelect = document.createElement('select');
    chSelect.className = 'select';
    chSelect.title = chLabel.title;
    for (const v of ['RGB', 'BGR']) {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      chSelect.appendChild(opt);
    }
    chSelect.value = cfg.channel_order || 'RGB';

    imageOptsRow.appendChild(layoutLabel);
    imageOptsRow.appendChild(layoutSelect);
    imageOptsRow.appendChild(chLabel);
    imageOptsRow.appendChild(chSelect);
    imageOptsRow._layoutSelect = layoutSelect;
    imageOptsRow._chSelect = chSelect;
  }
  body.appendChild(imageOptsRow);

  // Resize H / W (file mode + 4D image). The H/W axes depend on the chosen
  // layout, so the fields are rebuilt when the layout changes. Each field
  // syncs into the matching row .input-dim so the resolved shape follows.
  const resizeRow = document.createElement('div');
  resizeRow.className = 'edit-modal-field edit-modal-resize';
  const resizeHint = document.createElement('span');
  resizeHint.className = 'edit-modal-hint';
  resizeHint.textContent = isImage
    ? 'Image is resized to the target H × W before feeding the model.'
    : 'Resize applies to 4D image inputs only.';
  resizeRow.appendChild(resizeHint);

  const hwWrap = document.createElement('div');
  hwWrap.className = 'edit-modal-hw';
  if (isImage) resizeRow.appendChild(hwWrap);
  body.appendChild(resizeRow);

  const wrap = els.inputShapesCont?.querySelector(
    `.input-shape-row[data-input-name="${CSS.escape(inputName)}"]`);

  function buildHwFields() {
    if (!isImage) return;
    hwWrap.innerHTML = '';
    const chosen = imageOptsRow._layoutSelect.value;
    const eff = chosen === 'auto' ? detectedLayout : chosen;
    const indices = _hwIndicesFor(eff);
    indices.forEach((idx, k) => {
      const dim = inp.shape[idx];
      const lbl = document.createElement('span');
      lbl.className = 'edit-modal-label';
      lbl.textContent = k === 0 ? 'H' : 'W';
      lbl.title = k === 0
        ? 'Target height the image is resized to before feeding. Editable only for dynamic dimensions.'
        : 'Target width the image is resized to before feeding. Editable only for dynamic dimensions.';
      const field = document.createElement('input');
      field.type = 'number';
      field.className = 'input input-dim';
      field.min = '1';
      field.dataset.inputName = inputName;
      field.dataset.dimIndex = String(idx);
      field.title = lbl.title;
      const dynamic = dim === null || dim === undefined || typeof dim === 'string';
      if (dynamic) {
        field.placeholder = typeof dim === 'string' ? dim : `dim ${idx}`;
        const rowField = wrap ? wrap.querySelector(
          `.input-dim[data-dim-index="${idx}"]`) : null;
        field.value = rowField ? rowField.value : _defaultDimValue(inp.shape, idx);
        field.addEventListener('input', () => {
          if (rowField) {
            rowField.value = field.value;
            rowField.dispatchEvent(new Event('input', { bubbles: true }));
          }
        });
      } else {
        field.value = String(dim);
        field.disabled = true;
        field.title = 'Model-fixed dimension — cannot resize.';
      }
      hwWrap.appendChild(lbl);
      hwWrap.appendChild(field);
    });
  }
  if (isImage) {
    imageOptsRow._layoutSelect.addEventListener('change', buildHwFields);
    buildHwFields();
  }

  // Constant value (constant mode)
  const constRow = document.createElement('div');
  constRow.className = 'edit-modal-field edit-modal-constant';
  const constLabel = document.createElement('label');
  constLabel.className = 'edit-modal-label';
  constLabel.textContent = 'Value';
  constLabel.title = 'Fill value used for every element when Source = constant. Cast to the input dtype.';
  const constInput = document.createElement('input');
  constInput.type = 'number';
  constInput.className = 'input';
  constInput.step = 'any';
  constInput.title = constLabel.title;
  constInput.value = (cfg.mode === 'constant' && cfg.value != null) ? cfg.value : '0';
  constRow.appendChild(constLabel);
  constRow.appendChild(constInput);
  body.appendChild(constRow);

  // Inline values (inline mode)
  const inlineRow = document.createElement('div');
  inlineRow.className = 'edit-modal-field edit-modal-inline';
  const inlineLabel = document.createElement('label');
  inlineLabel.className = 'edit-modal-label';
  inlineLabel.textContent = 'Values';
  inlineLabel.title = 'Raw values for this input as a flat or nested list, reshaped to the input’s resolved shape. The total element count must match.';
  const inlineTa = document.createElement('textarea');
  inlineTa.className = 'input edit-modal-textarea';
  inlineTa.rows = 3;
  inlineTa.title = inlineLabel.title;
  const resolved = _resolvedShapeForName(inputName);
  inlineTa.placeholder = `flat or nested list, e.g. ${_shapeDisplay(resolved)} → "640, 480"`;
  if (cfg.mode === 'inline' && cfg.value != null) {
    inlineTa.value = Array.isArray(cfg.value) ? cfg.value.join(', ') : String(cfg.value);
  }
  inlineRow.appendChild(inlineLabel);
  inlineRow.appendChild(inlineTa);
  body.appendChild(inlineRow);

  // Show/hide fields based on mode
  function _syncModeVisibility() {
    const m = modeSelect.value;
    fileRow.hidden = m !== 'file';
    imageOptsRow.hidden = !(m === 'file' && isImage);
    resizeRow.hidden = !(m === 'file' && isImage);
    constRow.hidden = m !== 'constant';
    inlineRow.hidden = m !== 'inline';
  }
  modeSelect.addEventListener('change', _syncModeVisibility);
  _syncModeVisibility();

  let pendingFile = cfg.file || null;

  // Actions
  const actions = document.createElement('div');
  actions.className = 'edit-modal-actions';
  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'btn btn-secondary btn-sm';
  cancelBtn.type = 'button';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', () => _closeModal(modal));
  const applyBtn = document.createElement('button');
  applyBtn.className = 'btn btn-primary btn-sm';
  applyBtn.type = 'button';
  applyBtn.textContent = 'Apply';
  applyBtn.addEventListener('click', () => {
    const mode = modeSelect.value;
    const next = { mode, value: null, file: null, layout: 'auto', channel_order: 'RGB' };
    if (mode === 'file') {
      next.file = pendingFile || null;
      if (isImage) {
        next.layout = imageOptsRow._layoutSelect.value;
        next.channel_order = imageOptsRow._chSelect.value;
      }
    } else if (mode === 'constant') {
      const v = parseFloat(constInput.value);
      next.value = isNaN(v) ? 0 : v;
    } else if (mode === 'inline') {
      next.value = _parseInline(inlineTa.value);
    }
    _inputConfig[inputName] = next;
    _syncRowFileDisplay(inputName, next.file);
    _setRowBadge(inputName, mode);
    _closeModal(modal);
  });
  actions.appendChild(cancelBtn);
  actions.appendChild(applyBtn);
  body.appendChild(actions);

  modal.appendChild(body);
  backdrop.appendChild(modal);
  document.body.appendChild(backdrop);

  backdrop.addEventListener('click', (e) => { if (e.target === backdrop) _closeModal(modal); });

  // Center the modal, then enable header-drag.
  _centerModal(modal);
  _enableModalDrag(modal, header);
}

/** Parse an inline-values textarea into a flat/nested number list. */
function _parseInline(text) {
  const t = (text || '').trim();
  if (!t) return null;
  // Accept JSON arrays or comma/whitespace-separated scalars.
  if (t.startsWith('[')) {
    try { return JSON.parse(t); } catch { /* fall through */ }
  }
  return t.split(/[,\s]+/).filter(s => s.length > 0).map(Number);
}

function _closeModal(modal) {
  if (modal) {
    // Remove the whole backdrop (the modal's .modal-backdrop parent), not
    // just the inner panel — otherwise the dim layer stays and blocks input.
    const backdrop = modal.closest('.modal-backdrop') || modal.parentNode;
    if (backdrop && backdrop.parentNode) backdrop.parentNode.removeChild(backdrop);
  }
  if (_activeModal === modal) _activeModal = null;
}

function _centerModal(modal) {
  const r = modal.getBoundingClientRect();
  modal.style.left = `${Math.max(8, (window.innerWidth - r.width) / 2)}px`;
  modal.style.top = `${Math.max(8, (window.innerHeight - r.height) / 2)}px`;
}

function _enableModalDrag(modal, handle) {
  handle.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.edit-modal-close')) return;
    handle.setPointerCapture(e.pointerId);
    const rect = modal.getBoundingClientRect();
    const startX = e.clientX, startY = e.clientY;
    const startLeft = rect.left, startTop = rect.top;
    const onMove = (ev) => {
      const left = Math.max(0, Math.min(window.innerWidth - rect.width, startLeft + (ev.clientX - startX)));
      const top = Math.max(0, Math.min(window.innerHeight - rect.height, startTop + (ev.clientY - startY)));
      modal.style.left = `${left}px`;
      modal.style.top = `${top}px`;
    };
    const onUp = (ev) => {
      handle.removeEventListener('pointermove', onMove);
      handle.removeEventListener('pointerup', onUp);
      handle.removeEventListener('pointercancel', onUp);
    };
    handle.addEventListener('pointermove', onMove);
    handle.addEventListener('pointerup', onUp);
    handle.addEventListener('pointercancel', onUp);
  });
}

/** Collect dynamic dim values from UI fields. */
function _collectDynamicDims() {
  if (!_hasDynamicDims) return null;
  const fields = els.inputShapesCont?.querySelectorAll('.input-dim');
  if (!fields || fields.length === 0) return null;

  const dims = {};
  for (const f of fields) {
    const name = f.dataset.inputName;
    const idx = parseInt(f.dataset.dimIndex, 10);
    const val = parseInt(f.value, 10);
    if (isNaN(val) || val < 1) {
      continue;
    }
    if (!dims[name]) dims[name] = {};
    dims[name][String(idx)] = val;
  }
  return Object.keys(dims).length > 0 ? dims : null;
}

/** Resolve dynamic input dimensions via backend. */
async function _resolveInputs() {
  const dynamicDims = _collectDynamicDims();
  if (!dynamicDims) {
    _setStatus('No dynamic dims to resolve');
    return;
  }

  _setStatus('Resolving input shapes…');
  if (els.btnResolveInputs) els.btnResolveInputs.disabled = true;

  try {
    const resp = await fetch('/api/resolve-inputs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dynamic_dims: dynamicDims }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }

    _resolvedShapes = await resp.json();
    _setStatus('Input shapes resolved');
  } catch (err) {
    console.error('Resolve inputs failed:', err);
    _setStatus(`Resolve failed: ${err.message}`);
  } finally {
    if (els.btnResolveInputs) els.btnResolveInputs.disabled = false;
  }
}

// ── Inference ──────────────────────────────────────────────────────────

function _getBrightnessMode() {
  const checked = document.querySelector('input[name="brightness-mode"]:checked');
  return checked?.value || 'mean';
}

/** Update per-input source badges from the inference response and return a
 *  concise "name←mode, …" summary. */
function _applyInputSources(sources) {
  const parts = [];
  for (const name of Object.keys(_inputFileEls)) {
    const src = (sources && sources[name]) ? sources[name] : 'random';
    const badge = _inputSourceBadges[name];
    if (badge) {
      badge.hidden = false;
      badge.textContent = src;
      badge.classList.toggle('file', src === 'file');
      badge.classList.toggle('random', src === 'random');
      badge.classList.toggle('config', ['zeros', 'ones', 'constant', 'inline'].includes(src));
    }
    parts.push(`${name}←${src}`);
  }
  return parts.join(', ');
}

async function _runInference() {
  if (!_modelData) return;

  _showCanvasLoading();
  _setCanvasLoadingText('Running inference…');
  _setStatus('Running inference…');
  if (els.btnRunInference) els.btnRunInference.disabled = true;
  disableControls();

  const preserveNodes = els.optPreserve?.checked ?? false;
  const brightnessMode = _getBrightnessMode();
  const dynamicDims = _collectDynamicDims();

  // Gather per-input config. Files come from _inputConfig (set by the row
  // picker or the Edit modal); fill/inline modes go into input_config_json.
  const fileEntries = [];          // [[name, File]]
  const inputConfig = {};          // { name: { mode, value? } }
  for (const name of Object.keys(_inputConfig)) {
    const cfg = _inputConfig[name];
    if (!cfg) continue;
    if (cfg.mode === 'file' && cfg.file) {
      fileEntries.push([name, cfg.file]);
      // Send layout / channel_order so the backend loads the image in the
      // model's native layout (NCHW vs NHWC) and with the chosen channel order.
      inputConfig[name] = {
        mode: 'file',
        layout: cfg.layout || 'auto',
        channel_order: cfg.channel_order || 'RGB',
      };
    } else if (['zeros', 'ones', 'constant', 'inline'].includes(cfg.mode)) {
      const entry = { mode: cfg.mode };
      if (cfg.mode === 'constant' || cfg.mode === 'inline') entry.value = cfg.value;
      inputConfig[name] = entry;
    }
  }

  const hasConfig = fileEntries.length > 0 || Object.keys(inputConfig).length > 0;

  try {
    let resp;

    if (hasConfig) {
      // Multipart upload: per-input files + fill/inline config. Files and
      // explicit config override random, so force use_random=false and route
      // each file explicitly by input name (no filename matching required).
      const formData = new FormData();
      formData.append('use_random', 'false');
      formData.append('brightness_mode', brightnessMode);
      formData.append('preserve_nodes', String(preserveNodes));
      if (dynamicDims) {
        formData.append('dynamic_dims_json', JSON.stringify(dynamicDims));
      }
      if (Object.keys(inputConfig).length > 0) {
        formData.append('input_config_json', JSON.stringify(inputConfig));
      }
      const mapping = {};
      for (const [name, file] of fileEntries) {
        mapping[name] = file.name;
        formData.append('files', file, file.name);
      }
      if (Object.keys(mapping).length > 0) {
        formData.append('file_mapping_json', JSON.stringify(mapping));
      }
      resp = await fetch('/api/inference', {
        method: 'POST',
        body: formData,
      });
    } else {
      // No files/config — random quick path (honors the Random Gaussian checkbox).
      const useRandom = els.optRandom?.checked ?? true;
      resp = await fetch('/api/inference-json', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          use_random: useRandom,
          brightness_mode: brightnessMode,
          preserve_nodes: preserveNodes,
          dynamic_dims: dynamicDims,
        }),
      });
    }

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }

    const result = await resp.json();

    if (!result.frame_metadata || result.frame_metadata.length === 0) {
      _showCanvasError('No frames captured. Model may have no intermediate layers.');
      _setStatus('No frames captured');
      return;
    }

    // Load frames into viewer + controls. Keep the loading overlay up while
    // the per-layer grids stream in — placeholder shells stay hidden until
    // the cube is fully built, then the overlay drops.
    _showCanvasLoading();
    _setCanvasLoadingText('Loading activations…');
    await loadFrames(result.frame_metadata);
    // Now that the cube is populated, grey out + disable tree leaves whose
    // nodes have no activation frame (metadata/shape tensors).
    refreshActivationStates();
    _hideCanvasStates();

    // Confirm per-input data sources so the user can verify their files
    // were actually used (badges on each input row + a status summary).
    const srcSummary = _applyInputSources(result.input_sources);

    // Show skipped warning if any
    if (result.skipped_count > 0) {
      _setStatus(`${result.total_frames} frames, ${result.skipped_count} skipped by ORT`);
    } else {
      _setStatus(`${result.total_frames} frames captured${srcSummary ? ` · ${srcSummary}` : ''}`);
    }

    // Load outputs
    if (result.outputs_count > 0) {
      await _loadOutputs(result.outputs_count);
    }
  } catch (err) {
    console.error('Inference failed:', err);
    _showCanvasError(err.message || 'Inference failed');
    _setStatus('Inference failed');
  } finally {
    if (els.btnRunInference) els.btnRunInference.disabled = false;
  }
}

// ── Output Display ──────────────────────────────────────────────────────

async function _loadOutputs(count) {
  if (!els.outputList) return;
  els.outputList.innerHTML = '';

  for (let i = 0; i < count; i++) {
    try {
      const resp = await fetch(`/api/output/${i}`);
      if (!resp.ok) continue;
      const data = await resp.json();

      const card = document.createElement('div');
      card.className = 'output-card';

      // Header: name + shape
      const header = document.createElement('div');
      header.className = 'output-header';
      header.textContent = `${data.name} [${(data.shape || []).join(', ')}]`;
      card.appendChild(header);

      if (data.truncated) {
        const stats = data.stats;
        if (stats) {
          const statsEl = document.createElement('div');
          statsEl.className = 'output-stats';
          statsEl.textContent = `min=${stats.min?.toFixed(4)} max=${stats.max?.toFixed(4)} mean=${stats.mean?.toFixed(4)} std=${stats.std?.toFixed(4)} (too large to display)`;
          card.appendChild(statsEl);
        }
      } else if (data.values) {
        const heatmap = _renderOutputHeatmap(data.values, data.shape);
        card.appendChild(heatmap);
      }

      els.outputList.appendChild(card);
    } catch (e) {
      console.warn(`Failed to load output ${i}:`, e);
    }
  }

  if (els.outputList.children.length > 0) {
    if (els.outputSection) els.outputSection.hidden = false;
  }
}

/** Render raw output values as a simple bar/heatmap canvas. */
function _renderOutputHeatmap(values, shape) {
  const canvas = document.createElement('canvas');
  canvas.className = 'output-canvas';
  canvas.width = 256;
  canvas.height = 64;

  const ctx = canvas.getContext('2d');
  const flat = Array.isArray(values[0]) ? values.flat(Infinity) : values;
  const arr = Float64Array.from(flat);

  // Normalize
  let min = Infinity, max = -Infinity;
  for (const v of arr) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;

  // Determine grid layout
  const total = arr.length;
  let h = 1, w = total;
  if (total > 1) {
    h = Math.ceil(Math.sqrt(total));
    w = Math.ceil(total / h);
  }

  const cellW = canvas.width / w;
  const cellH = canvas.height / h;

  for (let i = 0; i < total; i++) {
    const norm = (arr[i] - min) / range;
    const row = Math.floor(i / w);
    const col = i % w;
    // Simple grayscale + viridis-ish color
    const r = Math.round(255 * (0.5 + 0.5 * Math.sin(norm * 2.5)));
    const g = Math.round(255 * norm);
    const b = Math.round(255 * (1 - norm));
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    ctx.fillRect(col * cellW, row * cellH, Math.ceil(cellW), Math.ceil(cellH));
  }

  return canvas;
}

// ── Event Wiring ───────────────────────────────────────────────────────

function _init() {
  // Load model button — opens the file picker dialog
  if (els.btnLoadModel) {
    els.btnLoadModel.addEventListener('click', () => {
      els.modelFileInput?.click();
    });
  }
  // When a file is selected via the picker, auto-load the model
  if (els.modelFileInput) {
    els.modelFileInput.addEventListener('change', () => {
      if (els.modelFileInput.files && els.modelFileInput.files.length > 0) {
        _setStatus(`Selected: ${els.modelFileInput.files[0].name}`);
        _loadModel();
      }
    });
  }

  // Resolve inputs button
  if (els.btnResolveInputs) {
    els.btnResolveInputs.addEventListener('click', _resolveInputs);
  }

  // Run inference button
  if (els.btnRunInference) {
    els.btnRunInference.addEventListener('click', _runInference);
  }

  // "Random Gaussian input" checkbox — when checked, reset every input to
  // random mode (clears any per-input file/fill/inline config).
  if (els.optRandom) {
    els.optRandom.addEventListener('change', () => {
      if (!els.optRandom.checked) return;
      for (const name of Object.keys(_inputConfig)) {
        _inputConfig[name] = { mode: 'random', value: null, file: null };
        const fi = _inputFileEls[name];
        if (fi) fi.value = '';
        _syncRowFileDisplay(name, null);
        _setRowBadge(name, 'random');
      }
    });
  }

  // Canvas error retry
  if (els.canvasErrorRetry) {
    els.canvasErrorRetry.addEventListener('click', _runInference);
  }

  // Auto-resolve when dynamic dims change
  if (els.inputShapesCont) {
    els.inputShapesCont.addEventListener('change', (e) => {
      if (e.target.classList.contains('input-dim')) {
        _resolveInputs();
      }
    });
  }
}

// ── Auto-init when DOM ready ───────────────────────────────────────────

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _init);
} else {
  _init();
}