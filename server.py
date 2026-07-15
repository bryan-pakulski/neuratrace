"""FastAPI server for ONNX Activation Viewer.

Provides REST + WebSocket endpoints for:
  - Loading ONNX models and introspecting graph structure
  - Resolving dynamic input dimensions
  - Running ORT inference with intermediate activation capture
  - Retrieving brightness grids (frames) and raw output tensors
  - Streaming frame metadata over WebSocket during inference

Run: uvicorn onnx_viewer.server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from onnx_loader import OnnxGraphIntrospector
from capture import ActivationCapture
from input_handler import InputHandler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------
# Session state (module-level — single-user tool)
# -----------------------------------------------------------------------

_introspector: OnnxGraphIntrospector | None = None
_input_handler: InputHandler | None = None
_capture_result: dict[str, Any] | None = None  # Last inference result
_model_path: str | None = None
_brightness_mode: str = "mean"
_preserve_nodes: bool = False
# Temp file paths for uploaded input files
_temp_files: list[str] = []

# -----------------------------------------------------------------------
# Pydantic models for request bodies
# -----------------------------------------------------------------------


class LoadModelRequest(BaseModel):
    model_path: str
    brightness_mode: str = "mean"
    preserve_nodes: bool = False
    # Note: /api/load-model now accepts multipart file upload instead of JSON body.
    # This Pydantic model is kept for backward compatibility with the JSON endpoint variant.


class ResolveInputsRequest(BaseModel):
    dynamic_dims: dict[str, dict[str, int]] | None = None


class InferenceJsonRequest(BaseModel):
    use_random: bool = False
    brightness_mode: str = "mean"
    preserve_nodes: bool = False
    dynamic_dims: dict[str, dict[str, int]] | None = None
    input_config: dict[str, dict[str, Any]] | None = None


# -----------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------

app = FastAPI(title="ONNX Activation Viewer", version="1.0.0")

# Static files (create directory if missing — frontend served from here)
_static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(_static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _serialize_tensor_spec(spec) -> dict[str, Any]:
    """Convert TensorSpec dataclass to JSON-serializable dict."""
    return {
        "name": spec.name,
        "shape": [d if d is not None else None for d in spec.shape],
        "dtype": spec.dtype,
    }


def _serialize_node_info(node) -> dict[str, Any]:
    """Convert NodeInfo dataclass to JSON-serializable dict."""
    return {
        "name": node.name,
        "op_type": node.op_type,
        "inputs": node.inputs,
        "outputs": node.outputs,
        "exec_order": node.exec_order,
    }


def _frame_metadata(frame: dict[str, Any]) -> dict[str, Any]:
    """Extract lightweight frame metadata (no grid data) for list responses."""
    return {
        "frame_idx": frame["frame_idx"],
        "tensor_name": frame["tensor_name"],
        "node_name": frame["node_name"],
        "op_type": frame["op_type"],
        "exec_order": frame["exec_order"],
        "shape": frame["shape"],
        "original_shape": frame["original_shape"],
        "is_uniform": frame["is_uniform"],
        "sampled": frame["sampled"],
        "raw_min": frame.get("raw_min"),
        "raw_max": frame.get("raw_max"),
        "dtype": frame.get("dtype", "float32"),
    }


def _cleanup_temp_files():
    """Remove temporary uploaded files."""
    for f in _temp_files:
        try:
            os.remove(f)
        except OSError:
            pass
    _temp_files.clear()


# -----------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------


@app.post("/api/load-model")
async def load_model(
    file: UploadFile = File(...),
    brightness_mode: str = Form("mean"),
    preserve_nodes: str = Form("false"),
):
    """Load an ONNX model from file upload and return graph metadata.

    Returns: {inputs, outputs, nodes, node_tree, intermediate_names}
    """
    global _introspector, _model_path, _brightness_mode, _preserve_nodes, _capture_result

    if not file.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "No file provided."},
        )

    # Save uploaded file to temp path
    suffix = os.path.splitext(file.filename)[1] or ".onnx"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="onnx_viewer_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())
    except Exception:
        os.unlink(tmp_path)
        raise

    try:
        _introspector = OnnxGraphIntrospector(tmp_path)
        _introspector.parse()
        _model_path = tmp_path
        _brightness_mode = brightness_mode
        _preserve_nodes = preserve_nodes.lower() in ("true", "1", "yes")
        _capture_result = None  # Reset previous inference

        inputs = [_serialize_tensor_spec(s) for s in _introspector.inputs]
        outputs = [_serialize_tensor_spec(s) for s in _introspector.outputs]
        nodes = [_serialize_node_info(n) for n in _introspector.nodes]
        node_tree = _introspector.build_node_tree()
        intermediate_names = _introspector.get_intermediate_tensor_names()

        logger.info(
            "Model loaded: %s (%d nodes, %d inputs, %d outputs)",
            file.filename,
            len(nodes),
            len(inputs),
            len(outputs),
        )

        return {
            "model_path": file.filename,
            "inputs": inputs,
            "outputs": outputs,
            "nodes": nodes,
            "node_tree": node_tree,
            "intermediate_names": intermediate_names,
            "num_nodes": len(nodes),
            "num_intermediate": len(intermediate_names),
        }

    except Exception as e:
        logger.exception("Failed to load model: %s", file.filename)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__},
        )


@app.post("/api/resolve-inputs")
async def resolve_inputs(req: ResolveInputsRequest):
    """Resolve dynamic input dimensions to concrete values.

    Returns: {resolved_shapes, dynamic_dims_info}
    """
    global _input_handler

    if _introspector is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No model loaded. Call /api/load-model first."},
        )

    try:
        _input_handler = InputHandler(_introspector.inputs)
        resolved = _input_handler.resolve_shapes(req.dynamic_dims)
        dynamic_info = _input_handler.get_dynamic_dims()

        return {
            "resolved_shapes": {
                name: list(shape) for name, shape in resolved.items()
            },
            "dynamic_dims_info": dynamic_info,
        }

    except Exception as e:
        logger.exception("Failed to resolve inputs")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__},
        )


@app.post("/api/inference")
async def run_inference(
    use_random: bool = Form(False),
    brightness_mode: str = Form("mean"),
    preserve_nodes: bool = Form(False),
    dynamic_dims_json: str = Form(None),
    file_mapping_json: str = Form(None),
    input_config_json: str = Form(None),
    files: list[UploadFile] = File(default=[]),
):
    """Run ORT inference with intermediate activation capture.

    Accepts multipart form data:
      - use_random: If true, generate random Gaussian input
      - brightness_mode: 'mean' or 'max'
      - preserve_nodes: If true, disable ORT graph optimization
      - dynamic_dims_json: JSON string of {input_name: {dim_name: value}}
      - files: Uploaded input files (image or numpy). Filename must match
               an input tensor name, or be provided via 'file_mapping_json'.

    Returns: {total_frames, frame_metadata, skipped_count, outputs_count, skipped}
    """
    global _capture_result, _brightness_mode, _preserve_nodes

    if _introspector is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No model loaded. Call /api/load-model first."},
        )

    try:
        # Parse dynamic dims if provided
        dynamic_dims = None
        if dynamic_dims_json:
            try:
                dynamic_dims = json.loads(dynamic_dims_json)
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid dynamic_dims_json: not valid JSON"},
                )

        # Parse explicit file→input mapping if provided.
        file_mapping: dict[str, str] = {}
        if file_mapping_json:
            try:
                file_mapping = json.loads(file_mapping_json) or {}
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid file_mapping_json: not valid JSON"},
                )

        # Parse per-input data-source config (zeros/ones/constant/inline).
        input_config: dict[str, dict] = {}
        if input_config_json:
            try:
                input_config = json.loads(input_config_json) or {}
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid input_config_json: not valid JSON"},
                )

        # Build input handler
        input_handler = InputHandler(_introspector.inputs)
        input_handler.resolve_shapes(dynamic_dims)

        # Save uploads to temp files and build {input_name: file_path}.
        # Route explicitly via file_mapping_json when present; otherwise fall
        # back to matching filename (no extension) == input name.
        file_paths: dict[str, str] = {}
        if files:
            _cleanup_temp_files()
            saved_by_name: dict[str, str] = {}   # original filename → tmp path
            for f in files:
                suffix = os.path.splitext(f.filename or "")[1]
                fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                os.close(fd)
                with open(tmp_path, "wb") as out:
                    content = await f.read()
                    out.write(content)
                _temp_files.append(tmp_path)
                saved_by_name[f.filename or ""] = tmp_path

            if file_mapping:
                for input_name, fname in file_mapping.items():
                    if fname in saved_by_name:
                        file_paths[input_name] = saved_by_name[fname]
            else:
                for spec in _introspector.inputs:
                    for basename, tmp_path in saved_by_name.items():
                        name_no_ext = os.path.splitext(basename)[0]
                        if basename == spec.name or name_no_ext == spec.name:
                            file_paths[spec.name] = tmp_path
                            break
                # Single-input model: assign the lone upload to it.
                if not file_paths and len(_introspector.inputs) == 1 and saved_by_name:
                    file_paths[_introspector.inputs[0].name] = next(
                        iter(saved_by_name.values())
                    )

            # Files were uploaded but none matched any input — refuse rather
            # than silently run all-random.
            if not file_paths:
                expected = ", ".join(spec.name for spec in _introspector.inputs)
                got = ", ".join(saved_by_name.keys())
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            "Uploaded file(s) did not match any model input. "
                            f"Got: [{got}]. Expected one of: [{expected}]. "
                            "Name files <input_name>.<ext> or pass file_mapping_json."
                        ),
                    },
                )

        # Generate or load input data
        input_feed = input_handler.load_inputs(
            file_paths=file_paths if file_paths else None,
            dynamic_dims=dynamic_dims,
            use_random=use_random,
            input_config=input_config if input_config else None,
        )

        # Run activation capture
        _brightness_mode = brightness_mode
        _preserve_nodes = preserve_nodes

        intermediate_names = _introspector.get_intermediate_tensor_names()
        capture = ActivationCapture(
            model_path=_model_path,
            intermediate_names=intermediate_names,
            brightness_mode=brightness_mode,
            preserve_nodes=preserve_nodes,
        )

        result = capture.run_inference(input_feed)
        _capture_result = result

        # Build lightweight frame metadata for response
        frame_meta = [_frame_metadata(f) for f in result["frames"]]

        logger.info(
            "Inference complete: %d frames, %d skipped, %d outputs",
            len(frame_meta),
            len(result["skipped"]),
            len(result["outputs"]),
        )

        return {
            "total_frames": len(frame_meta),
            "frame_metadata": frame_meta,
            "skipped_count": len(result["skipped"]),
            "skipped": result["skipped"],
            "outputs_count": len(result["outputs"]),
            "total_requested": result["total_requested"],
            "total_captured": result["total_captured"],
            # Per-input data source so the UI can confirm what fed the model.
            "input_sources": getattr(input_handler, "input_sources", {}),
        }

    except ValueError as e:
        # Shape mismatches / unsupported file types from InputHandler.
        logger.warning("Inference input error: %s", e)
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "type": "ValueError"},
        )
    except Exception as e:
        logger.exception("Inference failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__},
        )
    finally:
        _cleanup_temp_files()


@app.post("/api/inference-json")
async def run_inference_json(req: InferenceJsonRequest):
    """Run inference with random input (JSON body, no file upload).

    Convenience endpoint for quick inference without multipart.
    """
    global _capture_result, _brightness_mode, _preserve_nodes

    if _introspector is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No model loaded. Call /api/load-model first."},
        )

    try:
        input_handler = InputHandler(_introspector.inputs)
        input_feed = input_handler.load_inputs(
            use_random=req.use_random,
            dynamic_dims=req.dynamic_dims,
            input_config=req.input_config,
        )

        _brightness_mode = req.brightness_mode
        _preserve_nodes = req.preserve_nodes

        intermediate_names = _introspector.get_intermediate_tensor_names()
        capture = ActivationCapture(
            model_path=_model_path,
            intermediate_names=intermediate_names,
            brightness_mode=req.brightness_mode,
            preserve_nodes=req.preserve_nodes,
        )

        result = capture.run_inference(input_feed)
        _capture_result = result

        frame_meta = [_frame_metadata(f) for f in result["frames"]]

        return {
            "total_frames": len(frame_meta),
            "frame_metadata": frame_meta,
            "skipped_count": len(result["skipped"]),
            "skipped": result["skipped"],
            "outputs_count": len(result["outputs"]),
            "total_requested": result["total_requested"],
            "total_captured": result["total_captured"],
            "input_sources": getattr(input_handler, "input_sources", {}),
        }

    except ValueError as e:
        # Shape mismatches / inline-value errors from InputHandler.
        logger.warning("Inference (JSON) input error: %s", e)
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "type": "ValueError"},
        )
    except Exception as e:
        logger.exception("Inference (JSON) failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__},
        )


@app.get("/api/frame/{frame_idx}")
async def get_frame(frame_idx: int):
    """Retrieve full brightness grid for a specific frame.

    Returns: {frame_idx, grid, shape, node_name, op_type, exec_order,
              is_uniform, sampled, original_shape, dtype}
    """
    if _capture_result is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No inference results. Call /api/inference first."},
        )

    frames = _capture_result["frames"]
    if frame_idx < 0 or frame_idx >= len(frames):
        return JSONResponse(
            status_code=404,
            content={"error": f"Frame index {frame_idx} out of range (0-{len(frames)-1})"},
        )

    frame = frames[frame_idx]
    return {
        "frame_idx": frame["frame_idx"],
        "grid": frame["grid"],
        "shape": frame["shape"],
        "node_name": frame["node_name"],
        "op_type": frame["op_type"],
        "exec_order": frame["exec_order"],
        "is_uniform": frame["is_uniform"],
        "sampled": frame["sampled"],
        "original_shape": frame["original_shape"],
        "raw_min": frame.get("raw_min"),
        "raw_max": frame.get("raw_max"),
        "dtype": frame.get("dtype", "float32"),
    }


def _frame_or_404(frame_idx: int):
    """Return (frame, None) or (None, JSONResponse) for a frame index."""
    if _capture_result is None:
        return None, JSONResponse(
            status_code=400,
            content={"error": "No inference results. Call /api/inference first."},
        )
    frames = _capture_result["frames"]
    if frame_idx < 0 or frame_idx >= len(frames):
        return None, JSONResponse(
            status_code=404,
            content={"error": f"Frame index {frame_idx} out of range (0-{len(frames)-1})"},
        )
    return frames[frame_idx], None


def _load_drilldown_tensor(frame: dict[str, Any]):
    """Memmap-read a frame's full-resolution cached tensor (raw float32 bytes
    written by capture._cache_full_tensor) as a read-only ndarray, or return
    None if no cache exists for this frame. Bounds RAM: only the served frame
    is materialized, and reductions in _pca_2d produce copies rather than
    mutating the mapped buffer.
    """
    path = frame.get("tensor_path")
    shape = frame.get("tensor_shape")
    if not path or not shape:
        return None
    if not os.path.isfile(path):
        return None
    return np.memmap(path, dtype=np.float32, mode="r", shape=tuple(shape))


@app.get("/api/frame/{frame_idx}/tensor")
async def get_frame_tensor(frame_idx: int):
    """Return the FULL-resolution raw activation tensor for the drill-down, as
    binary little-endian float32 (pixel-exact, no downsampling).

    One-time fetch the client caches: channel grid + per-channel strip are
    computed from this, and /pca reuses it server-side. Shape/min/max/dtype
    travel in response headers so the body is pure float32 bytes (compact +
    fast to decode into a Float32Array, unlike the old JSON `.tolist()`).
    """
    frame, err = _frame_or_404(frame_idx)
    if err:
        return err
    arr = _load_drilldown_tensor(frame)
    if arr is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": (
                    "No drill-down tensor cached for this frame. Re-run inference "
                    "(older capture predates the drill-down cache)."
                ),
            },
        )
    headers = {
        "X-Tensor-Shape": ",".join(str(int(d)) for d in arr.shape),
        "X-Original-Shape": ",".join(str(int(d)) for d in frame["original_shape"]),
        "X-Tensor-Min": str(frame.get("tensor_min")),
        "X-Tensor-Max": str(frame.get("tensor_max")),
        "X-Dtype": str(frame.get("dtype", "float32")),
    }
    return Response(
        content=arr.tobytes(),
        media_type="application/octet-stream",
        headers=headers,
    )


# Cap on PCA points returned, so the scatter payload stays small and the
# plot readable. Rows are stride-subsampled when the sample count exceeds it.
_PCA_MAX_POINTS = 8000


def _pca_2d(
    tensor: np.ndarray,
    feature_axis: int,
    sample_axes: list[int],
    channel_axis: int | None,
) -> dict[str, Any]:
    """Project a (role-assigned) tensor to 2D via PCA on the feature dimension.

    Args:
        tensor: cached subsampled raw tensor (already batch-dropped).
        feature_axis: axis index used as the vector dimension D.
        sample_axes: axis indices flattened into samples N (rows).
        channel_axis: optional axis index folded into samples whose index
            becomes a per-point colour label.

    Returns: {x, y, label (or null), n, d}.
    """
    ndim = tensor.ndim
    if ndim == 0:
        raise ValueError("tensor has no axes")
    if not 0 <= feature_axis < ndim:
        raise ValueError(f"feature_axis {feature_axis} out of range (0-{ndim-1})")
    for a in sample_axes:
        if not 0 <= a < ndim:
            raise ValueError(f"sample_axis {a} out of range (0-{ndim-1})")
    if feature_axis in sample_axes:
        raise ValueError("feature_axis cannot also be a sample axis")
    if channel_axis is not None:
        if not 0 <= channel_axis < ndim:
            raise ValueError(f"channel_axis {channel_axis} out of range (0-{ndim-1})")
        if channel_axis == feature_axis or channel_axis in sample_axes:
            raise ValueError("channel_axis must differ from feature and sample axes")

    samp = sorted(set(sample_axes))
    chan = channel_axis
    used = {feature_axis, *samp}
    if chan is not None:
        used.add(chan)
    reduce_axes = [a for a in range(ndim) if a not in used]

    t = tensor.astype(np.float32, copy=False)
    if reduce_axes:
        t = t.mean(axis=tuple(reduce_axes))
        remaining = [a for a in range(ndim) if a not in reduce_axes]
    else:
        remaining = list(range(ndim))
    pos = {a: i for i, a in enumerate(remaining)}

    feat_pos = pos[feature_axis]
    samp_positions = [pos[a] for a in samp]
    chan_pos = pos[chan] if chan is not None else None

    order = samp_positions + ([chan_pos] if chan_pos is not None else []) + [feat_pos]
    t = np.transpose(t, order)

    feat_size = int(t.shape[-1])
    if feat_size < 2:
        raise ValueError("feature dimension < 2 — nothing to project")

    # Per-row channel label (channel index cycles fastest in C-order flatten).
    label = None
    if chan_pos is not None:
        chan_count = int(t.shape[-2])
        shape_pre = t.shape[:-1]
        label = np.broadcast_to(np.arange(chan_count), shape_pre).reshape(-1).astype(np.int32)

    n_rows = int(np.prod(t.shape[:-1])) if t.ndim > 1 else 1
    if n_rows < 2:
        raise ValueError("fewer than 2 samples — nothing to project")

    mat = t.reshape(n_rows, feat_size)
    # Subsample rows if very large, keeping labels aligned.
    if n_rows > _PCA_MAX_POINTS:
        stride = int(math.ceil(n_rows / _PCA_MAX_POINTS))
        mat = mat[::stride]
        if label is not None:
            label = label[::stride]
        n_rows = mat.shape[0]

    # Centre, then PCA via covariance eigh (N >= D) or SVD (N < D).
    mat = mat - mat.mean(axis=0, keepdims=True)
    if n_rows >= feat_size:
        cov = (mat.T @ mat) / max(1, n_rows - 1)
        evals, evecs = np.linalg.eigh(cov)
        comps = evecs[:, np.argsort(evals)[::-1][:2]]   # D x 2
    else:
        _, _, vt = np.linalg.svd(mat, full_matrices=False)
        comps = vt[:2].T                               # D x 2
    proj = mat @ comps                                 # N x 2

    return {
        "x": proj[:, 0].tolist(),
        "y": proj[:, 1].tolist(),
        "label": label.tolist() if label is not None else None,
        "n": n_rows,
        "d": feat_size,
    }


@app.get("/api/frame/{frame_idx}/pca")
async def get_frame_pca(
    frame_idx: int,
    feature_axis: int,
    sample_axes: str,
    channel_axis: int = -1,
):
    """2D PCA projection of a frame's cached raw tensor.

    Query params (axis indices into the tensor_shape):
      - feature_axis: vector dimension D.
      - sample_axes: comma-separated axis indices flattened to samples N.
      - channel_axis: optional axis folded into samples; its index becomes a
        per-point colour label. -1 (default) = none.

    The tensor is the full-resolution activation (no subsampling); PCA projects
    over every sample but the returned scatter is capped to _PCA_MAX_POINTS.
    Returns: {x, y, label (or null), n, d}.
    """
    frame, err = _frame_or_404(frame_idx)
    if err:
        return err
    raw = _load_drilldown_tensor(frame)
    if raw is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No drill-down tensor cached for this frame. Re-run inference."},
        )
    try:
        samp = [int(x) for x in sample_axes.split(",") if x.strip() != ""]
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "sample_axes must be comma-separated ints"})
    if not samp:
        return JSONResponse(status_code=400, content={"error": "sample_axes is required"})
    chan = None if channel_axis is None or channel_axis < 0 else int(channel_axis)
    try:
        return _pca_2d(raw, int(feature_axis), samp, chan)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ── Connection strength between two layers ─────────────────────────────────
# Used by the connection-visualization highlight: when a layer is clicked, the
# frontend fetches this for each incident edge to show how strongly the source
# and destination activations relate. Both tensors are the cached subsampled
# raw activations from capture._subsample_tensor.


def _channel_mean_vec(tensor: np.ndarray) -> np.ndarray:
    """Reduce a tensor to a 1-D per-channel-mean vector (axis 0 = channel).

    Shape-agnostic reduction so two layers with different spatial sizes can be
    compared: [C,H,W] -> mean over H,W -> length C; [N,D] -> mean over D ->
    length N; 1-D -> itself.
    """
    t = np.asarray(tensor, dtype=np.float32)
    if t.ndim == 0:
        t = t.reshape(1)
    if t.ndim == 1:
        return t
    return t.mean(axis=tuple(range(1, t.ndim)))


def _resample_vec(v: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a 1-D vector to length n (up or down)."""
    v = np.asarray(v, dtype=np.float32)
    if v.shape[0] == n or n <= 0:
        return v
    if v.shape[0] == 1:
        return np.full(n, float(v[0]), dtype=np.float32)
    xp = np.arange(v.shape[0], dtype=np.float32)
    xi = np.linspace(0, v.shape[0] - 1, n, dtype=np.float32)
    return np.interp(xi, xp, v).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tensor_energy(tensor: np.ndarray) -> float:
    t = np.asarray(tensor, dtype=np.float32)
    if t.size == 0:
        return 0.0
    return float(max(abs(float(t.min())), abs(float(t.max()))))


@app.get("/api/frames/pair-strength")
async def get_pair_strength(a: int, b: int):
    """Activation-based connection strength between two frames (layers).

    Cosine similarity between the two full-resolution tensors' per-channel-mean
    vectors (resampled to a common length when channel counts differ) —
    well-defined for skip/residual links (same channel count -> direct cosine
    = how much the destination preserves the source's per-channel pattern) with
    a reasonable fallback otherwise. Also returns each side's activation energy
    (peak magnitude) and the tensor shapes.

    Returns: {similarity (0-1), cosine (-1..1), energy_a, energy_b, shape_a, shape_b}.
    """
    fa, err = _frame_or_404(a)
    if err:
        return err
    fb, err2 = _frame_or_404(b)
    if err2:
        return err2
    ta = _load_drilldown_tensor(fa)
    tb = _load_drilldown_tensor(fb)
    if ta is None or tb is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No drill-down tensor cached for one or both frames. Re-run inference."},
        )
    va = _channel_mean_vec(ta)
    vb = _channel_mean_vec(tb)
    n = min(int(va.shape[0]), int(vb.shape[0]))
    n = max(1, n)
    cos = _cosine(_resample_vec(va, n), _resample_vec(vb, n))
    return {
        "similarity": (cos + 1.0) / 2.0,
        "cosine": cos,
        "energy_a": _tensor_energy(ta),
        "energy_b": _tensor_energy(tb),
        "shape_a": list(fa.get("tensor_shape", list(np.asarray(ta).shape))),
        "shape_b": list(fb.get("tensor_shape", list(np.asarray(tb).shape))),
    }


@app.get("/api/output/{output_idx}")
async def get_output(output_idx: int):
    """Retrieve raw output tensor data.

    Returns: {name, shape, values (or null if too large)}
    """
    if _capture_result is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No inference results. Call /api/inference first."},
        )

    outputs = _capture_result["outputs"]
    if output_idx < 0 or output_idx >= len(outputs):
        return JSONResponse(
            status_code=404,
            content={"error": f"Output index {output_idx} out of range (0-{len(outputs)-1})"},
        )

    output = outputs[output_idx]
    raw = output.get("raw")

    # Serialize: if already has 'values' (small tensor), use it
    if output["values"] is not None:
        values = output["values"]
    elif raw is not None:
        # Decide whether to serialize based on size
        if raw.size < 50000:
            values = raw.tolist()
        else:
            # For large tensors, return shape + stats only
            values = None
    else:
        values = None

    return {
        "name": output["name"],
        "shape": output["shape"],
        "values": values,
        "stats": {
            "min": float(np.min(raw)) if raw is not None else None,
            "max": float(np.max(raw)) if raw is not None else None,
            "mean": float(np.mean(raw)) if raw is not None else None,
            "std": float(np.std(raw)) if raw is not None else None,
        } if raw is not None else None,
        "truncated": values is None and raw is not None,
    }


@app.get("/api/status")
async def get_status():
    """Return current session status."""
    return {
        "model_loaded": _introspector is not None,
        "model_path": _model_path,
        "inference_run": _capture_result is not None,
        "num_frames": len(_capture_result["frames"]) if _capture_result else 0,
        "num_outputs": len(_capture_result["outputs"]) if _capture_result else 0,
        "brightness_mode": _brightness_mode,
        "preserve_nodes": _preserve_nodes,
    }


# -----------------------------------------------------------------------
# WebSocket
# -----------------------------------------------------------------------


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming frame metadata.

    Client sends a JSON message to trigger inference:
      {"action": "inference", "use_random": true, "brightness_mode": "mean"}
    Server streams:
      {"type": "progress", "captured": N, "total": M}
      {"type": "frame", "frame": {...metadata}}
      {"type": "complete", "total_frames": N, "skipped_count": M}
      {"type": "error", "error": "message"}
    """
    await websocket.accept()

    global _capture_result

    try:
        while True:
            # Wait for client message
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("action") != "inference":
                await websocket.send_json({
                    "type": "error",
                    "error": "Unknown action. Use {\"action\": \"inference\", ...}",
                })
                continue

            if _introspector is None:
                await websocket.send_json({
                    "type": "error",
                    "error": "No model loaded. Call /api/load-model first.",
                })
                continue

            use_random = msg.get("use_random", True)
            brightness_mode = msg.get("brightness_mode", "mean")
            preserve_nodes = msg.get("preserve_nodes", False)
            dynamic_dims = msg.get("dynamic_dims")

            try:
                # Prepare inputs
                input_handler = InputHandler(_introspector.inputs)
                input_feed = input_handler.load_inputs(
                    use_random=use_random,
                    dynamic_dims=dynamic_dims,
                )

                # Run capture
                intermediate_names = _introspector.get_intermediate_tensor_names()
                capture = ActivationCapture(
                    model_path=_model_path,
                    intermediate_names=intermediate_names,
                    brightness_mode=brightness_mode,
                    preserve_nodes=preserve_nodes,
                )

                result = capture.run_inference(input_feed)
                _capture_result = result

                # Stream frame metadata
                total = len(result["frames"])
                for i, frame in enumerate(result["frames"]):
                    await websocket.send_json({
                        "type": "frame",
                        "frame": _frame_metadata(frame),
                        "progress": i + 1,
                        "total": total,
                    })

                await websocket.send_json({
                    "type": "complete",
                    "total_frames": total,
                    "skipped_count": len(result["skipped"]),
                    "outputs_count": len(result["outputs"]),
                    "skipped": result["skipped"],
                })

            except Exception as e:
                logger.exception("WebSocket inference failed")
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass


# -----------------------------------------------------------------------
# Root — serve index.html from static
# -----------------------------------------------------------------------


@app.get("/")
async def root():
    """Serve the main HTML page from static directory."""
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.isfile(index_path):
        from fastapi.responses import FileResponse

        return FileResponse(index_path)
    return JSONResponse(
        status_code=404,
        content={"error": "index.html not found in static directory"},
    )


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8765,
        reload=True,
        log_level="info",
    )
