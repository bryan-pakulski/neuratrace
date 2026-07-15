"""ORT inference & activation capture — brightness normalization for visualization."""

from __future__ import annotations

import atexit
import logging
import math
import os
import shutil
import tempfile
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

logger = logging.getLogger(__name__)

# ORT dtype map: ONNX elem_type → numpy dtype string
_ORT_DTYPE_MAP: dict[int, str] = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32", 7: "int64", 9: "bool", 10: "float16", 11: "float64",
    12: "uint32", 13: "uint64", 16: "bfloat16",
}

# Pure graph-bookkeeping ops whose outputs are shape/index metadata, not
# neuron activations. Excluded from the visualization (float dtype filter
# catches the rest). Kept deliberately small — Reshape/Transpose/Cast/Slice
# can act on real activations, so they stay.
_META_OPS: frozenset[str] = frozenset({
    "Constant", "ConstantOfShape", "Shape",
    "Unsqueeze", "Squeeze", "Gather", "Expand", "Where", "Equal",
})

# Disk cache for the drill-down inspector's FULL-resolution raw tensors.
# /tmp is tmpfs (RAM-backed) on the dev machine, so we default to a real-disk
# path under ~/.cache to actually bound RAM — only the served frame is loaded
# into memory at a time (the tensor/PCA endpoints memmap-read on demand).
# Override with the ONNX_VIEWER_DRILLDOWN_DIR env var. The dir is wiped at the
# start of each inference and on process exit. See `_cache_full_tensor`.
DRILLDOWN_CACHE_DIR = os.environ.get("ONNX_VIEWER_DRILLDOWN_DIR") or os.path.expanduser(
    "~/.cache/onnx_viewer_drilldown"
)


def _reset_drilldown_cache() -> str:
    """Wipe + recreate the drill-down tensor cache dir; return its path.

    Called at the start of each inference so a new run's `.bin` files don't
    mix with the previous capture's (whose frames may still be referenced by
    the server's `_capture_result` until this point).
    """
    path = DRILLDOWN_CACHE_DIR
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
    except OSError as e:
        logger.warning("Could not clear drill-down cache %s: %s", path, e)
    os.makedirs(path, exist_ok=True)
    return path


def _cleanup_drilldown_cache() -> None:
    try:
        if os.path.isdir(DRILLDOWN_CACHE_DIR):
            shutil.rmtree(DRILLDOWN_CACHE_DIR)
    except OSError:
        pass


atexit.register(_cleanup_drilldown_cache)


class ActivationCapture:
    """Run ONNX Runtime inference, capture intermediate activations, compute brightness grids.

    Usage:
        introspector = OnnxGraphIntrospector(model_path)
        introspector.parse()
        intermediate = introspector.get_intermediate_tensor_names()
        capture = ActivationCapture(model_path, intermediate, brightness_mode='mean')
        result = capture.run_inference(input_feed)
        # result.frames → list of brightness grid dicts
        # result.outputs → raw graph output tensors
    """

    def __init__(
        self,
        model_path: str,
        intermediate_names: list[dict[str, Any]],
        brightness_mode: str = "mean",
        preserve_nodes: bool = False,
    ):
        """Initialize capture.

        Args:
            model_path: Path to ONNX model file.
            intermediate_names: List of dicts from OnnxGraphIntrospector.get_intermediate_tensor_names().
                Each: {tensor_name, node_name, op_type, exec_order}
            brightness_mode: 'mean' or 'max' — channel reduction strategy.
            preserve_nodes: If True, disable ORT graph optimization (preserves all nodes).
        """
        self.model_path = model_path
        self.intermediate_names = intermediate_names
        self.brightness_mode = brightness_mode
        self.preserve_nodes = preserve_nodes
        self._session: ort.InferenceSession | None = None
        self._tmp_model_path: str | None = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _augment_model_outputs(self) -> str:
        """Add intermediate tensor names as graph outputs in a modified ONNX model.

        ORT only lets you request names that appear in the model's graph.output list.
        Intermediate node outputs are normally not graph outputs, so we augment the
        model by adding each intermediate tensor as an additional graph output, then
        save to a temp file.

        Returns:
            Path to the modified temp ONNX model file.
        """
        model = onnx.load(self.model_path)
        graph = model.graph

        # Run shape inference to populate value_info with correct types
        # for all intermediate tensors. This avoids type mismatches when
        # adding tensors whose output type isn't float (e.g. ConstantOfShape
        # outputs int64).
        try:
            model = onnx.shape_inference.infer_shapes(model)
            graph = model.graph
            logger.info("Shape inference completed: %d value_info entries", len(graph.value_info))
        except Exception as e:
            logger.warning("Shape inference failed (%s) — using original value_info only", e)

        # Build set of existing graph output names
        existing_output_names = {o.name for o in graph.output}

        # Build a map of tensor name → type_proto from value_info
        vi_type_map = {vi.name: vi.type for vi in graph.value_info}

        # Also build a map from initializer names → their elem_type
        init_type_map = {init.name: init.data_type for init in graph.initializer}

        added = 0
        skipped_no_type = 0
        for desc in self.intermediate_names:
            tname = desc["tensor_name"]
            if tname in existing_output_names:
                continue

            if tname in vi_type_map:
                # Use inferred type info (correct dtype + shape)
                vi = onnx.ValueInfoProto()
                vi.name = tname
                vi.type.CopyFrom(vi_type_map[tname])
                graph.output.append(vi)
                added += 1
            elif tname in init_type_map:
                # It's an initializer — use its known data type
                vi = onnx.ValueInfoProto()
                vi.name = tname
                vi.type.tensor_type.elem_type = init_type_map[tname]
                graph.output.append(vi)
                added += 1
            else:
                # No type info available even after shape inference.
                # Skip adding as graph output — ORT's session.run(output_names=[...])
                # can still request it at runtime without a declared type.
                # Hardcoding FLOAT would cause type mismatches for non-float tensors.
                skipped_no_type += 1
                logger.debug(
                    "Skipping '%s' (node '%s', op %s) — no type info, will request at runtime",
                    tname, desc.get("node_name", "?"), desc.get("op_type", "?"),
                )

        logger.info(
            "Augmented model with %d intermediate output tensors (%d skipped — no type info)",
            added, skipped_no_type,
        )

        # Clamp IR version to ORT's max supported version.
        # onnx 1.20.1 defaults to IR 13, but ORT 1.23.2 only supports IR ≤ 11.
        # If the original model has IR > ORT's max, ORT will reject it on load.
        _ORT_MAX_IR = 11  # ORT 1.23.x max supported IR version
        if model.ir_version > _ORT_MAX_IR:
            logger.info(
                "Clamping IR version %d → %d for ORT compatibility",
                model.ir_version, _ORT_MAX_IR,
            )
            model.ir_version = _ORT_MAX_IR

        # Save to temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix="_augmented.onnx", prefix="onnx_viewer_")
        os.close(tmp_fd)
        onnx.save(model, tmp_path)
        self._tmp_model_path = tmp_path
        return tmp_path

    def _create_session(self) -> ort.InferenceSession:
        """Create ORT session with intermediate tensors exposed as outputs.

        Two strategies:
        1. preserve_nodes=True: ORT_DISABLE_ALL optimization, augment model outputs.
        2. preserve_nodes=False: ORT_ENABLE_ALL but still augment (some may get fused).
        """
        # Augment model to expose intermediate tensors as graph outputs
        augmented_path = self._augment_model_outputs()

        sess_options = ort.SessionOptions()

        if self.preserve_nodes:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            logger.info("ORT graph optimization DISABLED (preserve_nodes=True)")
        else:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Try CUDA first, fall back to CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(
                augmented_path,
                sess_options=sess_options,
                providers=providers,
            )
            active = session.get_providers()
            logger.info("ORT session created with providers: %s", active)
        except Exception as e:
            logger.warning("CUDA provider failed (%s), falling back to CPU", e)
            session = ort.InferenceSession(
                augmented_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            logger.info("ORT session created with CPU-only provider")

        # ORT has now read the augmented model into memory — remove the temp
        # file so repeated inference runs don't accumulate large copies in /tmp
        # (one augmented ONNX per /api/inference call, often tens of MB).
        try:
            if self._tmp_model_path and os.path.exists(self._tmp_model_path):
                os.remove(self._tmp_model_path)
                self._tmp_model_path = None
        except OSError:
            pass

        # Log available outputs for debugging
        avail = [o.name for o in session.get_outputs()]
        logger.info("Session has %d available outputs", len(avail))

        return session

    def _get_session(self) -> ort.InferenceSession:
        """Get or create ORT session (lazy)."""
        if self._session is None:
            self._session = self._create_session()
        return self._session

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference(self, input_feed: dict[str, np.ndarray]) -> dict[str, Any]:
        """Run inference and capture all intermediate activations.

        Args:
            input_feed: Dict mapping input tensor names to numpy arrays.

        Returns:
            Dict with:
              - frames: list of brightness grid dicts, sorted by exec_order
              - outputs: list of raw graph output arrays
              - skipped: list of {tensor_name, node_name, reason} for missing tensors
              - total_requested: number of intermediate tensors requested
              - total_captured: number successfully captured
        """
        # Fresh disk cache for this run's full-resolution drill-down tensors
        # (drops the previous inference's .bin files).
        _reset_drilldown_cache()
        session = self._get_session()

        # Build output name list: graph outputs + intermediate tensor names
        # After augmentation, intermediate tensors that survive optimization are
        # exposed as session outputs. Some may still be fused away by ORT even
        # when added to graph.output — those will simply be absent from
        # session.get_outputs() and we skip requesting them.
        session_output_names = {o.name for o in session.get_outputs()}
        intermediate_tensor_name_set = {d["tensor_name"] for d in self.intermediate_names}
        # Original graph outputs = session outputs minus augmented intermediates
        graph_output_names = [o.name for o in session.get_outputs()
                              if o.name not in intermediate_tensor_name_set]

        intermediate_tensor_names = [d["tensor_name"] for d in self.intermediate_names]

        # Only request names that the session actually exposes
        all_output_names = list(graph_output_names) + [
            n for n in intermediate_tensor_names
            if n in session_output_names and n not in graph_output_names
        ]

        # Track which intermediates are NOT available (fused/optimized away)
        unavailable = [n for n in intermediate_tensor_names if n not in session_output_names]

        logger.info(
            "Running inference: %d graph outputs, %d intermediate tensors requested, "
            "%d unavailable (fused/optimized away)",
            len(graph_output_names),
            len(all_output_names) - len(graph_output_names),
            len(unavailable),
        )

        # Run inference
        raw_results = session.run(output_names=all_output_names, input_feed=input_feed)

        # Build result map: name → numpy array
        result_map: dict[str, np.ndarray] = {}
        for name, arr in zip(all_output_names, raw_results):
            # Ensure CPU numpy array (no GPU tensor retention)
            if hasattr(arr, "get"):
                arr = arr.get()  # cupy → numpy
            result_map[name] = np.ascontiguousarray(arr)

        # Extract graph outputs
        outputs = []
        for name in graph_output_names:
            if name in result_map:
                arr = result_map[name]
                # Sanitize non-finite floats so JSON serialization can't crash
                # (some outputs — logits, divisions — can be inf/nan).
                safe = self._sanitize_finite(arr) if arr.dtype.kind == "f" else arr
                outputs.append({
                    "name": name,
                    "values": safe.tolist() if arr.size < 10000 else None,
                    "shape": list(arr.shape),
                    "raw": safe,  # keep (sanitized) raw for server to serialize selectively
                })

        # Build frames from intermediate tensors
        frames = []
        skipped = []

        # Sort intermediate_names by exec_order
        sorted_intermediate = sorted(self.intermediate_names, key=lambda d: d["exec_order"])

        frame_idx = 0
        for desc in sorted_intermediate:
            tensor_name = desc["tensor_name"]
            if tensor_name not in result_map:
                skipped.append({
                    "tensor_name": tensor_name,
                    "node_name": desc["node_name"],
                    "op_type": desc["op_type"],
                    "reason": "optimized_away",
                })
                logger.warning(
                    "Tensor '%s' (node '%s', op %s) not in ORT output — likely fused/optimized away",
                    tensor_name, desc["node_name"], desc["op_type"],
                )
                continue

            tensor = result_map[tensor_name]

            # Keep only real activations: floating-point tensors whose producer
            # isn't a pure graph-bookkeeping op. Integer/bool tensors are shape/
            # index metadata (Constant, Shape, Gather, …), not neuron outputs —
            # visualizing them adds ~57% noise (single points / 1D bars) to the
            # cube. See architecture review.
            if tensor.dtype.kind != "f":
                skipped.append({
                    "tensor_name": tensor_name,
                    "node_name": desc["node_name"],
                    "op_type": desc["op_type"],
                    "reason": "non_float_metadata",
                })
                continue
            if desc["op_type"] in _META_OPS:
                skipped.append({
                    "tensor_name": tensor_name,
                    "node_name": desc["node_name"],
                    "op_type": desc["op_type"],
                    "reason": "metadata_op",
                })
                continue

            brightness = self._compute_brightness(tensor, self.brightness_mode)

            # Cache the FULL-resolution raw tensor to disk for the drill-down
            # inspector (channel grid / PCA / per-channel strip) so the
            # per-channel views are pixel-exact (no striding). Only
            # /api/frame/{idx}/tensor + /pca read it (memmap on demand), so RAM
            # stays bounded to one frame at a time.
            tensor_path, tensor_shape, tmin, tmax = self._cache_full_tensor(
                tensor, frame_idx
            )

            frame = {
                "frame_idx": frame_idx,
                "tensor_name": tensor_name,
                "node_name": desc["node_name"],
                "op_type": desc["op_type"],
                "exec_order": desc["exec_order"],
                "grid": brightness["grid"],
                "shape": brightness["shape"],
                "original_shape": brightness["original_shape"],
                "is_uniform": brightness["is_uniform"],
                "sampled": brightness["sampled"],
                "raw_min": brightness["raw_min"],
                "raw_max": brightness["raw_max"],
                "dtype": str(tensor.dtype),
                # Drill-down only (disk-backed, not served by /api/frame):
                "tensor_path": tensor_path,
                "tensor_shape": tensor_shape,
                "tensor_min": tmin,
                "tensor_max": tmax,
            }
            frames.append(frame)
            frame_idx += 1

        logger.info(
            "Inference complete: %d frames captured, %d skipped",
            len(frames), len(skipped),
        )

        return {
            "frames": frames,
            "outputs": outputs,
            "skipped": skipped,
            "total_requested": len(self.intermediate_names),
            "total_captured": len(frames),
        }

    # ------------------------------------------------------------------
    # Brightness computation
    # ------------------------------------------------------------------

    def _cache_full_tensor(
        self, tensor: np.ndarray, frame_idx: int
    ) -> tuple[str, list[int], float, float]:
        """Sanitize + cache the FULL-resolution raw tensor to disk for the
        drill-down inspector (channel grid / PCA / per-channel strip), so the
        per-channel views are pixel-exact (no striding/downsampling).

        Drops a leading batch-1 axis (mirroring `_compute_brightness`'s
        batch-0 take), sanitizes non-finite values, casts to float32, and
        writes the C-contiguous bytes to `<DRILLDOWN_CACHE_DIR>/frame_{idx}.bin`
        (raw float32, no header — the shape is stored separately on the frame).
        Bounds RAM: the file is memmap-read on demand by the tensor/PCA
        endpoints, so only one frame is in memory at a time.

        Returns:
            (path, shape, finite min, finite max) where min/max are computed
            over the full tensor's raw values.
        """
        arr = np.asarray(tensor)
        arr = arr.astype(np.float32, copy=False)
        if arr.ndim >= 1 and arr.shape[0] == 1:
            arr = arr[0]            # drop leading batch-1 axis
        if arr.ndim == 0:
            arr = arr.reshape(1)
        arr = self._sanitize_finite(arr)
        arr = np.ascontiguousarray(arr)
        path = os.path.join(DRILLDOWN_CACHE_DIR, f"frame_{frame_idx}.bin")
        arr.tofile(path)
        return path, list(arr.shape), float(arr.min()), float(arr.max())

    def _compute_brightness(
        self,
        tensor: np.ndarray,
        mode: str = "mean",
    ) -> dict[str, Any]:
        """Compute 2D brightness grid from activation tensor.

        Handles 1D-4D tensors:
          - 4D [N,C,H,W] → mean/max over C → 2D [H,W]
          - 3D [S,C] → reshape S to near-square → reduce C → 2D
          - 2D [C,F] → reshape to near-square → 2D
          - 1D [N] → bar strip [1, N]

        Normalizes to [0,1] per-frame min-max with 1e-8 epsilon.
        Detects dead layers (uniform activation → 0.5 + is_uniform flag).
        Subsamples if >100k elements via stride=2 until <25k.

        Args:
            tensor: Raw activation array from ORT.
            mode: 'mean' or 'max' for channel reduction.

        Returns:
            Dict: {grid, shape, original_shape, is_uniform, sampled}
        """
        original_shape = list(tensor.shape)

        # Cast non-float to float32
        if tensor.dtype not in (np.float32, np.float64):
            tensor = tensor.astype(np.float32)
        else:
            tensor = tensor.astype(np.float32)

        # Always batch index 0
        if tensor.ndim == 0:
            # Scalar → 1x1 grid
            grid = np.array([[float(tensor)]], dtype=np.float32)
            return self._normalize_grid(grid, original_shape, False)

        if tensor.ndim >= 1 and tensor.shape[0] == 0:
            # Empty tensor → 1x1 zero grid
            grid = np.zeros((1, 1), dtype=np.float32)
            return self._normalize_grid(grid, original_shape, True)

        # Take batch index 0 for 4D/3D with batch dim
        # `reduce_src` holds the pre-reduction (batch-0) tensor for any branch
        # that collapses an axis, so a bad reduction can be undone (below).
        reduce_src = None
        if tensor.ndim == 4:
            # [N, C, H, W] → take batch 0 → [C, H, W]
            t = tensor[0]
            reduce_src = t
            # Reduce over channel axis
            if mode == "max":
                grid_2d = t.max(axis=0)
            else:
                grid_2d = t.mean(axis=0)
            grid_2d = np.asarray(grid_2d, dtype=np.float32)

        elif tensor.ndim == 3:
            # Could be [N, C, H*W] or [N, S, C] or [C, H, W]
            # Heuristic: if shape[0] is small (≤4) treat as [N, C, F]
            # else treat as [S, C, F] or [C, H, W]
            if tensor.shape[0] <= 4 and tensor.ndim == 3:
                # [N, C, F] → take batch 0 → [C, F]
                t = tensor[0]
            else:
                # [C, H, W] or [S, C, F] → use as-is
                t = tensor

            if t.ndim == 3:
                # [C, H, W] → reduce C
                reduce_src = t
                if mode == "max":
                    grid_2d = t.max(axis=0)
                else:
                    grid_2d = t.mean(axis=0)
            else:
                # [C, F] → reshape to near-square
                grid_2d = self._reshape_near_square(t)
            grid_2d = np.asarray(grid_2d, dtype=np.float32)

        elif tensor.ndim == 2:
            # [C, F] or [H, W] → reshape to near-square for display
            grid_2d = self._reshape_near_square(tensor)

        elif tensor.ndim == 1:
            # [N] → near-square so 1D activations fill the slab instead of
            # collapsing to a single edge line in the 3D cube.
            grid_2d = self._reshape_near_square(tensor)

        else:
            # >4D: flatten and reshape
            flat = tensor.reshape(-1)
            grid_2d = self._reshape_near_square(flat.reshape(1, -1))

        # A channel reduction assumes axis 1 is "channels". Attention tensors
        # are [N, tokens, heads, head_dim] — axis 1 is tokens, so reducing it
        # collapses e.g. [1,257,6,64] (98688 elts) to a [6,64] strip that
        # renders as a few striped lines. When the reduction yields a thin
        # strip (small min-dim + high aspect), the collapsed axis was not
        # channels: fall back to a near-square flattening of the full batch-0
        # tensor, preserving the real per-element variation.
        if reduce_src is not None and grid_2d.ndim == 2 and grid_2d.size > 1:
            r, c = grid_2d.shape
            mn = min(r, c)
            if mn <= 8 and max(r, c) / max(1, mn) > 3:
                grid_2d = self._reshape_near_square(reduce_src)

        # Collapse any residual single-axis bar ([1,N] or [N,1]) to a
        # near-square so every slab fills the cube's XY footprint instead of
        # rendering as one edge line (e.g. attention [400,1,128] → [1,128]).
        if grid_2d.ndim == 2 and grid_2d.size > 1 and (
            grid_2d.shape[0] == 1 or grid_2d.shape[1] == 1
        ):
            grid_2d = self._reshape_near_square(grid_2d)

        # Subsample if too large
        sampled = False
        total_elements = grid_2d.size
        if total_elements > 100_000:
            while grid_2d.size > 25_000 and grid_2d.shape[0] > 2 and grid_2d.shape[1] > 2:
                grid_2d = grid_2d[::2, ::2]
                sampled = True

        return self._normalize_grid(grid_2d, original_shape, sampled)

    def _normalize_grid(
        self,
        grid: np.ndarray,
        original_shape: list[int],
        sampled: bool,
    ) -> dict[str, Any]:
        """Normalize grid to [0,1] with min-max + epsilon, detect dead layers.

        Args:
            grid: 2D float32 array.
            original_shape: Shape of the original tensor before reduction.
            sampled: Whether subsampling was applied.

        Returns:
            Dict: {grid, shape, original_shape, is_uniform, sampled}
        """
        # Replace NaN/±Inf before min/max + tolist(): json.dumps rejects
        # non-finite floats and some activations (div-by-zero, log(0)) produce
        # them. ±Inf → nearest finite extremum, NaN → 0.
        grid = self._sanitize_finite(grid)

        gmin = float(grid.min())
        gmax = float(grid.max())
        grange = gmax - gmin

        is_uniform = grange < 1e-8

        if is_uniform:
            # Dead layer — set to 0.5 (mid-gray)
            normalized = np.full_like(grid, 0.5, dtype=np.float32)
        else:
            normalized = (grid - gmin) / (grange + 1e-8)
            normalized = normalized.astype(np.float32)

        return {
            "grid": normalized.tolist(),
            "shape": list(normalized.shape),
            "original_shape": original_shape,
            "is_uniform": is_uniform,
            "sampled": sampled,
            # Pre-normalization min/max of the (reduced, subsampled) brightness
            # grid. Lets the frontend reconstruct each cell's network-relative
            # magnitude for a global color scale without shipping raw tensors.
            "raw_min": gmin,
            "raw_max": gmax,
        }

    @staticmethod
    def _sanitize_finite(arr: np.ndarray) -> np.ndarray:
        """Return a float32 array with NaN/±Inf replaced by finite values.

        NaN → 0.0, +Inf → max finite value present, -Inf → min finite value
        present (0.0 if none finite). Integer/bool arrays pass through.
        Guarantees the result is JSON-serializable (no non-finite floats).
        """
        a = np.asarray(arr)
        if a.dtype.kind != "f":
            return a
        a = a.astype(np.float32, copy=False)
        if np.all(np.isfinite(a)):
            return a
        finite = a[np.isfinite(a)]
        fmin = finite.min() if finite.size else np.float32(0.0)
        fmax = finite.max() if finite.size else np.float32(0.0)
        a = a.copy()
        a[np.isposinf(a)] = fmax
        a[np.isneginf(a)] = fmin
        a[np.isnan(a)] = np.float32(0.0)
        return a

    @staticmethod
    def _reshape_near_square(arr: np.ndarray) -> np.ndarray:
        """Reshape 2D [C, F] or 1D [N] to a near-square 2D grid.

        For [C, F]: flatten to C*F, reshape to near-square.
        For [1, N]: reshape to near-square.

        Args:
            arr: 1D or 2D array.

        Returns:
            2D near-square array.
        """
        if arr.ndim == 1:
            total = arr.shape[0]
        else:
            total = arr.size

        if total == 0:
            return np.zeros((1, 1), dtype=np.float32)

        # Find near-square dimensions
        side = int(math.isqrt(total))
        if side * side == total:
            h, w = side, side
        elif side * (side + 1) >= total:
            h, w = side, side + 1
        else:
            h, w = side + 1, side + 1

        # Pad if needed
        needed = h * w
        flat = arr.reshape(-1).astype(np.float32)
        if flat.size < needed:
            flat = np.pad(flat, (0, needed - flat.size), mode="constant")

        return flat[:needed].reshape(h, w)