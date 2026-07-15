"""Input shape resolution and data loading for ONNX inference.

Handles:
  - Dynamic dimension resolution (None / symbolic → concrete values)
  - Image file loading (JPG/PNG/BMP) with auto-resize to model input HxW
  - Numpy file loading (.npy/.npz) with shape validation
  - Random Gaussian input generation
  - Multiple model inputs via dict {name: array}
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ONNX dtype string → numpy dtype
_DTYPE_TO_NUMPY: dict[str, np.dtype] = {
    "FLOAT": np.float32,
    "UINT8": np.uint8,
    "INT8": np.int8,
    "UINT16": np.uint16,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "BOOL": np.bool_,
    "FLOAT16": np.float16,
    "DOUBLE": np.float64,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "BFLOAT16": np.float32,  # bfloat16 not natively supported, use float32
}

# Default values for common symbolic dim names
_DEFAULT_DIM_VALUES: dict[str, int] = {
    "batch": 1,
    "batch_size": 1,
    "N": 1,
    "n": 1,
    "seq_len": 128,
    "sequence_length": 128,
    "seq_length": 128,
    "S": 128,
    "s": 128,
    "num_channels": 3,
    "C": 3,
    "c": 3,
    "height": 224,
    "H": 224,
    "h": 224,
    "width": 224,
    "W": 224,
    "w": 224,
    "num_classes": 80,
    "num_heads": 8,
    "head_dim": 64,
    "hidden_size": 256,
}

# Image file extensions
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_NUMPY_EXTS = {".npy", ".npz"}

# Dim sizes that are almost certainly a channel count (not spatial).
_CHANNEL_SIZES = {1, 2, 3, 4}


def _to_numpy_dtype(dtype_name: str) -> np.dtype:
    """Convert ONNX dtype string to numpy dtype. Falls back to float32."""
    return _DTYPE_TO_NUMPY.get(dtype_name, np.float32)


def _is_channel_dim(dim: Any) -> bool:
    """True if a dim is a concrete int that looks like a channel count."""
    return isinstance(dim, int) and dim in _CHANNEL_SIZES


def _detect_layout(shape: tuple) -> str:
    """Infer whether a 4D image input is NCHW or NHWC.

    Heuristic: the channel axis is the one holding a small concrete size
    (1/2/3/4) while the two spatial axes are large or dynamic. Channels-first
    → NCHW, channels-last → NHWC. Falls back to NCHW when ambiguous.

    Examples:
      [1, 3, 224, 224]      → NCHW   (channel at index 1)
      [1, 224, 224, 3]      → NHWC   (channel at index 3)
      [unk, unk, unk, 3]    → NHWC   (only the last dim is channel-like)
      [unk, 3, unk, unk]    → NCHW
    """
    if len(shape) != 4:
        return "NCHW"
    _, d1, d2, d3 = shape
    ch_first = _is_channel_dim(d1) and not _is_channel_dim(d2)
    ch_last = _is_channel_dim(d3) and not _is_channel_dim(d2)
    if ch_first and not ch_last:
        return "NCHW"
    if ch_last and not ch_first:
        return "NHWC"
    # Ambiguous (e.g. all dynamic, or two small dims) — prefer NCHW unless only
    # the last axis is channel-like.
    if ch_last:
        return "NHWC"
    return "NCHW"


def _dim_default(shape: list, i: int, layout: str | None = None) -> int:
    """Position-based default for a dynamic dim, layout-aware for 4D images.

    Mirrored by the frontend's ``_defaultDimValue`` so UI and backend agree.
    """
    if i == 0:
        return 1  # batch / N
    if len(shape) == 4:
        lay = layout or _detect_layout(shape)
        if lay == "NHWC":
            if i == 1:
                return 224  # H
            if i == 2:
                return 224  # W
            if i == 3:
                return 3   # C (rarely dynamic)
        else:  # NCHW
            if i == 1:
                return 3   # C
            if i == 2:
                return 224  # H
            if i == 3:
                return 224  # W
        return 1
    if len(shape) == 2 and i == 1:
        return 128  # [batch, seq_len]
    return 1


class InputHandler:
    """Resolve dynamic input dimensions and load/generate input data for ONNX models.

    Args:
        input_specs: List of TensorSpec (from OnnxGraphIntrospector.inputs) or
                     list of dicts with keys {name, shape, dtype}.
    """

    def __init__(self, input_specs: list[Any]):
        self.input_specs: list[dict[str, Any]] = []
        for spec in input_specs:
            if hasattr(spec, "name") and hasattr(spec, "shape") and hasattr(spec, "dtype"):
                # TensorSpec dataclass
                self.input_specs.append({
                    "name": spec.name,
                    "shape": list(spec.shape),
                    "dtype": spec.dtype,
                })
            elif isinstance(spec, dict):
                self.input_specs.append({
                    "name": spec["name"],
                    "shape": list(spec.get("shape", [])),
                    "dtype": spec.get("dtype", "FLOAT"),
                })
            else:
                raise ValueError(f"Unsupported input spec type: {type(spec)}")

        self.resolved_shapes: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------------
    # Shape resolution
    # ------------------------------------------------------------------

    def resolve_shapes(
        self,
        dynamic_dims: dict[str, dict[str, int]] | None = None,
    ) -> dict[str, tuple[int, ...]]:
        """Resolve all dynamic dimensions to concrete values.

        Args:
            dynamic_dims: Optional user-provided values.
                Format: {input_name: {dim_name_or_index: value}}
                Example: {"images": {"batch": 1, "height": 640, "width": 640}}
                Or by index: {"images": {0: 1, 2: 640, 3: 640}}

        Returns:
            Dict mapping input name → concrete shape tuple.
        """
        dynamic_dims = dynamic_dims or {}
        self.resolved_shapes = {}

        for spec in self.input_specs:
            name = spec["name"]
            shape = spec["shape"]
            user_vals = dynamic_dims.get(name, {})
            layout = _detect_layout(shape) if len(shape) == 4 else "NCHW"
            resolved = []

            for i, dim in enumerate(shape):
                if isinstance(dim, int) and dim > 0:
                    # Concrete dimension — use as-is
                    resolved.append(dim)
                else:
                    # Dynamic: None or symbolic string
                    dim_key = dim if isinstance(dim, str) else None
                    value = None

                    # 1. Check user-provided by dim name
                    if dim_key and dim_key in user_vals:
                        value = user_vals[dim_key]
                    # 2. Check user-provided by index
                    elif i in user_vals:
                        value = user_vals[i]
                    elif str(i) in user_vals:
                        value = user_vals[str(i)]
                    # 3. Default by symbolic name
                    elif dim_key and dim_key in _DEFAULT_DIM_VALUES:
                        value = _DEFAULT_DIM_VALUES[dim_key]
                    # 4. Position-based, layout-aware defaults
                    else:
                        value = _dim_default(shape, i, layout)

                    resolved.append(int(value))

            self.resolved_shapes[name] = tuple(resolved)
            logger.debug("Resolved input '%s': %s → %s", name, shape, resolved)

        return self.resolved_shapes

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(
        self,
        file_path: str,
        target_shape: tuple[int, ...],
        layout: str = "auto",
        channel_order: str = "RGB",
    ) -> np.ndarray:
        """Load an image file and convert to a numpy array matching ``target_shape``.

        Handles both NCHW ``[N, C, H, W]`` and NHWC ``[N, H, W, C]`` layouts
        (common for models exported from frameworks like TensorFlow), and
        optional RGB↔BGR channel remapping.

        Args:
            file_path: Path to image file (JPG/PNG/BMP/TIFF/WEBP).
            target_shape: 4D shape tuple in the model's native layout.
            layout: ``"auto"`` (infer from ``target_shape``), ``"NCHW"``, or
                ``"NHWC"``.
            channel_order: ``"RGB"`` (default, matches PIL) or ``"BGR"`` (swaps
                the R and B channels — useful for OpenCV-trained models).

        Returns:
            Numpy array of shape ``target_shape``, float32, normalized [0, 1].
        """
        if len(target_shape) != 4:
            raise ValueError(
                f"load_image expects a 4D target shape, got {target_shape}"
            )

        if layout == "auto":
            layout = _detect_layout(target_shape)
        layout = "NHWC" if layout.upper() == "NHWC" else "NCHW"

        if layout == "NHWC":
            n, h, w, c = target_shape
        else:  # NCHW
            n, c, h, w = target_shape

        img = Image.open(file_path)
        if c == 1:
            img = img.convert("L")
        elif c == 4:
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")
        img = img.resize((w, h), Image.BILINEAR)

        # HWC (or HW for grayscale) → float32, normalized to [0, 1]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, None]

        # Match the target channel count (pad / truncate / replicate).
        if arr.shape[2] != c:
            if arr.shape[2] > c:
                arr = arr[:, :, :c]
            elif c == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            else:
                pad = np.zeros((h, w, c - arr.shape[2]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=2)

        # Channel order remap (RGB ↔ BGR). Swap R(0) and B(2), keep the rest.
        if channel_order.upper() == "BGR" and arr.shape[2] >= 3:
            order = list(range(arr.shape[2]))
            order[0], order[2] = order[2], order[0]
            arr = arr[:, :, order]

        # Arrange into the model's native layout and add the batch dim.
        if layout == "NHWC":
            out = np.stack([arr] * n, axis=0)             # N, H, W, C
        else:
            chw = np.transpose(arr, (2, 0, 1))            # C, H, W
            out = np.stack([chw] * n, axis=0)             # N, C, H, W

        return out

    # ------------------------------------------------------------------
    # Numpy file loading
    # ------------------------------------------------------------------

    def load_numpy(self, file_path: str, input_name: str) -> np.ndarray:
        """Load a .npy or .npz file and validate against resolved shape.

        Args:
            file_path: Path to .npy or .npz file.
            input_name: Name of the model input to validate against.

        Returns:
            Numpy array matching the resolved shape for the given input.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in _NUMPY_EXTS:
            raise ValueError(f"Unsupported numpy file extension: {ext}")

        if ext == ".npy":
            arr = np.load(file_path, allow_pickle=False)
        else:
            # .npz — load and find first array or one matching input_name
            npz = np.load(file_path, allow_pickle=False)
            if input_name in npz:
                arr = npz[input_name]
            else:
                arr = npz[npz.files[0]]
            npz.close()

        # Validate shape if resolved
        if input_name in self.resolved_shapes:
            expected = self.resolved_shapes[input_name]
            if arr.shape != tuple(expected):
                raise ValueError(
                    f"Numpy array shape {arr.shape} does not match expected "
                    f"shape {expected} for input '{input_name}'"
                )

        return arr

    # ------------------------------------------------------------------
    # Fill / inline input generation
    # ------------------------------------------------------------------

    def generate_fill(
        self,
        name: str,
        value: float | int | bool,
    ) -> np.ndarray:
        """Generate a constant-filled array for one input.

        Args:
            name: Model input name (must be in resolved_shapes).
            value: Fill value (cast to the input's dtype).

        Returns:
            Numpy array with the resolved shape, filled with ``value``.
        """
        if name not in self.resolved_shapes:
            self.resolve_shapes()
        shape = self.resolved_shapes[name]
        spec = next(s for s in self.input_specs if s["name"] == name)
        np_dtype = _to_numpy_dtype(spec["dtype"])
        # Integer dtypes require an integer fill value.
        if np.issubdtype(np_dtype, np.integer):
            value = int(value)
        return np.full(shape, value, dtype=np_dtype)

    def load_inline(self, values: list | int | float, input_name: str) -> np.ndarray:
        """Build an array from a JSON-ready inline value list.

        Accepts a flat (``[640, 480]``) or nested (``[[640], [480]]``) list and
        reshapes it to the input's resolved shape. The total element count must
        match the resolved shape's product.

        Args:
            values: JSON-ready nested/flat list of numbers (or a scalar).
            input_name: Model input name to validate against.

        Returns:
            Numpy array with the resolved shape and input dtype.
        """
        if input_name not in self.resolved_shapes:
            self.resolve_shapes()
        expected = self.resolved_shapes[input_name]
        spec = next(s for s in self.input_specs if s["name"] == input_name)
        np_dtype = _to_numpy_dtype(spec["dtype"])

        arr = np.array(values, dtype=np_dtype)
        expected_size = int(np.prod(expected)) if expected else 1
        if arr.size != expected_size:
            raise ValueError(
                f"Inline values for '{input_name}' have {arr.size} element(s) "
                f"but the resolved shape {list(expected)} requires {expected_size}."
            )
        try:
            arr = arr.reshape(expected)
        except ValueError as e:
            raise ValueError(
                f"Cannot reshape inline values for '{input_name}' to "
                f"{list(expected)}: {e}"
            ) from e
        return arr

    # ------------------------------------------------------------------
    # Random input generation
    # ------------------------------------------------------------------

    def generate_random(self) -> dict[str, np.ndarray]:
        """Generate Gaussian random input for all model inputs.

        Returns:
            Dict mapping input name → numpy array with resolved shape.
        """
        if not self.resolved_shapes:
            self.resolve_shapes()

        result = {}
        for spec in self.input_specs:
            name = spec["name"]
            shape = self.resolved_shapes[name]
            np_dtype = _to_numpy_dtype(spec["dtype"])

            if spec["dtype"] in ("INT32", "INT64", "UINT32", "UINT64"):
                # Integer dtypes: generate random non-negative integers
                arr = np.random.randint(0, 100, size=shape).astype(np_dtype)
            elif spec["dtype"] == "BOOL":
                arr = np.random.choice([True, False], size=shape)
            else:
                # Float dtypes: standard Gaussian noise
                arr = np.random.randn(*shape).astype(np_dtype)

            result[name] = arr

        logger.info("Generated random input for %d tensors", len(result))
        return result

    # ------------------------------------------------------------------
    # Multi-input loading
    # ------------------------------------------------------------------

    def load_inputs(
        self,
        file_paths: dict[str, str] | None = None,
        dynamic_dims: dict[str, dict[str, int]] | None = None,
        use_random: bool = False,
        input_config: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, np.ndarray]:
        """High-level: resolve shapes and load/generate all inputs.

        Per-input precedence:
          1. **file** — if a path is present in ``file_paths`` it is always
             used (image auto-resized to resolved H/W; numpy shape-validated).
          2. **input_config mode** — ``zeros`` / ``ones`` / ``constant`` /
             ``inline`` fill the array deterministically.
          3. **random** — fallback for inputs with neither file nor config.

        ``use_random`` forces all-random only when neither files nor an
        ``input_config`` are supplied (the no-file JSON quick path).

        Args:
            file_paths: Dict mapping input name → file path.
            dynamic_dims: User-provided dynamic dimension values.
            use_random: If True AND no files/config are provided, generate
                        random input for all inputs.
            input_config: Per-input data-source config, e.g.
                ``{"orig_target_sizes": {"mode": "zeros"}}`` or
                ``{"images": {"mode": "constant", "value": 0.5}}`` or
                ``{"sizes": {"mode": "inline", "value": [640, 480]}}``.

        Returns:
            Dict mapping input name → numpy array.
        """
        self.resolve_shapes(dynamic_dims)
        self.input_sources: dict[str, str] = {}

        file_paths = file_paths or {}
        input_config = input_config or {}

        # No files and no config → random for everything (quick path).
        if not file_paths and not input_config and use_random:
            result = self.generate_random()
            self.input_sources = {name: "random" for name in result}
            return result

        result = {}

        for spec in self.input_specs:
            name = spec["name"]
            cfg = input_config.get(name) or {}
            if name in file_paths:
                path = file_paths[name]
                ext = os.path.splitext(path)[1].lower()
                if ext in _IMAGE_EXTS:
                    result[name] = self.load_image(
                        path,
                        self.resolved_shapes[name],
                        layout=cfg.get("layout", "auto"),
                        channel_order=cfg.get("channel_order", "RGB"),
                    )
                elif ext in _NUMPY_EXTS:
                    result[name] = self.load_numpy(path, name)
                else:
                    raise ValueError(f"Unsupported file type for input '{name}': {ext}")
                self.input_sources[name] = "file"
                continue

            mode = cfg.get("mode") if cfg else None
            if mode in ("zeros", "ones", "constant", "inline"):
                if mode == "zeros":
                    result[name] = self.generate_fill(name, 0)
                elif mode == "ones":
                    result[name] = self.generate_fill(name, 1)
                elif mode == "constant":
                    result[name] = self.generate_fill(name, cfg.get("value", 0))
                else:  # inline
                    result[name] = self.load_inline(cfg.get("value"), name)
                self.input_sources[name] = mode
                continue

            # No file and no fill config — generate random for this input.
            shape = self.resolved_shapes[name]
            np_dtype = _to_numpy_dtype(spec["dtype"])
            if spec["dtype"] in ("INT32", "INT64", "UINT32", "UINT64"):
                result[name] = np.random.randint(0, 100, size=shape).astype(np_dtype)
            elif spec["dtype"] == "BOOL":
                result[name] = np.random.choice([True, False], size=shape)
            else:
                result[name] = np.random.randn(*shape).astype(np_dtype)
            self.input_sources[name] = "random"

        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_dynamic_dims(self) -> dict[str, list[dict[str, Any]]]:
        """Return info about dynamic dimensions for UI display.

        Returns:
            Dict mapping input name → list of dynamic dim descriptors.
            Each descriptor: {index, dim_name (or None), default_value}
        """
        result = {}
        for spec in self.input_specs:
            name = spec["name"]
            shape = spec["shape"]
            layout = _detect_layout(shape) if len(shape) == 4 else "NCHW"
            dynamics = []
            for i, dim in enumerate(shape):
                if not (isinstance(dim, int) and dim > 0):
                    dim_name = dim if isinstance(dim, str) else None
                    if dim_name and dim_name in _DEFAULT_DIM_VALUES:
                        default = _DEFAULT_DIM_VALUES[dim_name]
                    else:
                        default = _dim_default(shape, i, layout)

                    dynamics.append({
                        "index": i,
                        "dim_name": dim_name,
                        "default_value": default,
                    })
            if dynamics:
                result[name] = dynamics

        return result