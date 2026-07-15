"""ONNX graph introspection — parse model structure for visualization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import onnx
from onnx import TensorProto

logger = logging.getLogger(__name__)

# elem_type int → human-readable name
_DTYPE_MAP: dict[int, str] = {
    1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16",
    6: "INT32", 7: "INT64", 9: "BOOL", 10: "FLOAT16", 11: "DOUBLE",
    12: "UINT32", 13: "UINT64", 16: "BFLOAT16",
}


def _dtype_name(elem_type: int) -> str:
    return _DTYPE_MAP.get(elem_type, f"UNKNOWN({elem_type})")


def _extract_shape(value_info) -> list[int | str | None]:
    """Extract shape from ValueInfoProto, returning list where dims are
    int (concrete), str (symbolic dim_param), or None (unknown)."""
    shape = []
    tt = value_info.type.tensor_type
    if tt.HasField("shape"):
        for dim in tt.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)  # symbolic name like "batch_size"
            elif dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(None)
    else:
        # scalar — no shape field
        pass
    return shape


@dataclass
class NodeInfo:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    exec_order: int


@dataclass
class TensorSpec:
    name: str
    shape: list[int | str | None]
    dtype: str


class OnnxGraphIntrospector:
    """Load an ONNX model and introspect its computation graph.

    Provides:
      - inputs / outputs / nodes in topological order
      - initializer names (weights — excluded from visualization)
      - intermediate tensor names (node outputs, minus initializers & graph outputs)
      - hierarchical node tree (path-based or op_type grouping)
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.graph = None
        self.inputs: list[TensorSpec] = []
        self.outputs: list[TensorSpec] = []
        self.initializer_names: set[str] = set()
        self.nodes: list[NodeInfo] = []
        self._output_to_node: dict[str, NodeInfo] = {}

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(self) -> None:
        """Load model and extract all graph metadata."""
        self.model = onnx.load(self.model_path)
        self.graph = self.model.graph

        # --- inputs ---
        for vi in self.graph.input:
            shape = _extract_shape(vi)
            dtype = _dtype_name(vi.type.tensor_type.elem_type)
            self.inputs.append(TensorSpec(vi.name, shape, dtype))

        # --- outputs ---
        for vi in self.graph.output:
            shape = _extract_shape(vi)
            dtype = _dtype_name(vi.type.tensor_type.elem_type)
            self.outputs.append(TensorSpec(vi.name, shape, dtype))

        # --- initializers (weights) ---
        self.initializer_names = {init.name for init in self.graph.initializer}

        # --- nodes (topological order) ---
        output_names_seen: set[str] = set()
        for idx, node in enumerate(self.graph.node):
            name = node.name if node.name else f"{node.op_type}_{idx}"
            inputs = list(node.input)
            # handle empty output names
            outputs = []
            for i, out_name in enumerate(node.output):
                if out_name:
                    outputs.append(out_name)
                else:
                    gen = f"{name}_output_{i}"
                    outputs.append(gen)
            node_info = NodeInfo(
                name=name,
                op_type=node.op_type,
                inputs=inputs,
                outputs=outputs,
                exec_order=idx,
            )
            self.nodes.append(node_info)
            for out in outputs:
                self._output_to_node[out] = node_info
                output_names_seen.add(out)

        logger.info(
            "Parsed ONNX model: %d nodes, %d inputs, %d outputs, %d initializers",
            len(self.nodes), len(self.inputs), len(self.outputs),
            len(self.initializer_names),
        )

    # ------------------------------------------------------------------
    # Intermediate tensor names
    # ------------------------------------------------------------------

    def get_intermediate_tensor_names(self) -> list[dict[str, Any]]:
        """Return list of intermediate tensor descriptors.

        Each dict: {tensor_name, node_name, op_type, exec_order}
        Excludes:
          - initializer names (weights)
          - graph output names (final outputs, handled separately)
        """
        graph_output_names = {o.name for o in self.outputs}
        result = []
        for node in self.nodes:
            for out in node.outputs:
                if out in self.initializer_names:
                    continue
                if out in graph_output_names:
                    continue
                result.append({
                    "tensor_name": out,
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "exec_order": node.exec_order,
                })
        return result

    # ------------------------------------------------------------------
    # Node tree
    # ------------------------------------------------------------------

    def build_node_tree(self) -> dict[str, Any]:
        """Build hierarchical tree for UI display.

        The tree is a leveled dataflow view: each node is placed at its
        topological depth (longest producer chain from a graph input /
        initializer), so parallel branches of the network sit at the same
        level. Depth groups are branches; nodes within a group are ordered
        by execution order (order of appearance in the model). No node is
        duplicated.

        Returns dict:
          {type: "dataflow", root: {children: [depth groups]}}
        """
        tree_type = "dataflow"
        root = self._build_dataflow_tree()
        return {"type": tree_type, "root": root}

    def _build_dataflow_tree(self) -> dict[str, Any]:
        """Build a leveled dataflow tree grouped by topological depth.

        Depth = longest path (in nodes) along *real* dataflow edges, so
        parallel branches of the network sit at the same level.

        Two refinements keep the leveling useful for real exports (which are
        full of ``Constant``/shape ops with no inputs):

          1. A node is "source-only" if none of its inputs is an intermediate
             tensor produced by another node (e.g. ``Constant`` has no inputs;
             the first ``Conv`` takes graph inputs + initializers). Source-only
             nodes are *ignored as parents* when computing depth — they don't
             push their consumers deeper, so the leveling reflects real
             computation, not the sea of constants.
          2. Source-only nodes are then repositioned to sit just before their
             earliest consumer (``depth = min(consumer depth) - 1``), so each
             constant/shape op appears next to where it is actually used
             instead of all piling up at depth 0.

        ``self.nodes`` is in topological (exec) order, so a single forward
        pass computes the real-node depths; the repositioning is a second
        pass after consumers are known.
        """
        is_source: dict[str, bool] = {}
        for node in self.nodes:
            has_real_prod = any(
                self._output_to_node.get(inp) is not None for inp in node.inputs
            )
            is_source[node.name] = not has_real_prod

        # Pass 1: longest-path depth over real (non-source) producer edges.
        depths: dict[str, int] = {}
        for node in self.nodes:
            parent_depths: list[int] = []
            for inp in node.inputs:
                prod = self._output_to_node.get(inp)
                if prod is not None and not is_source[prod.name]:
                    parent_depths.append(depths[prod.name])
            depths[node.name] = 0 if not parent_depths else 1 + max(parent_depths)

        # Consumer map: producer name -> list of consuming node names.
        consumers: dict[str, list[str]] = {n.name: [] for n in self.nodes}
        for node in self.nodes:
            for inp in node.inputs:
                prod = self._output_to_node.get(inp)
                if prod is not None:
                    consumers[prod.name].append(node.name)

        # Pass 2: reposition source-only nodes just before their first consumer.
        for node in self.nodes:
            if not is_source[node.name]:
                continue
            cons = [depths[c] for c in consumers[node.name] if c in depths]
            depths[node.name] = max(0, min(cons) - 1) if cons else 0

        by_depth: dict[int, list[dict[str, Any]]] = {}
        max_depth = 0
        for node in self.nodes:
            d = depths[node.name]
            max_depth = max(max_depth, d)
            by_depth.setdefault(d, []).append({
                "name": node.name,
                "op_type": node.op_type,
                "exec_order": node.exec_order,
                "outputs": node.outputs,
            })

        children = []
        for d in range(max_depth + 1):
            nodes = sorted(by_depth.get(d, []), key=lambda n: n["exec_order"])
            children.append({
                "name": f"depth {d}",
                "group": True,            # render as a collapsible branch
                "count": len(nodes),
                "children": [],
                "nodes": nodes,
            })

        return {"name": "root", "children": children, "nodes": []}

    def _build_path_tree(self) -> dict[str, Any]:
        """Build tree from '/'-delimited node name paths.

        Each node name like '/backbone/stages.0/blocks.0/conv1/Conv'
        becomes nested tree nodes. Leaf nodes carry node metadata.
        """
        root: dict[str, Any] = {
            "name": "",
            "children": {},
            "nodes": [],
        }

        for node in self.nodes:
            # strip leading '/' then split
            clean = node.name.lstrip("/")
            segments = [s for s in clean.split("/") if s]
            if not segments:
                segments = [node.name]

            cursor = root
            for i, seg in enumerate(segments):
                if seg not in cursor["children"]:
                    cursor["children"][seg] = {
                        "name": seg,
                        "children": {},
                        "nodes": [],
                    }
                cursor = cursor["children"][seg]
                if i == len(segments) - 1:
                    cursor["nodes"].append({
                        "name": node.name,
                        "op_type": node.op_type,
                        "exec_order": node.exec_order,
                        "outputs": node.outputs,
                    })

        # convert children dicts to sorted lists
        self._finalize_tree(root)
        return root

    def _build_optype_tree(self) -> dict[str, Any]:
        """Build flat tree grouped by op_type with counts."""
        groups: dict[str, list[dict]] = {}
        for node in self.nodes:
            groups.setdefault(node.op_type, []).append({
                "name": node.name,
                "op_type": node.op_type,
                "exec_order": node.exec_order,
                "outputs": node.outputs,
            })

        children = []
        for op_type, nodes in sorted(groups.items()):
            children.append({
                "name": f"{op_type} ({len(nodes)})",
                "op_type": op_type,
                "count": len(nodes),
                "children": [],
                "nodes": nodes,
            })

        return {
            "name": "root",
            "children": children,
            "nodes": [],
        }

    @staticmethod
    def _finalize_tree(node: dict[str, Any]) -> None:
        """Recursively convert children dict to sorted list."""
        children = node["children"]
        child_list = sorted(children.values(), key=lambda c: c["name"])
        node["children"] = child_list
        for child in child_list:
            OnnxGraphIntrospector._finalize_tree(child)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_all_output_names(self) -> list[str]:
        """Return all tensor names that ORT can produce (for session.run)."""
        names = []
        for node in self.nodes:
            names.extend(node.outputs)
        return names