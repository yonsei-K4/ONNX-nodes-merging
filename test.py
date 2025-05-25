import onnx
import numpy as np
import networkx as nx
from onnx import numpy_helper, mapping, shape_inference

def get_tensor_size_bytes(data_type, shape):
    if 0 in shape or len(shape) == 0:
        return 0
    try:
        np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[data_type]
    except KeyError:
        return 0
    return np.prod(shape) * np.dtype(np_dtype).itemsize

def extract_shapes(graph):
    """Return a dict: tensor_name -> (dtype, shape)"""
    shape_info = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        name = vi.name
        tt = vi.type.tensor_type
        if tt.HasField("shape"):
            dims = [d.dim_value if d.HasField("dim_value") else 0 for d in tt.shape.dim]
            shape_info[name] = (tt.elem_type, dims)
    return shape_info

def build_graph(onnx_model):
    G = nx.DiGraph()
    for node in onnx_model.graph.node:
        for input_name in node.input:
            for output_name in node.output:
                G.add_edge(input_name, output_name)
    return G


def simulate_peak_memory(onnx_model):
    graph = onnx_model.graph
    inferred_model = shape_inference.infer_shapes(onnx_model)
    shape_info = extract_shapes(inferred_model.graph)

    # Build execution DAG
    G = build_graph(inferred_model)

    # Topological sort for execution order
    exec_order = list(nx.topological_sort(G))

    # Track: tensor_name -> (birth_time, death_time, size)
    tensor_lifetimes = {}
    time = 0

    for node_or_tensor in exec_order:
        if node_or_tensor in [i.name for i in graph.initializer]:
            continue  # skip constant initializers

        if node_or_tensor in [node.name for node in graph.node]:
            node = next(n for n in graph.node if n.name == node_or_tensor)

            # All outputs are born now
            for out in node.output:
                if out in shape_info:
                    dtype, shape = shape_info[out]
                    size = get_tensor_size_bytes(dtype, shape)
                    tensor_lifetimes[out] = {'birth': time, 'death': None, 'size': size}
        else:
            # it's a tensor name: may be input/output/initializer
            continue
        time += 1

    # Death time: last usage of each tensor
    for node in graph.node:
        for input_name in node.input:
            if input_name in tensor_lifetimes:
                tensor_lifetimes[input_name]['death'] = max(
            tensor_lifetimes[input_name]['death'] or -1,
            exec_order.index(node.name)
        )

    # Activation memory timeline simulation
    timeline = {}
    for t_name, info in tensor_lifetimes.items():
        birth = info['birth']
        death = info['death'] if info['death'] is not None else birth
        for t in range(birth, death + 1):
            timeline.setdefault(t, 0)
            timeline[t] += info['size']

    # Peak memory usage
    peak_memory = max(timeline.values()) if timeline else 0
    return peak_memory / 1024**2  # MB

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python estimate_peak_memory.py model.onnx")
        exit(1)

    model_path = sys.argv[1]
    model = onnx.load(model_path)
    peak_mem_mb = simulate_peak_memory(model)

    print(f"Estimated peak GPU memory usage: {peak_mem_mb:.2f} MB")
