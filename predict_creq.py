import onnx
from onnx import shape_inference
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from networkx.drawing.nx_agraph import to_agraph, graphviz_layout
from collections import defaultdict, deque


def get_shape_dict(graph):
    shape_dict = {}

    def extract_shape(value_info):
        shape = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            elif dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append('?')  # 미지정 차원
        return shape

    # 입력, 출력, 중간 값들 모두에서 shape 수집
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        name = value.name
        shape = extract_shape(value)
        shape_dict[name] = shape

    return shape_dict

def get_value_info_map(graph):
    tensor_info = {}

    # 1. from value_info, inputs, outputs (inferred ones)
    def extract(info):
        type_proto = info.type.tensor_type
        elem_type = onnx.TensorProto.DataType.Name(type_proto.elem_type)
        shape = [d.dim_value if d.HasField("dim_value") else '?' for d in type_proto.shape.dim]
        return (elem_type, shape)

    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        tensor_info[value_info.name] = extract(value_info)

    # 2. from initializers (e.g., weights/biases)
    for initializer in graph.initializer:
        name = initializer.name
        elem_type = onnx.TensorProto.DataType.Name(initializer.data_type)
        shape = list(initializer.dims)
        tensor_info[name] = (elem_type, shape)

    return tensor_info

def get_node_degrees(graph):
    # print(graph)

    for node_name, attrs in graph.nodes(data=True):
        in_degree = graph.in_degree(node_name)
        out_degree = graph.out_degree(node_name)
        graph.nodes[node_name]['in_degree']  = in_degree
        graph.nodes[node_name]['out_degree'] = out_degree

        if in_degree > 0 and out_degree > 0:
            if in_degree < out_degree:
                graph.nodes[node_name]['is_branch'] = True
            elif in_degree > out_degree:
                graph.nodes[node_name]['is_merge'] = True

        # print(f"Node: {node_name}, In-Degree: {attrs['in_degree']}, Out-Degree: {attrs['out_degree']}, Is Branch: {attrs['is_branch']}, Is Merge: {attrs['is_merge']}")

def resolve_dynamic_shapes(model, batch_size=1):
    for tensor in model.graph.input:
        shape = tensor.type.tensor_type.shape
        for dim in shape.dim:
            if dim.dim_param:  # dynamic ('batch_size')인 경우
                dim.dim_value = batch_size
                dim.ClearField("dim_param")  # 'batch_size' 제거

    for tensor in model.graph.output:
        shape = tensor.type.tensor_type.shape
        for dim in shape.dim:
            if dim.dim_param:  # dynamic ('batch_size')인 경우
                dim.dim_value = batch_size
                dim.ClearField("dim_param")  # 'batch_size' 제거

    for tensor in model.graph.value_info:
        shape = tensor.type.tensor_type.shape
        for dim in shape.dim:
            if dim.dim_param:  # dynamic ('batch_size')인 경우
                dim.dim_value = batch_size
                dim.ClearField("dim_param")  # 'batch_size' 제거
    return model


def is_branch_node(graph, node):
    return len(graph[node]) >= 2

def is_merge_node(graph, node):
    cnt = 0
    for u in graph:
        for v in graph[u]:
            if v == node:
                cnt += 1
    return cnt >= 2

def extract_subgraph(graph, start, end):
    subgraph = defaultdict(list)
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        for nxt in graph[node]:
            subgraph[node].append(nxt)
            if nxt != end:
                stack.append(nxt)
    return subgraph

def merge_ONNX_graph(graph, values):
    memo = {}
    def dfs(current, parent):
        if current in memo:
            return memo[current]
        if not graph[current]:  #merge node
            memo[parent] = values[parent]
            return values[parent]

        nexts = graph[current]
        sub_results = [dfs(next_node, current) for next_node in nexts]

        if len(nexts) == 1:
            req = max(values[current], sub_results[0])
        else:
            req = sum(sub_results)

        memo[current] = req
        return req
    return dfs

def path_exists(graph, start, end):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node == end:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(graph[node])
    return False

def compute_graph_weight(graph, values):
    while True:
        found = False
        branch_nodes = [n for n in graph if is_branch_node(graph, n)]
        merge_nodes = [n for n in graph if is_merge_node(graph, n)]

        for branch in branch_nodes:
            for merge in merge_nodes:
                if branch not in graph or merge not in graph:
                    continue
                if not path_exists(graph, branch, merge):
                    continue

                # safe check
                try:
                    subgraph = extract_subgraph(graph, branch, merge)
                    sub_values = {k: values[k] for k in subgraph}
                except KeyError:
                    continue  # skip if any key missing due to earlier deletion

                dfs = merge_ONNX_graph(subgraph, sub_values)
                merged_value = dfs(branch, branch)
                print(f"Merged {branch} -> {merge} = {merged_value}")

                # update value
                values[branch] = merged_value

                # redirect edges
                graph[branch] = [merge]

                # delete intermediate nodes
                for node in subgraph:
                    if node != branch and node != merge:
                        graph.pop(node, None)
                        values.pop(node, None)

                found = True
                break  # restart outer loop
            if found:
                break
        if not found:
            break  # no more merges

    # compute total weight from modified graph
    dfs = merge_ONNX_graph(graph, values)
    first_node = next(iter(graph))

    return dfs(first_node,first_node)

# result = compute_graph_weight(graph, values)
# print("Total weight:", result)

def calculate_node_flops(node, flat_shapes, initializers):
    def get_shape(name):
        return flat_shapes.get(name) or flat_shapes.get(name.split('/')[-1])

    op = node.op_type
    inputs = node.input

    if op == "Conv":
        input_tensor = get_shape(inputs[0])
        weight_tensor = get_shape(inputs[1])

        if weight_tensor is None and initializers:
            for init in initializers:
                if init.name == inputs[1]:
                    weight_tensor = list(init.dims)
                    flat_shapes[inputs[1]] = weight_tensor
                    break

        if input_tensor is None or weight_tensor is None:
            return 0

        batch_size = input_tensor[0]
        in_c = input_tensor[1]
        in_h, in_w = input_tensor[2], input_tensor[3]
        out_c, _, k_h, k_w = weight_tensor

        stride = [1, 1]
        pads = [0, 0, 0, 0]

        for attr in node.attribute:
            if attr.name == "strides":
                stride = list(attr.ints)
            elif attr.name == "pads":
                pads = list(attr.ints)

        out_h = int((in_h + pads[0] + pads[2] - k_h) / stride[0]) + 1
        out_w = int((in_w + pads[1] + pads[3] - k_w) / stride[1]) + 1

        return batch_size * out_c * out_h * out_w * (in_c * k_h * k_w * 2)

    elif op == "Gemm":
        A = get_shape(inputs[0])
        B = get_shape(inputs[1])
        if A is None or B is None:
            return 0
        M, K = A[-2], A[-1]
        N = B[-1]
        return 2 * M * N * K

    elif op == "MatMul":
        A = get_shape(inputs[0])
        B = get_shape(inputs[1])
        if A is None or B is None:
            return 0
        M, K = A[-2], A[-1]
        K2, N = B[-2], B[-1]
        if K != K2:
            return 0
        return 2 * M * N * K

    elif op == "MaxPool":
        input_tensor = get_shape(inputs[0])
        if input_tensor is None:
            return 0

        batch_size, in_c, in_h, in_w = input_tensor
        kernel_shape = [1, 1]
        stride = [1, 1]
        pads = [0, 0, 0, 0]

        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_shape = list(attr.ints)
            elif attr.name == "strides":
                stride = list(attr.ints)
            elif attr.name == "pads":
                pads = list(attr.ints)

        out_h = int((in_h + pads[0] + pads[2] - kernel_shape[0]) / stride[0]) + 1
        out_w = int((in_w + pads[1] + pads[3] - kernel_shape[1]) / stride[1]) + 1

        return batch_size * in_c * out_h * out_w * (kernel_shape[0] * kernel_shape[1] - 1)

    elif op == "Relu":
        A = get_shape(inputs[0])
        if A is None:
            return 0
        return np.prod(A)

    elif op == "Add":
        A = get_shape(inputs[0])
        B = get_shape(inputs[1])
        if A is None or B is None:
            return 0
        return np.prod(A)  # Element-wise add: 1 FLOP per element

    elif op == "BatchNormalization":
        A = get_shape(inputs[0])
        if A is None:
            return 0
        # 4 ops per element (sub, mul, add, div) is common
        return 4 * np.prod(A)

    elif op == "GlobalAveragePool":
        input_tensor = get_shape(inputs[0])
        if input_tensor is None:
            return 0
        batch_size, in_c, h, w = input_tensor
        return batch_size * in_c * (h * w - 1 + 1)  # (n-1) adds + 1 div per channel

    elif op == "Flatten":
        A = get_shape(inputs[0])
        if A is None:
            return 0
        return 0  # Just reshaping, no FLOPs

    else:
        return 0



def onnx_to_dag_with_shapes(onnx_model_path, batch_size=1):
    model = onnx.load(onnx_model_path)
    model = shape_inference.infer_shapes(model)
    model = resolve_dynamic_shapes(model, batch_size)
    graph = model.graph

    tensor_info_map = get_value_info_map(graph)
    initializers = list(graph.initializer)  # for weight lookup in Conv, etc.

    dag = nx.DiGraph()

    for node in graph.node:
        node_name = node.name or f"{node.op_type}_{id(node)}"

        input_shapes = {inp: tensor_info_map.get(inp, None) for inp in node.input}
        output_shapes = {out: tensor_info_map.get(out, None) for out in node.output}

        # flatten shape info to feed into calculate_node_flops
        flat_shapes = {}
        for k, v in input_shapes.items():
            if v:
                flat_shapes[k] = v[1]
        for k, v in output_shapes.items():
            if v:
                flat_shapes[k] = v[1]

        flops = calculate_node_flops(node, flat_shapes, initializers)

        dag.add_node(
            node_name,
            op_type=node.op_type,
            inputs=node.input,
            outputs=node.output,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            in_degree=0,
            out_degree=0,
            mreq=flops,  # Now stores FLOPs instead of memory size
            is_branch=False,
            is_merge=False,
        )

        for input_tensor in node.input:
            for prev_node in graph.node:
                if input_tensor in prev_node.output:
                    prev_node_name = prev_node.name or f"{prev_node.op_type}_{id(prev_node)}"
                    dag.add_edge(prev_node_name, node_name, tensor=input_tensor)

    get_node_degrees(dag)
    dag_dict = {node: list(dag.successors(node)) for node in dag.nodes()}

    return dag, dag_dict


def format_node_label(node_name, node_data):
    label = f"{node_name}\n"
    label += f"Op: {node_data['op_type']}\n"
    label += f"In: {', '.join(node_data['inputs'])}\n"
    label += f"Out: {', '.join(node_data['outputs'])}\n"
    label += f"Shape: {node_data['output_shapes']}"
    return label

# dag, dag_dict = onnx_to_dag_with_shapes("models/resnet152-v1-7.onnx", 4)
dag, dag_dict = onnx_to_dag_with_shapes("models/model.onnx", 4)
# dag, dag_dict = onnx_to_dag_with_shapes("models/yolov4.onnx", 4)

print(dag)

print("==" * 20)

values = {node: int(attrs['mreq']) for node, attrs in dag.nodes(data=True)}
print("Values:", values)

print("==" * 20)

result = compute_graph_weight(dag_dict, values)
print("Total weight:", result)


# for node_name, attrs in dag.nodes(data=True):
#     print(f"Node: {node_name}")
#     print(f"  Op Type: {attrs['op_type']}")
#     # print(f"  Inputs: {attrs['inputs']}")
#     print(f"  Input Shapes: {attrs['input_shapes']}")
#     # print(f"  Outputs: {attrs['outputs']}")
#     print(f"  Output Shapes: {attrs['output_shapes']}")
#     # print(f"  In-Degree: {attrs['in_degree']}")
#     # print(f"  Out-Degree: {attrs['out_degree']}")
#     # print(f"  Value Info: {attrs['input_shapes']}")
#     print(f"  MReq: {attrs['mreq']:,}")
#     print(f"  Is Branch: {attrs['is_branch']}")
#     print(f"  Is Merge: {attrs['is_merge']}")
#     print()

# 계층적 레이아웃 시도
# pos = nx.spring_layout(dag)  # 기본 레이아웃
# A = to_agraph(dag)
# A.graph_attr.update(ranksep="20", nodesep="10")

# 레이아웃 및 시각화
# labels = {
#     node: format_node_label(node, data)
#     for node, data in dag.nodes(data=True)
# }
# pos = graphviz_layout(dag, prog='dot')
# plt.figure(figsize=(10, 6))
# nx.draw(dag, pos, labels=labels, with_labels=True, node_size=500, node_color="lightblue", font_size=8, arrows=True)
# plt.title("DAG with Node Attributes")
# plt.tight_layout()
# plt.show()
