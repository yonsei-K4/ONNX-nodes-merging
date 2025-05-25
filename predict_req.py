import sys
import onnx
from onnx import shape_inference
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from networkx.drawing.nx_agraph import to_agraph, graphviz_layout
from collections import defaultdict, deque

ONNX_TYPE_SIZE = {
    'FLOAT': 4,
    'FLOAT16': 2,
    'DOUBLE': 8,
    'INT32': 4,
    'INT64': 8,
    'UINT8': 1,
    'INT8': 1,
    'BOOL': 1,
    'UINT16': 2,
    'INT16': 2,
    'UINT32': 4,
    'UINT64': 8,
    'BFLOAT16': 2,
}

def get_shape_dict(graph):
    """그래프의 모든 텐서들에 대한 shape 정보를 딕셔너리로 수집"""
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

def compute_graph_weight(graph, values):
    stack = []

    # make a copy of graph 
    graph = {k: list(v) for k, v in graph.items()}
    # print("graph:", graph)

    for node in graph:
        if is_branch_node(graph, node):
            stack.append((0, node))
        if is_merge_node(graph, node):
            stack.append((1, node))

    # print("stack:", stack)

    while len(stack) >= 2:
        i = 0
        while i < len(stack) - 1:
            if stack[i][0] == 0 and stack[i + 1][0] == 1:
                branch = stack[i][1]
                merge = stack[i + 1][1]

                # Extract subgraph and compute value
                subgraph = extract_subgraph(graph, branch, merge)
                sub_values = {k: values[k] for k in subgraph}
                dfs = merge_ONNX_graph(subgraph, sub_values)
                merged_value = dfs(branch, branch)
                # print(merged_value)

                # update values: branch node gets the computed value
                values[branch] = merged_value

                # remove all outgoing edges from branch and redirect to merge
                graph[branch] = [merge]

                # remove all nodes in subgraph except branch and merge
                for node in subgraph:
                    if node != branch and node != merge:
                        graph.pop(node, None)
                        values.pop(node, None)

                # clean up 
                stack = stack[:i] + stack[i+2:]
                break  # restart loop from beginning
            i += 1

    # compute total weight from modified graph
    dfs = merge_ONNX_graph(graph, values)
    first_node = next(iter(graph))
    # first_value = values[first_node]

    return dfs(first_node, first_node)

# result = compute_graph_weight(graph, values)
# print("Total weight:", result)


def onnx_to_dag_with_shapes(onnx_model_path, batch_size=1):
    model = onnx.load(onnx_model_path)
    model = shape_inference.infer_shapes(model)
    model = resolve_dynamic_shapes(model, batch_size)
    graph = model.graph

    tensor_info_map = get_value_info_map(graph)

    # print(tensor_info_map)
    # print("==" * 20)

    dag = nx.DiGraph()
    i = 0

    for node in graph.node:
        node_name = node.name if node.name else f"{node.op_type}_{i}"
        # print(f'[{i}] {node_name}')
        # print(f'  Inputs: {node.input}')
        # print(f'  Outputs: {node.output}')

        input_shapes = {inp: tensor_info_map.get(inp, None) for inp in node.input}
        output_shapes = {out: tensor_info_map.get(out, None) for out in node.output}
        mreq = 0

        for input_tensor in input_shapes:
            # print(f"Input Tensor: {input_tensor}, Shape: {input_shapes[input_tensor]}")
            shape = input_shapes[input_tensor][1]
            elem_size = ONNX_TYPE_SIZE.get(input_shapes[input_tensor][0], 0)
            num_elements = np.prod(shape) if shape else 0
            mreq += num_elements * elem_size

        for input_tensor in output_shapes:
            # print(f"Output Tensor: {input_tensor}, Shape: {output_shapes[input_tensor]}")
            shape = output_shapes[input_tensor][1]
            elem_size = ONNX_TYPE_SIZE.get(output_shapes[input_tensor][0], 0)
            num_elements = np.prod(shape) if shape else 0
            mreq += num_elements * elem_size

        # print(input_shapes)

        if mreq == 0:
            print(f'Warning: Node {node_name} has no memory requirement (mreq = 0).')
            print(f'Node: {node_name}, Inputs: {node.input}, Outputs: {node.output}')

        dag.add_node(
            node.name if node.name else f"{node.op_type}_{i}",
            op_type=node.op_type if node.op_type else "Unknown",
            inputs=node.input if node.input else [],
            outputs=node.output if node.output else [],
            input_shapes=input_shapes if input_shapes else {},
            output_shapes=output_shapes if output_shapes else {},
            in_degree=0,
            out_degree=0,
            mreq=mreq,
            is_branch=False,
            is_merge=False,
        )

        # if not dag.nodes[node_name]['mreq']:
        #     print(dag.nodes[node_name])

        for input_tensor in node.input:
            # print(f'  input = {input_tensor}')
            for prev_node, prev_attr in list(dag.nodes(data=True)):
                # print(f'    prev = {attr['outputs']}')
                if input_tensor in prev_attr['outputs']:
                    # print(f'    Found input tensor {input_tensor} in previous node {prev_node}')
                    dag.add_edge(prev_node, node_name, tensor=input_tensor)
                    # # prev_node_name = prev_node.name or f"{prev_node.op_type}_{id(prev_node)}"
                    # # dag.add_edge(prev_node_name, node_name, tensor=input_tensor)
                    # break
            #         prev_node_name = prev_node.name or f"{prev_node.op_type}_{id(prev_node)}"
            #         dag.add_edge(prev_node_name, node_name, tensor=input_tensor)

        i += 1


    get_node_degrees(dag)
    # print(dag)
    dag_dict = {node: list(dag.successors(node)) for node in dag.nodes()}
    # print("DAG Dictionary:", dag_dict)

    return dag, dag_dict

def format_node_label(node_name, node_data):
    label = f"{node_name}\n"
    label += f"Op: {node_data['op_type']}\n"
    label += f"In: {', '.join(node_data['inputs'])}\n"
    label += f"Out: {', '.join(node_data['outputs'])}\n"
    label += f"Shape: {node_data['output_shapes']}"
    return label

if __name__ == "__main__":
    sys.setrecursionlimit(100000) 

    model = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    dag, dag_dict = onnx_to_dag_with_shapes("/models/" + model + ".onnx", batch_size)

    print('resnetv27_stage1__plus1' in dag.nodes())

    # for node, attrs in dag.nodes(data=True):
    #     print(f"Node: {node}")
    #     print(f"  Op Type: {attrs['op_type']}")
    #     print(f"  Inputs: {attrs['inputs']}")
    #     print(f"  Input Shapes: {attrs['input_shapes']}")
    #     print(f"  Outputs: {attrs['outputs']}")
    #     print(f"  Output Shapes: {attrs['output_shapes']}")
    #     print(f"  In-Degree: {attrs['in_degree']}")
    #     print(f"  Out-Degree: {attrs['out_degree']}")
    #     print(f"  MReq: {attrs['mreq']:,}")
    #     print(f"  Is Branch: {attrs['is_branch']}")
    #     print(f"  Is Merge: {attrs['is_merge']}")
    #     print()

    # for node, attrs in dag.nodes(data=True):
    #     print(f"Node: {node}")
    #     print(f"  Op Type: {attrs}")

    values = {node: int(attrs['mreq']) for node, attrs in dag.nodes(data=True)}

    # print(dag)
    # print("==" * 20)

    # print("Values:", values)

    # print("==" * 20)

    result = compute_graph_weight(dag_dict, values)
    print("===== Memory Requirement =====")
    print("Model:", model + '.onnx')
    print("Batch Size:", batch_size)
    print("Total Memory Requirement: %d KB / %d MB / %d GB" % (result / 1024, result / (1024 * 1024), result / (1024 * 1024 * 1024)))


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