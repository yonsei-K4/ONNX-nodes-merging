from collections import defaultdict, deque

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

    # make a copy of graph so we can modify it safely
    graph = {k: list(v) for k, v in graph.items()}

    for node in graph:
        if is_branch_node(graph, node):
            stack.append((0, node))
        if is_merge_node(graph, node):
            stack.append((1, node))

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
                print(merged_value)

                # update values: branch node gets the computed value
                values[branch] = merged_value

                # remove all outgoing edges from branch and redirect to merge
                graph[branch] = [merge]

                # remove all nodes in subgraph except branch and merge
                for node in subgraph:
                    if node != branch and node != merge:
                        graph.pop(node, None)
                        values.pop(node, None)

                # clean up stack
                stack = stack[:i] + stack[i+2:]
                break  # restart loop from beginning
            i += 1

    # compute total weight from modified graph
    dfs = merge_ONNX_graph(graph, values)
    return dfs(1, 1)



#example
graph = {
    1: [2,10],
    2: [3,5],
    3: [4],
    4: [9],
    5: [6,7],
    6: [8],
    7: [8],
    8: [9],
    9: [14],
    10: [11,12],
    11:[13],
    12:[13],
    13:[14],
    14:[15],
    15:[]
}
values = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15
}

result = compute_graph_weight(graph, values)
print("Total weight:", result)
