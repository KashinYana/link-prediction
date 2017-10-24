import random as python_random

def safe_sample_edges(nodes, edges, sample_size):
    edges = set(edges)
    nodes = list(nodes)

    edge_label = {}
    node2edges = {node : [] for node in nodes}
    for edge in edges:
        node2edges[edge[0]].append(edge)
        node2edges[edge[1]].append(edge)
        edge_label[edge] = 'keep'

    def walk(source, visited):
        queue = set()
        if source not in visited:
            queue.add(source)
        while len(queue) > 0:
            current = queue.pop()
            visited.add(current)
            for edge in node2edges[current]:
                if edge_label[edge] == 'keep':
                    if edge[0] == current:
                        added = edge[1]
                    else:
                        added = edge[0]
                    if added not in visited:
                        queue.add(added)

    # choice giant component
    visited = set()
    walk(python_random.choice(nodes), visited)
    if len(visited) != len(nodes):
        print 'Graph is disconnected, will try to choice giant component'
        while len(visited) < 0.8 * len(nodes):
            visited = set()
            walk(python_random.choice(nodes), visited)
        print 'visited %d out of %d nodes' % (len(visited), len(nodes))
        edges = set([edge for edge in edges if edge[0] in visited and edge[1] in visited])
        nodes = list(visited)
        node2edges = {node : [] for node in nodes}
        for edge in edges:
            node2edges[edge[0]].append(edge)
            node2edges[edge[1]].append(edge)
            edge_label[edge] = 'keep'

    sampled_edges = set()
    iteration = 0
    while len(sampled_edges) < sample_size:
        candidates = python_random.sample(edges - sampled_edges, sample_size - len(sampled_edges))
        for edge in candidates:
            edge_label[edge] = 'candidate'
        visited = set()
        source = python_random.choice(nodes)
        while len(visited) < len(nodes):
            assert(source not in visited)
            walk(source, visited)
            for edge in candidates:
                if edge_label[edge] == 'candidate':
                    if edge[0] not in visited and edge[1] in visited:
                        edge_label[edge] = 'keep'
                        source = edge[0]
                        break
                    elif edge[1] not in visited and edge[0] in visited:
                        edge_label[edge] = 'keep'
                        source = edge[1]
                        break
                    elif edge[0] in visited and edge[1] in visited:
                        edge_label[edge] = 'remove'
                    else:
                        pass
        for edge in edges:
            if edge_label[edge] == 'remove':
                sampled_edges.add(edge)
            assert(edge_label[edge] != 'candidate')
        print 'Iteration %d, sampled edges %d' % (iteration, len(sampled_edges))
        iteration += 1

    return nodes, edges, sampled_edges