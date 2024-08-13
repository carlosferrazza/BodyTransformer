from .adjacency_matrix_cmu_humanoid import INPUT_WEIGHTED_GRAPH

V = len(INPUT_WEIGHTED_GRAPH)
# Code used from https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/

def floydWarshall(graph):
    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))

    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

CMU_HUMANOID_SP = floydWarshall(INPUT_WEIGHTED_GRAPH)
