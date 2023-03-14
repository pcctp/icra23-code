import networkx as nx
import numpy as np

def G1():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 4, 5])
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=1)
    G.add_edge(1, 4, weight=3)
    G.add_edge(2, 5, weight=3)

    sG = nx.Graph()
    sG.add_nodes_from([0, 1, 2, 4, 5])
    sG.add_edge(4, 5, weight=2, p=0.1)
    sG.add_edge(1, 4, weight=1, p=0.1)
    sG.add_edge(2, 5, weight=1, p=0.1)
    return G, sG, [1, 2, 4, 5]

def G3():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edge(0, 1, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=3)

    sG = nx.Graph()
    sG.add_nodes_from([0, 1, 2, 3])

    sG.add_edge(1, 3, weight=3, p=0.1)
    sG.add_edge(2, 3, weight=1, p=0.1)
    return G, sG, [1, 3]


def G2():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1, weight=10)
    sG = nx.Graph()
    sG.add_nodes_from([0, 1])
    sG.add_edge(0, 1, weight=2, p=0.2)
    return G, sG, [1]

def G4():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1, weight=1)
    sG = nx.Graph()
    sG.add_nodes_from([0, 1])
    sG.add_edge(0, 1, weight=2, p=0.2)
    return G, sG, [1]

def G5():
    G = nx.Graph()

    G.add_nodes_from([0, 1, 2, 3])
    G.add_edge(0, 1, weight=2)
    G.add_edge(0, 2, weight=2)
    G.add_edge(1, 3, weight=10)
    G.add_edge(2, 3, weight=10)

    sG = nx.Graph()
    sG.add_nodes_from([0, 1, 2, 3])
    sG.add_edge(1, 3, weight=4, p=0.1)
    sG.add_edge(2, 3, weight=4, p=0.5)
    return G, sG, [3]

def G6():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4])
    G.add_edge(0, 1, weight=11)
    G.add_edge(1, 2, weight=7)
    G.add_edge(2, 3, weight=3)
    G.add_edge(0, 4, weight=4)
    G.add_edge(1, 3, weight=4)
    G.add_edge(1, 4, weight=8)

    sG = nx.Graph()
    sG.add_nodes_from([0, 1, 2, 3, 4])
    sG.add_edge(3, 4, weight=6, p=0.1)
    return G, sG, [1, 2, 4]

def G7():
    # graph with potentially unreachable goals
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edge(0, 1, weight=2)
    G.add_edge(0, 2, weight=3)
    G.add_edge(1, 2, weight=4)

    sG = nx.Graph()
    sG.add_nodes_from([0, 1, 2, 3])
    sG.add_edge(1, 3, weight=3, p=0.1)
    sG.add_edge(2, 3, weight=5, p=0.4)
    return G, sG, [2, 3]

def G8():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5])
    G.add_edge(0, 1, weight=2)
    G.add_edge(0, 5, weight=4)
    G.add_edge(1, 5, weight=3)
    G.add_edge(3, 4, weight=2)

    sG = nx.Graph()
    sG.add_nodes_from([0, 1, 2, 3, 4, 5])
    sG.add_edge(1, 2, weight=1, p=0.3)
    sG.add_edge(2, 3, weight=1, p=0.2)
    return G, sG, [4, 5]

def G_example():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5])
    G.add_edge(0, 2, weight=0.05)
    G.add_edge(0, 4, weight=0.22)
    G.add_edge(2, 4, weight=0.27)
    G.add_edge(3, 1, weight=0.18)
    G.add_edge(5, 1, weight=0.41)
    G.add_edge(3, 5, weight=0.51)

    sG = nx.Graph()
    sG.add_nodes_from([0, 1, 2, 3, 4, 5])
    sG.add_edge(2, 3, weight=0.21, p=0.8)
    sG.add_edge(4, 5, weight=0.16, p=0.7)
    return G, sG, [1]

def generate_test_graphs(N, K, n_goals):
    # sample N points in a [0, 0] x [-10, 10] square

    graph_sparsity = 0.6
    points = np.random.rand(N, 2) * 10
    root = np.zeros(2, dtype=float)
    points = np.vstack((root, points))
    
    dist_matrix = np.sqrt(((points[None, :] - points[:, None])**2).sum(-1))
    nodes = list(range(N + 1))
    
    
    # sample random edges 
    adj = np.triu(np.random.rand(N + 1, N + 1), 1)
    adj = adj + adj.transpose()
    adj_w = np.copy(dist_matrix)
    adj_w[adj < graph_sparsity] = 0
    
    # convert to graph
    G = nx.convert_matrix.from_numpy_matrix(adj_w)

    # sample K stochastic edges

    sG = nx.Graph()
    sG.add_nodes_from(nodes)
    for i in range(K):
        u = np.random.choice(nodes)
        v = np.random.choice(nodes)
        while u == v or (v in G.adj[u]) or (v in sG.adj[u]):
            u = np.random.choice(nodes)
            v = np.random.choice(nodes)
        sG.add_edge(u, v, weight=dist_matrix[u, v], p=np.random.rand())

    # sample G goal nodes
    goals = sorted(np.random.choice(nodes[1:], size=n_goals, replace=False))

    return G, sG, goals

def run_example():
    from pcctp import run_one_instance
    from utils import save_policy
    G, sG, goals = G_example()
    expected_cost, ao_tree, rt = run_one_instance(G, sG, goals, save_steps=True, plot=True, log=True)
    save_policy(ao_tree, f"results/example.gv")

if __name__ == "__main__":
    run_example()