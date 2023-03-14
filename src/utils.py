import random
import networkx as nx
import numpy as np
import graphviz
import json
from itertools import chain, combinations
from pathlib import Path

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def plot_problem_graph(G, sG, goals, savedir):
    # Visualize problem graph
    dot = graphviz.Graph(format='png', engine='dot')
    for nid in G.nodes():
        if nid in goals:
            dot.node(str(nid), shape='doublecircle')
        else:
            dot.node(str(nid), shape='circle')

    for eid in G.edges():
        dot.edge(str(eid[0]), str(eid[1]), label=f"{round(G.edges[eid]['weight'], 3)}")
    dot.attr('edge', style='dashed')
    for eid in sG.edges():
        dot.edge(str(eid[0]), str(eid[1]), label=f"{round(sG.edges[eid]['weight'], 3)} ({round(sG.edges[eid]['p'], 3)})")
    dot.attr(overlap='false')

    dot.render(Path(savedir) / 'graph.gv', view=True)


def find_best_child(T, nid):
    """
    Returns the best child of OR node nid in T.
    """
    
    best = list(T.adj[nid])[0]
    best_cost = T.nodes[best]['f'] + T.edges[nid, best]['weight']
    for child in T.adj[nid]:
        cost = T.nodes[child]['f'] + T.edges[nid, child]['weight']
        if cost < best_cost:
            best = child
            best_cost = cost
    
    return best

def save_policy(ao_tree, save_path):
    root = 0
    policy_tree = nx.DiGraph()
    policy_tree.add_node(0, **ao_tree.nodes[root])

    q = [root]

    # BFS traverse the tree to generate tour towards all leaf
    ao_tree.nodes[root]['tour'] = [root]
    while len(q) > 0:
        nid = q.pop()
        node = ao_tree.nodes[nid]
        
        if node['type'] == 'OR':
            if len(ao_tree.adj[nid]) > 0:
                best = find_best_child(ao_tree, nid)
                edge = ao_tree.edges[nid, best]
                ao_tree.nodes[best]['tour'] = ao_tree.nodes[nid]['tour'] + edge['path']
                q.append(best)
                policy_tree.add_node(best, **ao_tree.nodes[best])
                policy_tree.add_edge(nid, best, **ao_tree.edges[nid, best])
        else:
            for j in ao_tree.adj[nid]:
                ao_tree.nodes[j]['tour'] = ao_tree.nodes[nid]['tour']
                q.append(j)
                policy_tree.add_node(j, **ao_tree.nodes[j])
                policy_tree.add_edge(nid, j, **ao_tree.edges[nid, j])

    nx.write_gpickle(policy_tree, save_path)
    return policy_tree

def plot_policy_tree(ao_tree, savepath, view=True):
    dot = graphviz.Graph(format='png')

    # Run bfs to plot the nodes in the policy tree
    root = 0
    q = [root]

    while len(q) > 0:
        nid = q.pop()
        
        node = ao_tree.nodes[nid]
        if node['type'] == 'OR':
            shape = 'box'
            xlp="0,0"
        else:
            shape = 'ellipse'
            xlp = "-100 ,-100"
        shape = 'box' if node['type'] == 'OR' else 'ellipse'
        S = node['traversed'] if len(node['traversed']) > 0 else '{}'
        dot.node(str(nid), f'{S}, {node["at"]}, {node["info"]}', shape=shape, xlabel=f"{round(node['f'], 3)}", xlp=xlp)
        
        if node['type'] == 'OR':
            # find best child
            if len(ao_tree.adj[nid]) > 0:
                best = find_best_child(ao_tree, nid)
                
                edge = ao_tree.edges[nid, best]
                print(edge['path'])
                dot.edge(str(nid), str(best), label=f"{edge['weight']:.3f}")

                q.append(best)
        else:
            for j in ao_tree.adj[nid]:
                edge = ao_tree.edges[nid, j]
                dot.edge(str(nid), str(j), label=f'p={edge["p"]:.3f}')

                q.append(j)

    dot.attr(overlap='false')
    dot.render(savepath, view=view)

def plot_aotree(ao_tree, savepath, view=True):
    dot = graphviz.Graph(format='png')

    for nid in ao_tree.nodes():
        node = ao_tree.nodes[nid]
        if node['type'] == 'OR':
            shape = 'box'
            xlp = "0,0"
        else:
            shape = 'ellipse'
            xlp = "-100 ,-100"
        shape = 'box' if node['type'] == 'OR' else 'ellipse'
        S = node['traversed'] if len(node['traversed']) > 0 else '{}'
        dot.node(str(nid), f'{S}, {node["at"]}, {node["info"]}', shape=shape, xlabel=f"{round(node['f'], 2)}", xlp=xlp, fontsize='14')

    for eid in ao_tree.edges():
        edge = ao_tree.edges[eid]
        if edge['p'] < 1:
            dot.edge(str(eid[0]), str(eid[1]), label=f'{edge["weight"]:.2f} ({edge["p"]:.1f})', fontsize='14')
        else:
            dot.edge(str(eid[0]), str(eid[1]), label=f"{edge['weight']:.2f}", fontsize='14')

    
    dot.attr(overlap='false')
    dot.render(savepath, view=view)

    # pyvis visualization in browser
    # nt = Network(width="80%", height="100%", layout = "hierarchy")
    # for nid in ao_tree.nodes():
    #     node = ao_tree.nodes[nid]
    #     shape = 'box' if node['type'] == 'OR' else 'ellipse'
    #     nt.add_node(nid, label=f'{node["traversed"]}, {node["at"]}, {node["info"]}',
    #         shape=shape)

    # for eid in ao_tree.edges():
    #     edge = ao_tree.edges[eid]
    #     nt.add_edge(eid[0], eid[1], title=edge["weight"], value=1)

    # nt.show_buttons(filter_=['physics'])
    # nt.show("nx.html")

def make_aostar_vod(folder='results/cache'):
    import cv2
    import os

    video_name = os.path.join(folder, 'aostar_expansion.avi')

    images = sorted([img for img in os.listdir(folder) if img.endswith(".png")])
    height, width, layers = 800, 2000, 3

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, 4, (width,height))

    for image in images:
        l_img = np.ones((height, width, layers), dtype=np.uint8) * 255
        img = cv2.imread(os.path.join(folder, image))
        if img.shape[0] > height or img.shape[1] > width:
            img = cv2.resize(img, (width, height))
        y_offset = 0
        x_offset = 0
        l_img[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
        video.write(l_img)

    cv2.destroyAllWindows()
    video.release()

def plot_runtime(n, k, f1, f2):
    import matplotlib.pyplot as plt
    mu = np.loadtxt(f1)
    std = np.loadtxt(f2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    N, K = np.meshgrid(n, k)
    mu = mu.T
    std = std.T

    ax.plot_surface(N, K, mu)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Number of Stochastic Edges')
    ax.set_zlabel('Average Runtime (s)')

    plt.savefig('results/runtime.png')

def load_images(directory, prefix):
    directory = directory + "/" + prefix + "/"
    prefix = directory + prefix

    try:
        image = np.load(f"{prefix}_image.npy")
    except Exception:
        return None  # Folder not found

    image_std = np.load(f"{prefix}_image_std.npy")
    image_boundary = np.load(f"{prefix}_image_bound.npy")
    coords = np.load(f"{prefix}_image_coords.npy")
    with open(f"{prefix}_image_proj.json") as json_file:
        proj = json.load(json_file)
    return image, image_std, image_boundary, coords