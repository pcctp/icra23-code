import enum
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import skfmm
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.morphology import binary_dilation
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.ndimage import grey_erosion
from sklearn.cluster import DBSCAN
from astar import AStar
from heapq import heappush, heappop, heapify
from pcctp import PCCTP, run_one_instance
from utils import find_best_child

def dilate_bound(image, image_boundary, wtr_thresh=0.9):
    pos_bound_pixels = (image * image_boundary) >= wtr_thresh
    neg_bound_pixels = image_boundary * ((image * image_boundary) < wtr_thresh)
    image_boundary[neg_bound_pixels.astype(bool)] = 0
    dilated_neg = binary_dilation(neg_bound_pixels)
    dilated_pos_bound = (image * dilated_neg) >= wtr_thresh
    image_boundary[dilated_pos_bound] = 1
    return image_boundary

def save_images(prefix, image, image_std, image_boundary, nodes):
    np.save(f"{prefix}_image.npy", image)
    np.save(f"{prefix}_image_std.npy", image_std)
    np.save(f"{prefix}_image_bound.npy", image_boundary)
    nodes = np.array(nodes)
    np.save(f"{prefix}_nodes.npy", nodes)

def load_images(prefix):
    image = np.load(f"{prefix}_image.npy")
    image_std = np.load(f"{prefix}_image_std.npy")
    image_boundary = np.load(f"{prefix}_image_bound.npy")
    nodes = np.load(f"{prefix}_nodes.npy")

    Nodes = [(n[0], n[1]) for n in nodes]
    return image, image_std, image_boundary, Nodes

# in case you want to plot the image
def pyplot(image, image_boundary, image_std, nodes):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 16), dpi=400)
    ax1.imshow(image, cmap='Blues')
    bound = np.zeros(image_boundary.shape + (4,))
    bound[:, :, 3] = image_boundary
    ax1.imshow(bound)
    ax2.imshow(image_std, cmap='Blues')

    for i, node in enumerate(nodes):
        idx, idy = node
        marker = 'o' if i == 0 else 'x'
        ax1.scatter(idx, idy, s=20, c='red', marker=marker)
        ax2.scatter(idx, idy, s=20, c='red', marker=marker)
        ax1.text(idx - 3, idy - 3, i, size = 10, color='red')
        ax2.text(idx - 3, idy - 3, i, size = 10, color='red')
    return fig, ax1, ax2

class LakeSolver(AStar):
    def __init__(self, image, cutoff, image_boundary=None, boundary_pen=0):
        self.image = image
        self.nrows = image.shape[0]
        self.ncols = image.shape[1]
        self.cutoff = cutoff
        self.image_boundary = image_boundary
        self.boundary_pen = boundary_pen
        if image_boundary is not None:
            self.dilated_bound = binary_dilation(image_boundary).astype(float)
            self.dilated_bound += image_boundary
    
    def neighbors(self, node):
        x, y = node
        adjs = []
        for nx, ny in [(x, y-1), (x, y+1), (x-1, y), (x+1, y), (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]:
            if 0 <= nx < self.ncols and 0 <= ny < self.nrows and self.image[ny, nx] >= self.cutoff:
                adjs.append((nx, ny))

        return adjs
    
    def distance_between(self, n1, n2):
        x1, y1 = n1
        x2, y2 = n2

        # high d rewards distance to boundary
        d = 0
        if self.image_boundary is not None:
            d = - self.dilated_bound[y2, x2]
            # d, ind = self.kdtree.query([x2, y2])
            # d = min(d, self.boundary_pen)

        if (abs(x2-x1) + abs(y2-y1)) == 1:
            return 1 - d
        else:
            return np.sqrt(2) - d
    
    def heuristic_cost_estimate(self, n1, n2):
        x1, y1 = n1
        x2, y2 = n2
        return np.linalg.norm([x2-x1, y2-y1])
    
    def astar(self, start, goal, reversePath=False, flimit=np.inf):
        if self.is_goal_reached(start, goal):
            return [start]
        searchNodes = AStar.SearchNodeDict()
        startNode = searchNodes[start] = AStar.SearchNode(
            start, gscore=.0, fscore=self.heuristic_cost_estimate(start, goal))
        openSet = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath)
            if current.fscore > flimit:
                return None
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + \
                    self.distance_between(current.data, neighbor.data)
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + \
                    self.heuristic_cost_estimate(neighbor.data, goal)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None

def totalCost(path):
    xs, ys = zip(*path)
    path = np.array([list(xs), list(ys)]).T
    dist = np.linalg.norm((path[:-1] - path[1:]), axis=1)
    return dist.sum() / 100


def testPinch(st, go, image, lower_bound, upper_bound, maxlen):
    # check the shortcut from ix to iy 
    shortP = LakeSolver(image, lower_bound).astar(st, go, flimit=maxlen)
    if shortP is not None:
        # check the long path from ix to iy
        shortP = list(shortP)
        longP = LakeSolver(image, upper_bound).astar(st, go, flimit=len(shortP)*2)
        if longP is None:
            return shortP
    return None

def findPinch(image, image_std, image_boundary, lower_bound=0.1, upper_bound=0.9, maxlen=20, log=False):
    indy, indx = image_boundary.nonzero()
    indices = set(zip(indy, indx))

    shortcuts = []
    frontiers = []
    for i, (iy, ix) in enumerate(zip(indy, indx)):
        for x in range(-10, 11):
            for y in range(0, 11):
                if y== 0 and (x <= 0):
                    continue
                if (iy + y, ix + x) in indices:
                    st = (ix, iy)
                    go = (ix + x, iy + y)
                    shortP = testPinch(st, go, image, lower_bound, upper_bound, maxlen)
                    if shortP is not None:
                        shortcuts.append(list(shortP))
                        frontiers.append([ix, iy, ix+x, iy+y])
                        if log:
                            print("pinch point", st, go)
    return shortcuts, frontiers

def plotPath(fig, ax1, ax2, path, pinch_points=[]):
    xs, ys = zip(*path)
    xs = list(xs)
    ys = list(ys)
    ax1.scatter(xs, ys, s=1, c='white', marker='.')
    ax2.scatter(xs, ys, s=1, c='black', marker='.')

    for p in pinch_points:
        x, y = p['st']
        ax1.scatter(x, y, s=1, c='red', marker='d')
        ax2.scatter(x, y, s=1, c='red', marker='d')
        x, y = p['ed']
        ax1.scatter(x, y, s=1, c='red', marker='d')
        ax2.scatter(x, y, s=1, c='red', marker='d')

    return fig, ax1, ax2

def plot_policy_map(image, image_boundary, image_std, Nodes, G, sG, ao_tree, save_path):
    root = 0
    q = [root]

    leafs = []
    # BFS traverse the tree to generate tour towards all leaf
    ao_tree.nodes[root]['tour'] = [root]
    while len(q) > 0:
        nid = q.pop()
        node = ao_tree.nodes[nid]
        
        if node['type'] == 'OR':
            best = find_best_child(ao_tree, nid)
            edge = ao_tree.edges[nid, best]
            ao_tree.nodes[best]['tour'] = ao_tree.nodes[nid]['tour'] + edge['path']
            q.append(best)
        else:
            if ao_tree.nodes[nid]['solved'] and len(ao_tree.adj[nid]) == 0:
                leafs.append(nid)
            for j in ao_tree.adj[nid]:
                ao_tree.nodes[j]['tour'] = ao_tree.nodes[nid]['tour']
                q.append(j)
    
    for i, leaf in enumerate(leafs):
        tour = ao_tree.nodes[leaf]['tour']
        fig, ax1, ax2 = pyplot(image, image_boundary, image_std, Nodes)
        for h in range(len(tour) - 1):
            st = tour[h]
            ed = tour[h+1]
            # plot path from st to p
            if (st, ed) in sG.edges():
                plot_shortcuts(fig, ax1, ax2, [sG.edges[st, ed]['path']])
                ax1.text(G.nodes[ed]['loc'][0] - 3, G.nodes[ed]['loc'][1] - 3, h + 1, size=6, color='white')
                ax2.text(G.nodes[ed]['loc'][0] - 3, G.nodes[ed]['loc'][1] - 3, h + 1, size=6)
            else:
                plotPath(fig, ax1, ax2, G.edges[st, ed]['path'])
                ax1.text(G.nodes[ed]['loc'][0] - 3, G.nodes[ed]['loc'][1] - 3, h + 1, size=6, color='white')
                ax2.text(G.nodes[ed]['loc'][0] - 3, G.nodes[ed]['loc'][1] - 3, h + 1, size=6)
        fig.savefig(f"{save_path}_{i}.png")

        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)


def plot_shortcuts(fig, ax1, ax2, sc, label=False):
    for i, path in enumerate(sc):
        xs, ys = zip(*path)
        xs = list(xs)
        ys = list(ys)
        ax1.scatter(xs, ys, s=1, c='darkorange', marker='.')
        ax2.scatter(xs, ys, s=1, c='darkorange', marker='.')

        mp = xs[len(xs)//2], ys[len(ys)//2]
        if label:
            ax1.text(mp[0] - 3, mp[1] - 3, i, size=8, color='darkorange')
            ax2.text(mp[0] - 3, mp[1] - 3, i, size=8, color='darkorange')
    return fig, ax1, ax2

def dist_metric(f1, f2):
    d1 = np.linalg.norm(f1[0:2] - f2[0:2]) + np.linalg.norm(f1[2:4] - f2[2:4])
    d2 = np.linalg.norm(f1[0:2] - f2[2:4]) + np.linalg.norm(f1[2:4] - f2[0:2])
    return min(d1, d2)

def pick_shortcuts(shortcuts, frontiers, eps=4, min_samples=2, log=False):
    frontiers = np.array(frontiers)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=dist_metric).fit(frontiers)
    labels = db.labels_

    n_edges_ = len(set(labels)) - (1 if -1 in labels else 0)
    if log:
        print(n_edges_, "stochastic edges", labels)
    shortcuts_np = np.array([list(s) for s in shortcuts], dtype=object)
    shortcuts_pick = []
    frontiers_pick = []

    for i in range(n_edges_):
        class_mask = labels == i
        f_i = frontiers[class_mask]
        s_i = shortcuts_np[class_mask]
        s_length = np.array([totalCost(s) for s in s_i])
        argmin = s_length.argmin()
        path = [tuple(x) for x in s_i[argmin]]
        front = f_i[argmin]

        shortcuts_pick.append(path)
        frontiers_pick.append((front[0], front[1]))
        frontiers_pick.append((front[2], front[3]))
    
    return shortcuts_pick, frontiers_pick

def build_sG(Nodes, shortcuts_pick, frontiers_pick, image, image_std, log=False):
    sG = nx.Graph()
    n_nodes = len(Nodes)

    frontier_ids = {}
    for i, path in enumerate(shortcuts_pick):
        cost = totalCost(path)
        inda = i * 2 + n_nodes
        indb = i * 2 + 1 + n_nodes
        xs, ys = zip(*path)
        prob = image[ys, xs].min()
        std = image_std[ys, xs].max()
        if log:
            print(inda, indb, cost, prob, std)
        
        f1, f2 = frontiers_pick[i*2], frontiers_pick[i*2+1]
        if f1 in frontier_ids:
            frontier_ids[f1].append(inda)
        else:
            frontier_ids[f1] = [inda]
        if f2 in frontier_ids:
            frontier_ids[f2].append(indb)
        else:
            frontier_ids[f2] = [indb]
        sG.add_edge(inda, indb, weight=cost, p=1-prob, path=path)
    
    removed = False
    for loc, fronts in frontier_ids.items():
        nSE = len(fronts)
        if nSE == 2:
            removed = True
            nodei, nodej = fronts[0], fronts[1]
            ni = list(sG.adj[nodei])[0]
            nj = list(sG.adj[nodej])[0]

            ci = sG.edges[nodei, ni]['weight']
            pi = sG.edges[nodei, ni]['p']
            pathi = sG.edges[nodei, ni]['path']
            cj = sG.edges[nodej, nj]['weight']
            pj = sG.edges[nodej, nj]['p']
            pathj = sG.edges[nodej, nj]['path']
                    
            cij = ci + cj
            pij = 1 - (1- pi) * (1 - pj)
            if tuple(pathi[0]) == tuple(pathj[0]):
                pathij = np.concatenate((pathi[::-1], pathj[1:]), 0)
            elif tuple(pathi[0]) == tuple(pathj[-1]):
                pathij = np.concatenate((pathj, pathi[1:]), 0)
            elif tuple(pathi[-1]) == tuple(pathj[0]):
                pathij = np.concatenate((pathi, pathj[1:]), 0)
            elif tuple(pathi[-1]) == tuple(pathj[-1]):
                pathij = np.concatenate((pathi[:-1], pathj[::-1]), 0)

            sG.add_edge(ni, nj, weight=cij, p=pij, path=pathij)
            sG.remove_node(nodei)
            sG.remove_node(nodej)

    return sG 

def build_G(Nodes, frontiers_pick, image, image_boundary, wtr_thresh=0.9, log=False):
    G = nx.Graph()

    AllNodes = Nodes[:]
    AllNodes += frontiers_pick
    for i, node in enumerate(AllNodes):
        G.add_node(i, loc=node)

    for i in range(len(AllNodes)-1):
        for j in range(i + 1, len(AllNodes)):
            start = AllNodes[i]
            goal = AllNodes[j]
            foundPath = LakeSolver(image, wtr_thresh, 
                                  image_boundary, 1).astar(start, goal)
            if foundPath:
                foundPath = list(foundPath)
                cost = totalCost(foundPath)
                if log:
                    print(f'{i}, {j}, cost={cost:.1f}km')
                G.add_edge(i, j, weight=cost, path=foundPath)

    return G

def prune_pinch(G, sG, shortcuts_pick=None, frontiers_pick=None, log=False):
    N = len(G.nodes)
    nodes = list(G.nodes)
    dist_matrix = np.ones((N, N)) * np.inf
    k2i = {n:i for i, n in enumerate(G.nodes)}

    for i, ki in enumerate(nodes):
        for kj in G.adj[ki].keys():
            j = k2i[kj]
            dist_matrix[i][j] = G.adj[ki][kj]['weight']
            dist_matrix[j][i] = dist_matrix[i][j]

    edit_list = False
    if shortcuts_pick is not None and frontiers_pick is not None:
        sp = shortcuts_pick.copy()
        ft = frontiers_pick.copy()
        edit_list = True

    k = 0
    while k < len(sG.edges()):
        e = list(sG.edges())[k]
        w = sG.edges[e]['weight']
        useful = False
        for i in range(N):
            if nodes[i] not in G.nodes():
                continue
            for j in range(i+1, N):
                if nodes[j] not in G.nodes():
                    continue
                e0, e1 = k2i[e[0]], k2i[e[1]]
                d1 = dist_matrix[i][e0] + dist_matrix[e1][j] + w
                d2 = dist_matrix[i][e1] + dist_matrix[e0][j] + w
                d = min(d1, d2)
                if d != float('inf') and d < dist_matrix[i][j]:
                    useful = True
                    #print(k, i, j, "d'=", d, "d=", dist_matrix[i][j])
        
        if log:
            print(k ,e[0], e[1], useful)

        if useful:
            k += 1
        else:
            sG.remove_node(e[0])
            sG.remove_node(e[1])
            G.remove_node(e[0])
            G.remove_node(e[1])

            if edit_list:
                del sp[k]
                del ft[k*2 + 1]
                del ft[k*2]
            k = 0

    if edit_list:
        return sp, ft

def get_barrier_fnc(image, wtr_thresh):
    Ny, Nx = image.shape[0], image.shape[1]
    phi = np.where(image < wtr_thresh, 1, -1)
    sol = skfmm.distance(phi, dx=0.5)
    sol[sol < -1] = -1
    sol[sol > 0] = 0
    sol += 1

    x = np.arange(Nx)
    y = np.arange(Ny)
    f = RectBivariateSpline(y, x, sol)
    return sol, f


def downsample_path(image, barrier_f, path, min_water_prob, steps=100):
    path = np.array(path)
    
    i = 0
    while i < steps:

        a, b = (np.random.rand(2) * len(path)).astype(int)
        if np.abs(a-b) <= 1:
            i += 1
            continue

        if a > b:
            a, b = b, a
        xa, xb = path[a], path[b]

        num = np.abs(b-a) * 100
        interpx = interp1d([0, 1], [xa[0], xb[0]])
        interpy = interp1d([0, 1], [xa[1], xb[1]])
        t = np.linspace(0, 1, num)
        
        nx = np.round(interpx(t))
        ny = np.round(interpy(t))
        obstacles = barrier_f(ny, nx, grid=False)
        if obstacles.max() <= 0.5:
            path = np.concatenate([path[:a+1], path[b:]])
        
        i += 1
    return path


def rsample_path(path, num):
    path = np.array(path)
    N = path.shape[0]
    t = np.linspace(0, 1, num=len(path))

    csx = interp1d(t, path[:, 0], kind='linear') 
    csy = interp1d(t, path[:, 1], kind='linear') 
    
    tt = np.linspace(0, 1, num)
    rsx = csx(tt)
    rsy = csy(tt)
    rs_path = np.vstack([rsx, rsy]).T
    return rs_path


def downsample_graph(image, G, sG):
    sol, barrier_f = get_barrier_fnc(image, 0.9)

    for e in G.edges:
        path = G.edges[e]['path']
        if len(path) > 1:
            ds_path = downsample_path(image, barrier_f, path, 0.9, steps=50 * (1 + np.log(len(path))))
            ds_path = rsample_path(ds_path, len(path) * 4)
            G.edges[e]['path'] = ds_path

    sol, barrier_f = get_barrier_fnc(image, 0.1)
    for e in sG.edges:
        path = sG.edges[e]['path']
        if len(path) > 1:
            ds_path = downsample_path(image, barrier_f, path, 0.1, steps=50 * (1 + np.log(len(path))))
            ds_path = rsample_path(ds_path, len(path) * 4)
            sG.edges[e]['path'] = ds_path

            
def smooth_path(image, barrier_f, path, min_water_prob, alpha, beta, gamma):
    path = np.array(path)
    start = path[0:1]
    end = path[-1:]
    ind = path.astype(int)
    lcol = (1 - image[ind[:, 1], ind[:, 0]]).max()
    print(f"init collision check {lcol}")

    def fn(z):
        z = z.reshape(-1, 2)

        z = np.concatenate([start, z, end])
        ld = ((z - path)**2).sum()
        dz = (z[1:] - z[:-1])
        norm = np.linalg.norm(dz, axis=1, keepdims=True)
        dz = dz / norm
        lp = (dz[:-1] * dz[1:]).sum()

        lcol = barrier_f(z[:, 1], z[:, 0], grid=False).sum()
        return ld + alpha * norm.sum() - lp * beta + lcol * gamma
    
    z0 = path[1:-1].flatten()
    res = minimize(fn, z0, method='L-BFGS-B')
    sm_path = np.concatenate([start, res.x.reshape(-1, 2), end])
    ind = sm_path.astype(int)
    lcol = (1 - image[ind[:, 1], ind[:, 0]]).max()
    # assert lcol <= (1 - min_water_prob)
    print(f"initial l {fn(z0):.0f} optimized l {res.fun:.0f} collision check {lcol}")

    return sm_path


def smooth_graph(image, G, sG):
    # Compute barrier fnc at p=0.9
    sol, barrier_f = get_barrier_fnc(image, 0.9)
    for e in G.edges:
        path = G.edges[e]['path']
        if len(path) > 1:
            sm_path = smooth_path(image, barrier_f, path, 0.9, 100, 0, 100)
            rs_path = rsample_path(sm_path, len(path) * 4)
            G.edges[e]['path'] = rs_path
  
    # Recompute barrier fnc at p=0.1 for stochastic edges
    sol, barrier_f = get_barrier_fnc(image, 0.1)
    for e in sG.edges:
        path = sG.edges[e]['path']
        if len(path) > 1:
            sm_path = smooth_path(image, barrier_f, path, 0.1, 100, 0, 100)
            rs_path = rsample_path(sm_path, len(path) * 4)
            sG.edges[e]['path'] = rs_path

def modify_graph_wind(G, sG, Nodes, image, image_boundary, bounddist_thresh=200, length_thresh=0, max_sedge_count=10, plot=False):
    # prune some leftover nodes
    for n in list(G.nodes()):
        if n >= len(Nodes) and n not in sG.nodes():
            G.remove_node(n)
    
    phi = np.where(image > 0.1, 1, -10)
    sol = skfmm.distance(phi, dx=10)
    sol[sol > 300] = 300
    new_sedge = []

    if plot:
        fig, ax1, ax2 = pyplot(image, image_boundary, sol, Nodes)
        for j, e in enumerate(sG.edges()):
            path = sG.edges[e]['path']
            xs, ys = zip(*path)
            xs = list(xs)
            ys = list(ys)
            ax1.scatter(xs, ys, s=1, c='darkorange', marker='.')

            mp = xs[len(xs)//2], ys[len(ys)//2]
            ax1.text(mp[0] - 3, mp[1] - 3, j, size=8, color='darkorange')

    for e in list(G.edges()):
        path = G.edges[e]['path']
        if G.edges[e]['weight'] > 0:
            px = path[:, 0].astype(int)
            py = path[:, 1].astype(int)
            bound_dist = sol[py, px]
            if totalCost(path) >= length_thresh and bound_dist.max() >= bounddist_thresh:
                new_sedge.append(e)
    
    new_sedge_count = max(0, max_sedge_count - len(sG.edges()))
    print(len(sG.edges()), " | ", len(new_sedge), " | ", end="")
    if len(new_sedge) > new_sedge_count:
        new_sedge = random.sample(new_sedge, new_sedge_count)

    for e in new_sedge:
        path = G.edges[e]['path']
        ne = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
        sG.add_edge(ne[0], ne[1], weight=G.edges[e]['weight'], path=path, p=random.random() * 0.3)
        G.remove_edge(e[0], e[1])
        if plot:
            xs, ys = zip(*path)
            xs, ys = list(xs), list(ys)
            ax2.scatter(xs, ys, s=1, c='white', marker='.')

    if plot:
        plt.show()

def test1():
    import time

    ts = time.time()
    filepath = "lakes/lower_9mlake"
    image, image_std, image_boundary, Nodes = load_images(filepath)
    shortcuts, frontiers = findPinch(image, image_std, image_boundary)
    shortcuts_pick, frontiers_pick = pick_shortcuts(shortcuts, frontiers)
    tpinch = time.time() - ts

    ts = time.time()
    # build sG
    sG = build_sG(Nodes, shortcuts_pick, image, image_std)
    # build G
    G = build_G(Nodes, frontiers_pick, image, image_boundary)
    tgraph = time.time() - ts

    ts = time.time()
    # prune pinch
    shortcuts_useful, frontiers_useful = prune_pinch(G, sG, shortcuts_pick, frontiers_pick)
    tprune = time.time() - ts

    fig, ax1, ax2 = pyplot(image, image_boundary, image_std, Nodes)
    plot_shortcuts(fig, ax1, ax2, shortcuts_useful)
    fig.savefig(f"{filepath}.png")

    # save graph
    nx.write_gpickle(G, f"{filepath}_G.gpickle")
    nx.write_gpickle(sG, f"{filepath}_sG.gpickle")

    goals = list(range(1, len(Nodes)))
    _, ao_tree, rt = run_one_instance(G, sG, goals, save_steps=False, plot=False)
    
    print("find pinch points time:", tpinch)
    print("build graph time:", tgraph)
    print("prune pinch time:", tprune)
    print("ao* time:", rt)

    plot_policy_map(image, image_boundary, image_std, Nodes, G, sG, ao_tree, f"{filepath}_policy")

def test2():
    # save graph
    filepath = "lakes/upper_9mlake"
    image, image_std, image_boundary, Nodes = load_images(filepath)

    G = nx.read_gpickle(f"{filepath}_G.gpickle")
    sG = nx.read_gpickle(f"{filepath}_sG.gpickle")

    goals = list(range(1, len(Nodes)))
    _, ao_tree, rt = run_one_instance(G, sG, goals, save_steps=False, plot=True)
    
    plot_policy_map(image, image_boundary, image_std, Nodes, G, sG, ao_tree, f"{filepath}_policy")



if __name__ == "__main__":
    test1()
