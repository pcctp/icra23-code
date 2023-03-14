import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import time
import shutil

import test_graph
from utils import powerset, make_aostar_vod, plot_runtime, save_policy
from utils import plot_aotree, plot_policy_tree, plot_problem_graph
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from python_tsp.exact import solve_tsp_dynamic_programming
from pathlib import Path


class PCCTP():
    def __init__(self, G, sG, start, goals, use_h = True, metric_G = False,
             search_reachable_only = True):
        self.G = G
        self.sG = sG
        self.sG_edges = list(sG.edges())
        self.start = start
        self.goals = goals

        self.K = len(self.sG.edges())
        self.INF = 1e9
        self.use_h = use_h
        self.metric_G = metric_G
        
        self.clinks = self.find_critical_links()

        info = ''.join(['T'] * len(self.sG_edges))
        self.reachable_goals = self.clinks[info]
        self.search_reachable_only = search_reachable_only

    def find_critical_links(self):
        # Step 1: build a disjoint set
        def bfs(n, G, id):
            q = [n]
            visited[n] = True
            while len(q) > 0:
                n = q.pop(0)
                for nbr in G.adj[n]:
                    id[nbr] = id[n]
                    if not visited[nbr]:
                        visited[nbr] = True
                        q.append(nbr)
        
        # Id is the component every node is in
        id = {n: i for i, n in enumerate(self.G.nodes)}
        visited = {n: False for n in self.G.nodes}
        for i, n in enumerate(self.G.nodes()):
            if not visited[n]:
                bfs(n, self.G, id)
        
        # Step 2: gradually add stochastic edges and see if edges are connected
        clinks = {}
        pset = powerset(self.sG_edges)
    
        for t_edges in pset:
            # update the disjoint set for the new edges
            id_t = id.copy()
            G = self.G.copy()
            info = ['U'] * len(self.sG_edges)
            for e in t_edges:
                # get edge index
                eid = self.index_sedge(*e)
                info[eid] = 'T'
                # update disjoint set
                G.add_edge(e[0], e[1])
                visited = {n: False for n in G.nodes}
                bfs(e[0], G, id_t)
            
            info = ''.join(info)
            clinks[info] = set()
            # check if the goal nodes are connected
            ids = id_t[self.start]
            for g in self.goals:
                if ids == id_t[g]:
                    clinks[info].add(g)
        
        return clinks

    def index_sedge(self, e0, e1):
        e = (e0, e1)
        if e in self.sG_edges:
            return self.sG_edges.index(e)
        e = (e1, e0)
        if e in self.sG_edges:
            return self.sG_edges.index(e)
        return -1

    def compute_dist_matrix(self, info, edge_states, predecessor=False):
        # build the optimistic graph
        G = self.G.copy()

        for i, state in enumerate(info):
            e = self.sG_edges[i]
            edge = self.sG.edges[e]
            if state in edge_states:
                if (e not in G.edges) or edge['weight'] < G.edges[e[0], e[1]]['weight']: 
                    G.add_edge(e[0], e[1], weight=edge['weight'])
        
        key_to_id = {n:i for i, n in enumerate(G.nodes)}
        #  Compute TSP weight matrices
        dist_matrix = nx.floyd_warshall_numpy(G)
        if predecessor:
            predecessors, _ = nx.floyd_warshall_predecessor_and_distance(G)
            return dist_matrix, key_to_id, predecessors
        else:
            return dist_matrix, key_to_id

    def heuristic(self, node):
        """
        node: a AO* node dictionary
        return the admissible cost to reach the goal from the current node
        """
        if not self.use_h:
            return 0
        
        if self.search_reachable_only:
            # find potentially unreachable nodes
            goals = node['goals']
            tinfo = node['info'].replace('A', 'T')
            uinfo = node['info'].replace('A', 'U')
            def_reachable = self.clinks[uinfo]
            pot_unreachable = goals - def_reachable

            critical_edge = []
            modified_info = list(node['info'])
            for i, edge in enumerate(node['info']):
                if edge=='A':
                    critical=False
                    # see if chaning unblocking this edge makes any node reachable
                    s = list(uinfo)
                    s[i]='T'
                    new_reachable = self.clinks["".join(s)] - def_reachable
                    if len(new_reachable) > 0:
                        critical=True
                    
                    # we can also check if blocking this edge makes any node unreachable
                    t = list(tinfo)
                    t[i]='U'
                    new_unreachable = goals - self.clinks["".join(t)]
                    if len(new_unreachable) > 0:
                        critical=True
                    if critical:
                        critical_edge.append(self.sG_edges[i])
                        modified_info[i] = 'U'
                    else:
                        modified_info[i] = 'T'
            modified_info = ''.join(modified_info)
            dist_matrix, key_to_id = self.compute_dist_matrix(modified_info, ['T'])
            # goal is to reach the critical edges too
            critical_nodes = np.array(critical_edge).flatten()
            critical_nodes = np.unique(critical_nodes)
            # we want to visit one of the critical node in tsp,
            # this is a generalized tsp problem, we cna transform to tsp
            if len(critical_nodes) >= 2:
                n0 = critical_nodes[0]
                nprev = n0
                dist0 = dist_matrix[key_to_id[n0]].copy()
                for i in range(1, len(critical_nodes)):
                    ni = critical_nodes[i]
                    dist_matrix[key_to_id[nprev]] = dist_matrix[key_to_id[ni]]
                    dist_matrix[key_to_id[nprev]][key_to_id[nprev]] = 0
                    nprev = ni
                dist_matrix[key_to_id[nprev]] = dist0
                dist_matrix[key_to_id[nprev]][key_to_id[nprev]] = 0
            
            goals = set(critical_nodes).union(def_reachable)
        else:
            goals = set(self.goals)

            dist_matrix, key_to_id = self.compute_dist_matrix(node['info'], ['A', 'T'])

        # remaining goals        
        keys = goals - node['traversed']
        if 'sedge_id' in node:
            sedge = self.sG_edges[node['sedge_id']]
            at_node = sedge[1] if sedge[0] == node['at'] else sedge[0]
        else:
            at_node = node['at']
        keys.add(at_node)
        keys.add(self.start)
        ids = [key_to_id[key] for key in keys]
        
        dist_matrix = dist_matrix[ids, :][:, ids]
        if at_node != self.start:
            # add dummy node to force a loop that begins at current node and end at staring node
            ncol = np.ones(len(ids)) * self.INF
            ncol[ids.index(key_to_id[self.start])] = 0
            ncol[ids.index(key_to_id[at_node])] = 0
            dist_matrix = np.c_[dist_matrix, ncol]

            nrow = np.ones(len(ids) + 1) * self.INF
            nrow[ids.index(key_to_id[self.start])] = 0
            nrow[-1] = 0
            nrow[ids.index(key_to_id[at_node])] = 0

            dist_matrix = np.r_[dist_matrix, [nrow]]
            

        # FIX the tsp route doesn't guarntee return to base
        #route, cost = self.solve_tsp(dist_matrix)
        route, cost = solve_tsp_dynamic_programming(dist_matrix)
        if 'sedge_id' in node:
            # add the cost of disambiguating a stochastic edge when at an AND node
            cost += self.sG.edges[sedge]['weight']
        return cost
    
    def terminal(self, node):
        if node['at'] == self.start:
            goals = node['goals'] if self.search_reachable_only else set(self.goals)
            if goals.issubset(node['traversed']):
                return True
        return False

    def next_states(self, node):
        if node['type'] == 'OR':
            return self.next_states_or(node)
        elif node['type'] == 'AND':
            return self.next_states_and(node)
        else:
            raise ValueError('Unknown node type')
    
    def get_ambiguous_fronts(self, at, info):
        fronts = set()
        for i, ei in enumerate(list(info)):
            if ei == 'A':
                fronts.add(self.sG_edges[i][0])
                fronts.add(self.sG_edges[i][1])
       
        return fronts
    
    def has_ambiguous_edges(self, at, info):
        if at in self.sG.adj:
            for nbr, datadict in self.sG.adj[at].items():
                edge_id = self.index_sedge(at, nbr)
                if edge_id != -1:
                    if info[edge_id] == 'A':
                        return True
        return False
    
    def get_ambiguous_edges(self, at, info):
        edges = []
        if at in self.sG.adj:
            for nbf, datadict in self.sG.adj[at].items():
                edge_id = self.index_sedge(at, nbf)
                if edge_id != -1:
                    if info[edge_id] == 'A':
                        edges.append(edge_id)
        return edges

    def next_states_and(self, node):
        neighbors = []

        cur = node['at']
        edge_id = node['sedge_id']
        edge = self.sG_edges[edge_id]
        blocking_prob = self.sG.edges[edge]['p']
        if node['info'][edge_id] == 'A':
            info = list(node['info'])
            
            n = {}
            info[edge_id] = 'T'
            n['at'] = edge[0] if edge[1] == cur else edge[1]
            n['traversed'] = node['traversed'].copy()
            n['traversed'].add(n['at'])
            n['traversed'] = n['traversed'].intersection(node['goals'])

            n['info'] = ''.join(info)
            n['prob'] = 1 - blocking_prob
            n['goals'] = node['goals']
            n['cost'] = self.sG.edges[edge]['weight']
            neighbors.append(n)

            n = n.copy()
            info[edge_id] = 'U'
            n['at'] = cur
            n['traversed'] = node['traversed'].copy()
            n['info'] = ''.join(info)
            n['prob'] = blocking_prob
            # check if every goal node is still reachable 
            reachable = self.clinks[n['info'].replace('A', 'T')]

            n['cost'] = self.sG.edges[edge]['weight']
            n['goals'] = reachable
            neighbors.append(n)
        return neighbors

    def next_states_or(self, node):
        """
        node: a AO* node dictionary
        Here is the list of attributes in a node:
        type: OR or AND
        AT: the current location
        traversed: the set of locations traversed
        info: the information vector, e.g. 'UUU'
        solved: whether the node is solved
        h: the heuristic cost-to-go
        f: the best estimated cost
    
        return a list of dictionary, each containing one possible next node
        with the associated cost, state, traversed set and info string
        """
        
        # goals
        goals = node['goals'] if self.search_reachable_only else set(self.goals)
        # Compute TSP weight matrices between where we at and unvisited nodes
        info = node['info']
        dist_matrix, k2i, predecessors = self.compute_dist_matrix(info, ['T'], predecessor=True)
 
        # list of neighbors
        neighbors = []
        # breadth-first search 
        queue = [(frozenset(node['traversed']), node['at'])]
        explored_cost = {(frozenset(node['traversed']), node['at']): 0}
        explored_path = {(frozenset(node['traversed']), node['at']): []}
        root = 0
        while queue:
            n = queue.pop(0)
            nS, na = n
            nc = explored_cost[n]
            # # check whether traversal is complete and can return to start
            # if goals.issubset(nS) and dist_matrix[k2i[na], k2i[self.start]] != np.inf:
            #     pre_j = nx.reconstruct_path(na, self.start, predecessors)
            #     path = explored_path[n] + pre_j[1:]
            #     next = {'at': self.start, 'traversed': set(nS), 'info': info, 'path': path,
            #         'cost': nc + dist_matrix[k2i[na], k2i[self.start]], 'prob': 1, 'goals': node['goals']}
            #     neighbors.append(next)
                
            # if self.has_ambiguous_edges(na, info) != False:
            #     # check whether this node has an ambigiuous stochastic edge
            #     path = explored_path[n]
            #     next = {'at': na, 'traversed': set(nS), 'info': info, 'path': path,
            #         'cost': nc, 'prob': 1, 'goals': node['goals']}
            #     neighbors.append(next)
            # else:
            # add all neighbors to the queue
            #if self.has_ambiguous_edges(na, info) == False or (na in goals) or (na == root):
            fronts = self.get_ambiguous_fronts(na, info)
            ids = list((goals - nS - set([root])).union(fronts))
            for j in ids:
                if (dist_matrix[k2i[na], k2i[j]] < np.inf):
                    # a neighbor is a path towards an unvisited frontier or goal nodes
                    S = set(nS)
                    pre_j = nx.reconstruct_path(na, j, predecessors)
                    S.update(pre_j)
                    S = S.intersection(goals)
                    next = (frozenset(S), j)
                    next_cost = nc + dist_matrix[k2i[na], k2i[j]]
                    if next in explored_cost:
                        if next_cost < explored_cost[next]:
                            explored_cost[next] = next_cost
                            explored_path[next] = explored_path[n] + pre_j[1:]
                            if next not in queue:
                                queue.append(next)
                    else:
                        explored_cost[next] = next_cost
                        explored_path[next] = explored_path[n] + pre_j[1:]
                        queue.append(next)
    
        # Append the go home node
        nS = goals
        mincost = np.inf
        mincost_path = None
        min_nS = None
        for n, nc in explored_cost.items():
            nS, na = n
            if goals.issubset(nS) and dist_matrix[k2i[na], k2i[self.start]] != np.inf:
                cost = explored_cost[(frozenset(nS), na)] + dist_matrix[k2i[na], k2i[self.start]]
                if cost < mincost:
                    mincost = cost
                    pre_j = nx.reconstruct_path(na, self.start, predecessors)
                    mincost_path = explored_path[(frozenset(nS), na)] + pre_j[1:]
                    min_nS = nS
            
            # Apend each disambiguation node
            if self.has_ambiguous_edges(na, info) != False:
                # check whether this node has an ambigiuous stochastic edge
                path = explored_path[n]
                edges = self.get_ambiguous_edges(na, info)
                for edge_id in edges:
                    next = {'at': na, 'traversed': set(nS), 'info': info, 'path': path,
                        'cost': nc, 'prob': 1, 'goals': node['goals'], 'sedge_id': edge_id}
                    neighbors.append(next)
        
        if mincost_path is not None:
            homenode = {'at': self.start, 'traversed': set(min_nS), 'info': info, 'path': mincost_path,
                        'cost': mincost, 'prob': 1, 'goals': node['goals']}
            neighbors.append(homenode)
        return neighbors
    
    def depth(self, tree, n):
        return len(nx.shortest_path(tree, 0, n))

    def expand(self, ao_tree):
        """
        find the expansion node in the ao tree

        find the most promising subtree until reach a leaf node
        """
        nid = 0
        while len(ao_tree.adj[nid]) >= 1:
            best = np.inf
            best_id = None
            for j in ao_tree.adj[nid]:
                if (not ao_tree.nodes[j]['solved']):
                    
                    cost = ao_tree.adj[nid][j]['weight'] + ao_tree.nodes[j]['f']
                    if (cost < best):
                        best = cost
                        best_id = j
            
            # If the most promising subtree has no solution (infinite cost) nor solved node, ao_tree is not solvable
            if best_id is None:
                #print("No feasible policy")
                return None
            nid = best_id
        return nid

    def aostar_stop(self, ao_tree, root):
        if ao_tree.nodes[root]['solved'] == False:
            return False
    
        # Note that if the root node is solved, the tree has the best results. 
        # This is because an OR node is only solved when the lowest cost child is solved. 
        # Hence it is impossible to have a possible lower cost child that is unsolved.
        # AO* will expand until a child node with lower cost is either solved or has higher cost than the root node.
        return True

    def aostar_plan(self, save_steps=False, log=False):
        info = ''.join(['A'] * self.K)
        id = 0 # f'{str(set([0]))}, {self.start}, {info}'
        root = id
        ao_tree = nx.DiGraph()
        
        ao_tree.add_node(id, type='OR', info=info, 
            at=self.start, traversed=set(), solved=False, goals=self.reachable_goals)
        ao_tree.nodes[id]['h'] = self.heuristic(ao_tree.nodes[id])
        ao_tree.nodes[id]['f'] = ao_tree.nodes[id]['h']
        if ao_tree.nodes[id]['h'] == np.inf:
            return np.inf, ao_tree

        while self.aostar_stop(ao_tree, root) == False:
            # select nodes
            if save_steps:
                plot_aotree(ao_tree, Path('results') / 'cache' / f'ao_tree_{id:03d}.gv', view=False)
            front = self.expand(ao_tree)
            if front is None:
                return np.inf, ao_tree
            if log:
                print(ao_tree.nodes[front])

            # expand
            neighbors = self.next_states(ao_tree.nodes[front])
            next_type = 'OR' if ao_tree.nodes[front]['type'] == 'AND' else 'AND'
            for n in neighbors:
                # add node
                id = id + 1 #f'{str(n["traversed"])}, {n["at"]}, {n["info"]}'
                ao_tree.add_node(id, type=next_type, info=n['info'], goals = n['goals'],
                    at=n['at'], traversed=n['traversed'], solved=False)
                if "sedge_id" in n:
                    ao_tree.nodes[id]['sedge_id'] = n['sedge_id']
                ao_tree.nodes[id]['h'] = self.heuristic(ao_tree.nodes[id])
                ao_tree.nodes[id]['f'] = ao_tree.nodes[id]['h']
                path = n['path'] if 'path' in n else None
                ao_tree.add_edge(front, id, weight=n['cost'], p=n['prob'], path=path)

                if self.terminal(ao_tree.nodes[id]):
                    ao_tree.nodes[id]['solved'] = True
                
            self.backpropagate(ao_tree, front)
            #print('depth', self.depth(ao_tree, front) - 1, ao_tree.nodes[front])

        cost = ao_tree.nodes[root]['f']
        if save_steps:
            plot_aotree(ao_tree, Path('results') / 'cache' / f'ao_tree_{id:03d}.gv', view=False)

        return cost, ao_tree


    def backpropagate(self, T, front):
        v = T.nodes[front]
        while front != None:
            children = list(T.neighbors(front))
            if len(children) > 0:
                if v['type'] == 'OR': 
                    # find best child
                    vs = [T.nodes[c]['f'] + T.edges[front, c]['weight'] for c in children]
                    best_child = children[np.argmin(vs)]

                    # update current value                    
                    v['f'] = T.edges[front, best_child]['weight'] + T.nodes[best_child]['f']
                    v['solved'] = True if T.nodes[best_child]['solved'] else False
                
                elif v['type'] == 'AND':
                    # go over all child and sum the expected cost
                    vs = [T.edges[front, c]['p'] * (T.nodes[c]['f'] + T.edges[front, c]['weight']) for c in children]
                    v['f'] = np.sum(vs)
                    v['solved'] = np.all([T.nodes[c]['solved'] for c in children])

                else:
                    raise ValueError('Unknown node type')
            
            # go back to parent until we hit root
            parents = list(T.predecessors(front))
            if len(parents) > 0:
                front = parents[0]
                v = T.nodes[front]
            else:
                front = None


    def permuteall_plan(self):
        pset = powerset(self.sG_edges)
        
        def infostr(t_edges):
            u = {True: 'T', False: 'U'}
            return ''.join([u[e in t_edges] for e in self.sG.edges()])

        answers = []
        expected_cost = 0
        for i, t_edges in enumerate(pset):

            # Add traversable edges to the original graph
            G = self.G.copy()

            prob = 1.0
            for e in self.sG.edges(): 
                edge = self.sG.edges[e]
                if e in t_edges:
                    prob *= (1 - edge['p'])
                    if (e not in G.edges) or edge['weight'] < G.edges[e[0], e[1]]['weight']: 
                        G.add_edge(e[0], e[1], weight=edge['weight'])
                else:
                    prob *= edge['p']
            
            # Compute TSP
            dist_matrix = nx.floyd_warshall_numpy(G)

            ids = np.insert(self.goals, 0, self.start)
            dist_matrix = dist_matrix[ids, :][:, ids]
            ham_path, cost = self.solve_tsp(dist_matrix)
            answers.append(dict(cost=cost, prob=prob, pset=t_edges,
                info_str=infostr(t_edges)))
            expected_cost += cost * prob
    
            print(f"CTP instance {i} with traversable edges @ {t_edges}: cost {cost} prob {prob} ")
        
        return expected_cost    


    def solve_tsp(self, dist_matrix):
        
        # create the data
        data = {}
        # scale dist matrix since lake data is in km
        data['distance_matrix'] = dist_matrix * 1e5
        data['num_vehicles'] = 1
        data['depot'] = 0
        
        # create routing model
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            l = data['distance_matrix'][from_node][to_node]
            if l == np.inf:
                l = self.INF
            return l

        # create distance callback
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # Set cost of travel
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # solution printer
        def get_routes(solution, routing, manager):
            """Get vehicle routes from a solution and store them in an array."""
            # Get vehicle routes and store them in a two dimensional array whose
            # i,j entry is the jth location visited by vehicle i along its route.
            routes = []
            for route_nbr in range(routing.vehicles()):
                index = routing.Start(route_nbr)
                route = [manager.IndexToNode(index)]
                while not routing.IsEnd(index):
                    index = solution.Value(routing.NextVar(index))
                    route.append(manager.IndexToNode(index))
                routes.append(route)
            return routes

        try:
            solution = routing.SolveWithParameters(search_parameters)
        except SystemError:
            return None, np.inf

        route = get_routes(solution, routing, manager)[0]
        # Display the routes.
        #print("TSP route", route, "TSP cost", solution.ObjectiveValue())
        
        return route, solution.ObjectiveValue() / 1e5


def run_one_instance(G, sG, goals, save_steps=False, plot=False, log=False):

    # clear cache
    if save_steps:
        if Path("results/cache").exists():
            shutil.rmtree("results/cache")

    # Visualize problem graph
    if plot:
        plot_problem_graph(G, sG, goals, 'results') 
    pcctp = PCCTP(G, sG, 0, goals, use_h=True, search_reachable_only=True)

    # aostar planning
    ts = time.time()
    expected_cost, ao_tree = pcctp.aostar_plan(save_steps=save_steps, log=log)
    rt = time.time() - ts

    # Visualize AO Tree
    if plot:
        if ao_tree.nodes[0]['solved']:
            plot_policy_tree(ao_tree, Path('results') / 'policy.gv')

    if save_steps:
        plot_aotree(ao_tree, Path('results') / 'ao_tree.gv')
        make_aostar_vod(Path('results') / 'cache')
    
    return expected_cost, ao_tree, rt


def test_runtime(repeat, test_correct = False, rerun=False):
    print("Test runtime")

    n_st, n_ed = 10, 30
    k_st, k_ed = 1, 7
    n = np.arange(n_st, n_ed)
    k = np.arange(k_st, k_ed)
    rt_mu = np.zeros((n_ed-n_st, k_ed-k_st))
    rt_std = np.zeros((n_ed-n_st, k_ed-k_st))

    if rerun:
        for N in n:
            for K in k:
                rts = []
                for _ in range(repeat):
                    G, sG, goals = test_graph.generate_test_graphs(N = N, K = K, n_goals=8)
                    cost, _, rt = run_one_instance(G, sG, goals, save_steps=False, plot=False)
                    if test_correct:
                        ts = time.time()
                        pcctp = PCCTP(G, sG, 0, goals, use_h=False)
                        cost2, _ = pcctp.aostar_plan(save_steps=False, log=False)
                        print("no heuristic", time.time() - ts)
                        assert cost == cost2
                    if cost != np.inf:
                        rts.append(rt)
                rt_mu[N-n_st, K-k_st] = np.mean(rts)
                rt_std[N-n_st, K-k_st] = np.std(rts)
                print(f"N={N} K={K} runtime = {np.mean(rts)} +- {np.std(rts)}")

        np.savetxt(f'results/rt_mu{repeat}.txt', rt_mu)
        np.savetxt(f'results/rt_std{repeat}.txt', rt_std)

    plot_runtime(n, k, f'results/rt_mu{repeat}.txt', f'results/rt_std{repeat}.txt')

if __name__ == "__main__": 
    G, sG, goals = test_graph.G_example()
    expected_cost, ao_tree, rt = run_one_instance(G, sG, goals, save_steps=True, plot=True, log=True)
    save_policy(ao_tree, f"policies_2s/904_4_0.gv")
    #print(rt, "s")
