"""
Modified CausalGraph class from causal-learn.

Patches applied (from RCD source):
- Added: mi (marginal independence) tracking, citest_cache, no_ci_tests counter
- Added: append_to_mi(), remove_edge() methods
- Modified: set_ind_test() sets ci_test_hash_key
- Modified: ci_test() adds caching and test counter
"""

import io
import warnings
from itertools import permutations

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.Helper import list_union, powerset


class CausalGraph:
    def __init__(self, no_of_var, labels={}):
        node_names = [("X%d" % (i + 1)) for i in range(no_of_var)]
        nodes = []
        for name in node_names:
            node = GraphNode(name)
            nodes.append(node)
        self.G = GeneralGraph(nodes)
        for i in range(no_of_var):
            for j in range(i + 1, no_of_var):
                self.G.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))

        self.labels = labels
        self.data = None
        self.test = None
        self.corr_mat = None
        self.sepset = np.empty((no_of_var, no_of_var), object)
        self.p_values = np.empty((no_of_var, no_of_var), object)
        # --- RCD additions ---
        self.mi = np.empty(no_of_var, object)  # store the set of Marginal Independent nodes
        self._mi_index = 0
        # --- end RCD additions ---
        self.definite_UC = []
        self.definite_non_UC = []
        self.PC_elapsed = -1
        self.redundant_nodes = []
        self.nx_graph = nx.DiGraph()
        self.nx_skel = nx.Graph()
        self.prt_m = {}
        self.mvpc = None
        self.cardinalities = None
        self.is_discrete = False
        # --- RCD additions ---
        self.citest_cache = dict()
        self.data_hash_key = None
        self.ci_test_hash_key = None
        self.no_ci_tests = 0
        # --- end RCD additions ---

    # --- RCD additions ---
    def append_to_mi(self, node):
        """Add a node to marginal independence set."""
        self.mi[self._mi_index] = node
        self._mi_index += 1

    def remove_edge(self, x, y):
        edge1 = self.G.get_edge(self.G.nodes[x], self.G.nodes[y])
        if edge1 is not None:
            self.G.remove_edge(edge1)
        edge2 = self.G.get_edge(self.G.nodes[y], self.G.nodes[x])
        if edge2 is not None:
            self.G.remove_edge(edge2)
    # --- end RCD additions ---

    def set_ind_test(self, indep_test, mvpc=False):
        """Set the conditional independence test that will be used"""
        if mvpc:
            self.mvpc = True
        self.test = indep_test
        # --- RCD addition ---
        self.ci_test_hash_key = hash(indep_test)
        # --- end RCD addition ---

    def ci_test(self, i, j, S):
        """Define the conditional independence test"""
        # --- RCD addition: count tests ---
        self.no_ci_tests += 1
        # --- end RCD addition ---
        if self.mvpc:
            return self.test(self.data, self.nx_skel, self.prt_m, i, j, S, self.data.shape[0])

        # --- RCD modification: caching ---
        i, j = (i, j) if (i < j) else (j, i)
        ijS_key = (i, j, frozenset(S), self.data_hash_key, self.ci_test_hash_key)
        if ijS_key in self.citest_cache:
            return self.citest_cache[ijS_key]
        # --- end RCD modification ---

        pValue = self.test(i, j, S)

        # --- RCD addition: store in cache ---
        self.citest_cache[ijS_key] = pValue
        # --- end RCD addition ---
        return pValue

    def neighbors(self, i):
        """Find the neighbors of node i in adjmat"""
        return np.where(self.G.graph[i, :] != 0)[0]

    def max_degree(self):
        """Return the maximum number of edges connected to a node in adjmat"""
        return max(np.sum(self.G.graph != 0, axis=1))

    def find_arrow_heads(self):
        L = np.where(self.G.graph == 1)
        return list(zip(L[1], L[0]))

    def find_tails(self):
        L = np.where(self.G.graph == -1)
        return list(zip(L[1], L[0]))

    def find_undirected(self):
        return [(edge[0], edge[1]) for edge in self.find_tails() if self.G.graph[edge[0], edge[1]] == -1]

    def find_fully_directed(self):
        return [(edge[0], edge[1]) for edge in self.find_arrow_heads() if self.G.graph[edge[0], edge[1]] == -1]

    def find_bi_directed(self):
        return [(edge[1], edge[0]) for edge in self.find_arrow_heads() if (
                self.G.graph[edge[1], edge[0]] == Endpoint.ARROW.value and self.G.graph[
            edge[0], edge[1]] == Endpoint.ARROW.value)]

    def find_adj(self):
        return list(self.find_tails() + self.find_arrow_heads())

    def is_undirected(self, i, j):
        return self.G.graph[i, j] == -1 and self.G.graph[j, i] == -1

    def is_fully_directed(self, i, j):
        return self.G.graph[i, j] == -1 and self.G.graph[j, i] == 1

    def find_unshielded_triples(self):
        return [(pair[0][0], pair[0][1], pair[1][1]) for pair in permutations(self.find_adj(), 2)
                if pair[0][1] == pair[1][0] and pair[0][0] != pair[1][1] and self.G.graph[pair[0][0], pair[1][1]] == 0]

    def find_triangles(self):
        Adj = self.find_adj()
        return [(pair[0][0], pair[0][1], pair[1][1]) for pair in permutations(Adj, 2)
                if pair[0][1] == pair[1][0] and pair[0][0] != pair[1][1] and (pair[0][0], pair[1][1]) in Adj]

    def find_kites(self):
        return [(pair[0][0], pair[0][1], pair[1][1], pair[0][2]) for pair in permutations(self.find_triangles(), 2)
                if pair[0][0] == pair[1][0] and pair[0][2] == pair[1][2]
                and pair[0][1] < pair[1][1] and self.G.graph[pair[0][1], pair[1][1]] == 0]

    def find_cond_sets(self, i, j):
        neigh_x = self.neighbors(i)
        neigh_y = self.neighbors(j)
        pow_neigh_x = powerset(neigh_x)
        pow_neigh_y = powerset(neigh_y)
        return list_union(pow_neigh_x, pow_neigh_y)

    def find_cond_sets_with_mid(self, i, j, k):
        return [S for S in self.find_cond_sets(i, j) if k in S]

    def find_cond_sets_without_mid(self, i, j, k):
        return [S for S in self.find_cond_sets(i, j) if k not in S]

    def rearrange(self, PATH):
        raw_col_names = list(pd.read_csv(PATH, sep='\t').columns)
        var_indices = []
        for name in raw_col_names:
            var_indices.append(int(name.split('X')[1]) - 1)
        new_indices = np.zeros_like(var_indices)
        for i in range(1, len(new_indices)):
            new_indices[var_indices[i]] = range(len(new_indices))[i]
        output = self.adjmat[:, new_indices]
        output = output[new_indices, :]
        self.adjmat = output

    def to_nx_graph(self):
        if self.labels == {}:
            nodes = range(len(self.G.graph))
            self.labels = {i: self.G.nodes[i].get_name() for i in nodes}

        self.nx_graph.add_nodes_from(self.labels.values())
        undirected = self.find_undirected()
        directed = self.find_fully_directed()
        bidirected = self.find_bi_directed()
        for (i, j) in undirected:
            self.nx_graph.add_edge(self.labels[i], self.labels[j], color='g')
        for (i, j) in directed:
            self.nx_graph.add_edge(self.labels[i], self.labels[j], color='b')
        for (i, j) in bidirected:
            self.nx_graph.add_edge(self.labels[i], self.labels[j], color='r')

    def to_nx_skeleton(self):
        nodes = range(len(self.G.graph))
        self.nx_skel.add_nodes_from(nodes)
        adj = [(i, j) for (i, j) in self.find_adj() if i < j]
        for (i, j) in adj:
            self.nx_skel.add_edge(i, j, color='g')

    def draw_nx_graph(self, skel=False):
        if not skel:
            print("Green: undirected; Blue: directed; Red: bi-directed\n")
        warnings.filterwarnings("ignore", category=UserWarning)
        g_to_be_drawn = self.nx_skel if skel else self.nx_graph
        edges = g_to_be_drawn.edges()
        colors = [g_to_be_drawn[u][v]['color'] for u, v in edges]
        pos = nx.circular_layout(g_to_be_drawn)
        nx.draw(g_to_be_drawn, pos=pos, with_labels=True, labels=self.labels, edge_color=colors)
        plt.draw()
        plt.show()

    def draw_pydot_graph(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        pyd = GraphUtils.to_pydot(self.G)
        tmp_png = pyd.create_png(f="png")
        pyd.write_png("result.png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        plt.axis('off')
        plt.imshow(img)
        plt.show()