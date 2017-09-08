import numpy as np
import networkx as nx
from itertools import chain
import scipy.sparse as ssp

from graph.graphs import DiGraph


class DAGStateSpace:
    def __init__(self, graph: DiGraph):

        self.adj = DiGraph(graph, shape=graph.shape, dtype=bool)

        ancestor_matrix = ssp.lil_matrix(self.adj.shape)

        for node in nx.topological_sort(nx.from_scipy_sparse_matrix(self, create_using=nx.DiGraph())):
            parents = self.adj.parents(node)
            ancestor_matrix[node, parents] = 1

        self._ancestor_matrix = ancestor_matrix.tocsr()

    def descendants(self, node):
        return self._ancestor_matrix.T[node].nonzero()[1]

    def ancestors(self, node):
        return self._ancestor_matrix[node].nonzero()[1]

    def can_add(self, u, v):
        return u != v and self._ancestor_matrix[u, v]

    def add_edge(self, u, v):
        self.adj[u, v] = True

    def add_edges(self, edges):
        us, vs = zip(*edges)
        self.adj[us, vs] = True

    def remove_edge(self, u, v):
        self.adj[u, v] = False

    def remove_edges(self, edges):
        us, vs = zip(*edges)
        self.adj[us, vs] = False

    def orphan(self, node):
        self.adj[node] = False


# class MBCStateSpace:
#     def __init__(self):
