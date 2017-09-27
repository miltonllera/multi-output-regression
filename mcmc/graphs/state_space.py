import numpy as np
import scipy.sparse as ssp
from itertools import chain

from structure.graphs import DiGraph, MBCGraph, topsort


class DAGState:
    """
    A state space defined as imposing a further restriction over DiGraphs to be Directed Acyclic. This property is
    efficiently checked with an ancestor matrix which is updated after each update of deletion.

    Parameters
    ----------
    graph: DiGraph
        The graph with which to initialize the DAGState. If it is not a DAG an exception is raised


    """

    def __init__(self, graph: DiGraph, ancestor_matrix=None, fan_in=5, copy=False):
        topsort(graph)

        self.adj = graph if not copy else graph.copy()

        if ancestor_matrix is None:
            ancestor_matrix = ssp.lil_matrix(self.adj.shape, dtype=bool)
        else:
            ancestor_matrix = ancestor_matrix.astype(bool)

        for node in topsort(graph):
            ancestors = self.adj.ancestors(node)
            ancestor_matrix[node, ancestors] = 1

        self.ancestor_matrix = ancestor_matrix
        self.fan_in = fan_in

    @property
    def shape(self):
        return self.adj.shape

    def descendants(self, node):
        return self.ancestor_matrix.T[node].nonzero()[1]

    def ancestors(self, node):
        return self.ancestor_matrix[node].nonzero()[1]

    def can_add(self, u, v):
        return self.adj.is_valid_edge(u, v) and self.ancestor_matrix[u, v]

    def add_edge(self, u, v):
        self.adj[u, v] = True
        self._propagate_add(u, v)

    def add_edges(self, edges):
        for u, v in edges:
            self.adj[u, v] = True
            self._propagate_add(u, v)

    def remove_edge(self, u, v):
        self.adj[u, v] = False
        self._propagate_delete(u, v)

    def remove_edges(self, edges):
        for u, v in zip(*edges):
            self.remove_edge(u, v)

    def orphan(self, node):
        if isinstance(node, int):
            parents = self.adj.parents(node)
            for u in parents:
                self.remove_edge(u, node)

        elif isinstance(node, list):
            for n in node:
                parents = self.adj.parents(n)
                for u in parents:
                    self.remove_edge(u, n)
        else:
            raise ValueError()

    def disconnect(self, node):
        parents = self.adj.parents(node)
        for u in parents:
            self.remove_edge(u, node)

        children = self.adj.children(node)
        for v in children:
            self.remove_edge(node, v)

    def has_path(self, u, v):
        return u in self.ancestors(v)

    def copy(self):
        return DAGState(self.adj.copy(), ancestor_matrix=self.ancestor_matrix.copy())

    def _propagate_add(self, u, v):
        """
        Update ancestor matrix after edge addition following the algorithm of Giudici (2003)

        Parameters
        ----------
        u : int
            The tail of the arc that was added
        v : int
            The head of the arc that was added
        """

        new_ancestors = list(self.ancestors(u))
        new_ancestors.append(u)

        to_update = list(self.descendants(v))
        to_update.append(v)

        self.ancestor_matrix[np.ix_(to_update, new_ancestors)] = True

    def _propagate_delete(self, u, v):
        """
        Update ancestor matrix after edge removal following the algorithm of Giudici (2003)

        Parameters
        ----------
        v : int
            The head of the arc that was removed
        """
        nodes = list(self.descendants(v))
        nodes.append(v)

        self.ancestor_matrix[nodes] = False

        for n in topsort(self.adj, nodes):
            parents = self.adj.parents(n)
            ancestors = chain.from_iterable(self.ancestors(u) for u in parents)

            self.ancestor_matrix[n, list(set(chain(parents, ancestors)))] = True

    def reversible_edges(self, rev=True):
        if not rev:
            raise NotImplementedError()

        return self.adj.nonzero()

    def non_admissible_edges(self):
        return slice(None), self.adj.A.sum(axis=0) >= self.fan_in


class MBCState(DAGState):
    def __init__(self, graph: MBCGraph, ancestor_matrix=None, fan_in=5):
        super().__init__(graph, ancestor_matrix, fan_in)
        self.fixed_direction_edges = \
            np.ix_(np.arange(self.adj.n_features), np.arange(self.adj.n_features + 1, self.shape[0]))

    def copy(self):
        return MBCState(self.adj.copy(), ancestor_matrix=self.ancestor_matrix.copy())

    def non_admissible_edges(self):
        edges = np.zeros(self.adj.shape, dtype=bool)
        edges[:, self.adj.A.sum(axis=0) >= self.fan_in] = True
        edges[np.ix_(range(self.adj.n_features), np.arange(self.adj.n_targets) + self.adj.n_features)] = True

        return edges.nonzero()

    def reversible_edges(self, rev=True):
        if not rev:
            raise NotImplementedError()

        edges = np.ones(self.adj.shape, dtype=bool)
        edges[np.ix_(np.arange(self.adj.n_targets) + self.adj.n_features, range(self.adj.n_features))] = False

        edges = np.logical_and(edges, self.adj.A)

        return edges.nonzero()


class RestrictionViolation(Exception):
    """Exception for restrictions violations imposed to graphs"""
