import scipy.sparse as ssp
import networkx as nx

from structure.graphs import DiGraph, RegressorDiGraph, topsort


class DAGStateSpace:
    def __init__(self, graph: DiGraph):

        self.adj = DiGraph(graph, shape=graph.shape, dtype=bool)

        ancestor_matrix = ssp.lil_matrix(self.adj.shape)

        for node in topsort(graph):
            parents = self.adj.ancestors(node)
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
        self._propagate_delete(u, v)

    def add_edges(self, edges):
        for u, v in zip(*edges):
            self.add_edge(u, v)

    def remove_edge(self, u, v):
        self.adj[u, v] = False
        self._propagate_delete(u, v)

    def remove_edges(self, edges):
        for u, v in  zip(*edges):
            self.remove_edge(u, v)

    def orphan(self, node):
        parents = self.adj.parents(node)
        for u in parents:
            self.remove_edge(u, node)

    def disconnect(self, node):
        parents = self.adj.parents(node)
        for u in parents:
            self.remove_edge(u, node)

        children = self.adj.children(node)
        for v in children:
            self.remove_edge(node, v)

    def has_path(self, u, v):
        return u in self.ancestors(v)

    def _propagate_add(self, u, v):
        """
        Update ancestor matrix after edge addition following the algorithm on Giudici, (2003)

        Parameters
        ----------
        u : int
            The tail of the arc that was added
        v : int
            The head of the arc that was added
        """

        to_update = self.ancestors(u)
        self._ancestor_matrix[v, u] = True
        self._ancestor_matrix[v, to_update] = True

        for w in self.descendants(v):
            self._ancestor_matrix[w, u] = True
            self._ancestor_matrix[w, to_update] = True

    def _propagate_delete(self, u, v):
        """
        Update ancestor matrix after edge removal following the algorithm on Giudici, (2003)

        Parameters
        ----------
        v : int
            The head of the arc that was removed
        """

        self._ancestor_matrix[v] = False
        self._ancestor_matrix[v, self.adj.parents(v)] = True

        for n in self.adj.parents(v):
            self._ancestor_matrix[v, self.ancestors(n)] = True

        descendants = self.descendants(v)

        for s in topsort(self.adj, descendants):
            pred = self.adj.parents(s)
            self._ancestor_matrix[s] = False
            self._ancestor_matrix[s, pred] = True
            for n in pred:
                self._ancestor_matrix[s, self.ancestors(n)] = True


class MBCStateSpace(DAGStateSpace):
    def __init__(self, graph: RegressorDiGraph):
        super().__init__(graph)


def check_consistency(state: DAGStateSpace):
    for v in state.adj.nodes_iter():
        for u in state.ancestors(v):
            if not state.has_path(u, v):
                raise nx.NetworkXNoPath('No path from {0} to {1} but ancestor matrix says there is'.format(u, v))
    return True
