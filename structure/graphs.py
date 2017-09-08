import networkx as nx
import pylab as pl
import scipy.sparse as ssp
import scipy.sparse.csgraph as csgraph


class DiGraph(ssp.lil_matrix):
    def __init__(self, arg1, shape=None, dtype=None, copy=False, names=None):
        if dtype is None:
            dtype = bool
        elif dtype != bool:
            raise ValueError('Either adjacency or weighted graph')

        super().__init__(arg1, shape, dtype, copy)
        self._names = names

    @property
    def n_nodes(self):
        return self.shape[0]

    @property
    def n_edges(self):
        return self.count_nonzero()

    def nodes(self, as_names=False):
        if as_names and self._names is not None:
            return self._names

        return list(range(self.shape[0]))

    def nodes_iter(self, as_names=False):
        if as_names and self._names is not None:
            return iter(self._names)

        return range(self.shape[0])

    def edges(self):
        return list(zip(*self.nonzero()))

    def edges_iter(self):
        return zip(*self.nonzero())

    def add_edge(self, u, v, value=None):
        if not self.can_add(u, v):
            raise ValueError('Edge {0}-->{1} cannot be added'.format(u, v))

        if value is None:
            value = 1
        self[u, v] = value

    def add_edges(self, edges, value=None):
        if any(map(lambda e: not self.can_add(*e), edges)):
            raise ValueError('At least one edge cannot be added')

        if value is None:
            value = 1
        us, vs = list(zip(*edges))
        self[us, vs] = value

    def remove_edge(self, u, v):
        self[u, v] = 0

    def remove_edges(self, edges):
        us, vs = zip(*edges)
        self[us, vs] = 0

    def parents(self, node):
        return self.T[node].nonzero()[1]

    def children(self, node):
        return self[node].nonzero()[1]

    def descendants(self, node, sort=False):
        descendants = csgraph.breadth_first_order(self, i_start=node, directed=True, return_predecessors=False)[1:]

        if not sort:
            return descendants
        return sorted(descendants)

    def ancestors(self, node, sort=False):
        ancestors = csgraph.breadth_first_order(self.T, i_start=node, directed=True, return_predecessors=False)[1:]

        if not sort:
            return ancestors
        return sorted(ancestors)

    def can_add(self, u, v):
        return u != v


class MBCGraph(DiGraph):
    def __init__(self, arg1, features, shape=None, dtype=None, copy=False, names=None):
        super().__init__(arg1, shape, dtype, copy, names)
        targets = set(self.nodes())
        targets.remove(features)

        self.features = set(features)
        self.targets = targets

    def can_add(self, u, v):
        return u != v and (u in self.targets or {u, v} <= self.features)


def topsort(G: ssp.spmatrix, nbunch=None, reverse=True):
    order = []
    seen = set()
    explored = set()

    if nbunch is None:
        nbunch = G.nodes_iter()
    for v in nbunch:  # process all vertices in G
        if v in explored:
            continue
        fringe = [v]  # nodes yet to look at
        while fringe:
            w = fringe[-1]  # depth first search
            if w in explored:  # already looked down this branch
                fringe.pop()
                continue
            seen.add(w)  # mark as seen
            # Check successors for cycles and for new nodes
            new_nodes = []
            for n in G[w].nonzero()[1]:
                if n not in explored:
                    if n in seen:  # CYCLE !!
                        raise nx.NetworkXUnfeasible("Graph contains a cycle.")
                    new_nodes.append(n)
            if new_nodes:  # Add new_nodes to fringe
                fringe.extend(new_nodes)
            else:  # No new nodes so w is fully explored
                explored.add(w)
                order.append(w)
                fringe.pop()  # done considering this node
    if reverse:
        return order
    else:
        return list(reversed(order))


def plot_digraph(graph: DiGraph):
    nx.draw_networkx(nx.from_scipy_sparse_matrix(graph))
    pl.plot()
