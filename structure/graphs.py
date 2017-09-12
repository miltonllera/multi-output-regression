import networkx as nx
import pylab as pl
import scipy.sparse as ssp
import scipy.sparse.csgraph as csgraph
import pygraphviz as pgv


class DiGraph(ssp.lil_matrix):
    """
    An implementation of a directed graph with a Sparse Matrix representation using Scipy's sparse module.
    Specifically the lil_matrix representation is used since it allows for efficient modification of the
    sparse structure, which is useful for sampling.

    Parameters
    ----------
    arg1: object

        This can be instantiated in several ways:
            DiGraph(D)
                with a dense matrix or rank-2 ndarray D

            DiGraph(S)
                with another sparse matrix S (equivalent to S.tolil())

            DiGraph((M, N), [dtype])
                to construct an empty matrix with shape (M, N)
                dtype is optional, defaulting to dtype='d'.

    shape: 2-tuple
        The size of the underlying dimensions

    dtype: type
        The type of the data. Supported are bool for adjacency representations, and float for weighted edges

    copy: bool
        In case arg1 is a sparse matrix, whether to copy its contents when constructing a new instance

    names: list of strings
        A list of true names for the nodes of the graph

    Attributes
    ----------
    names: list
        The list of names of the nodes if any. Useful if using non numerical identifiers for the nodes

    """
    def __init__(self, arg1, shape=None, dtype=bool, copy=False, names=None):
        if dtype is None:
            dtype = bool
        elif dtype not in [bool, float]:
            raise ValueError('Either adjacency or weighted graph')

        super().__init__(arg1, shape, dtype, copy)
        self._names = names

    @property
    def n_nodes(self):
        return self.shape[0]

    @property
    def n_edges(self):
        return self.count_nonzero()

    @property
    def names(self):
        if self._names is None:
            return list(range(self.n_nodes))
        return self._names

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

    def has_path(self, u, v):
        return u in self.ancestors(v)

    def can_add(self, u, v):
        return u != v

    def copy(self):
        a = DiGraph(arg1=self, copy=True, names=self._names)
        return a


class RegressorDiGraph(DiGraph):
    def __init__(self, arg1, n_features, shape=None, dtype=None, copy=False, names=None):
        super().__init__(arg1, shape, dtype, copy, names)
        targets = set(self.nodes())
        targets.remove(list(range(n_features)))

        self.features = set(n_features)
        self.targets = targets

    def can_add(self, u, v):
        return u != v and (u in self.targets or {u, v} <= self.features)


def topsort(G: ssp.spmatrix, nodes=None, reverse=True):
    order = []
    seen = set()
    explored = set()

    if nodes is None:
        nodes = G.nodes_iter()
    for v in nodes:  # process all vertices in G
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
    nx.draw_networkx(nx.from_scipy_sparse_matrix(graph, create_using=nx.DiGraph()))
    pl.plot()


def load_graph(path):
    dot_graph = pgv.AGraph(filename=path)

    if 'names' in dot_graph.graph_attr:
        names = dot_graph.graph_attr['names']
    else:
        names = None

    dtype = dot_graph.graph_attr['data_type']

    if dtype == 'bool':
        dtype = bool
    elif dtype == 'float64':
        dtype = float
    else:
        raise ValueError('Unrecognized data type')

    n_nodes = dot_graph.number_of_nodes()
    graph = DiGraph((n_nodes, n_nodes), dtype=dtype, names=names)

    if dtype == bool:
        u, v = zip(dot_graph.edges_iter())
        u = list(map(int, u))
        v = list(map(int, v))

        graph[u, v] = True

    else:
        for u, v in dot_graph.edges():
            weight = dot_graph.get_edge(u, v).attr['weight']
            graph[int(u), int(v)] = weight

    return graph


def save_graph(graph: DiGraph, path):
    if path[-3:] != '.gv' and path[-4:] != '.dot':
        path += '.gv'

    if graph._names is None:
        dot_graph = pgv.AGraph(data_type=str(graph.dtype))
    else:
        dot_graph = pgv.AGraph(data_type=str(graph.dtype), names=graph._names)

    dot_graph.add_nodes_from(graph.nodes())

    if graph.dtype == bool:
        dot_graph.add_edges_from(graph.edges())

    else:
        for u, v in graph.edges_iter():
            dot_graph.add_edge(u, v, weight=graph[u, v])

    dot_graph.write(path)
