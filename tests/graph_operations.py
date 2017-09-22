from numpy.random import RandomState
from networkx import nx

from structure.graph_generation import random_dag

seeds = list(range(101, 200))
rng = RandomState()
variables = list(range(15))

for i, s in enumerate(seeds):
    print('Test {0}/{1}'.format(i + 1, len(seeds)))

    rng.seed(s)
    g = random_dag(variables, rng=rng)
    nx_g = nx.from_scipy_sparse_matrix(g, create_using=nx.DiGraph())

    real_edges = g.edges()
    edges = real_edges.copy()

    while len(edges):

        if set(edges) != set(nx_g.edges()):
            raise ValueError('Error in graph created with seed {2}\n Expected edges: {0}\n got: {1}'
                             .format(s, nx_g.edges(), g.edge()))
        for v in variables:
            real_parents = set(nx_g.predecessors(v))
            parents = set(g.parents(v))

            if parents != real_parents:
                raise ValueError('Error in graph created with seed {2}\n Expected parents for node {3}: {0}\n got: {1}'
                                 .format(s, real_parents, parents, v))

            real_children = set(nx_g.successors(v))
            children = set(g.children(v))

            if parents != real_parents:
                raise ValueError('Error in graph created with seed {2}\n Expected children for node {3}: {0}\n got: {1}'
                                 .format(s, real_children, children, v))

            nx_g_closure = nx.transitive_closure(nx_g)

            real_ancestors = set(nx_g_closure.predecessors(v))
            ancestors = set(g.ancestors(v))

            if parents != real_parents:
                raise ValueError(
                    'Error in graph created with seed {2}\n Expected ancestors for node {3}: {0}\n got: {1}'
                                 .format(s, real_ancestors, ancestors, v))

            real_descendants = set(nx_g.successors(v))
            descendants = set(g.children(v))

            if parents != real_parents:
                raise ValueError('Error in graph created with seed {2}\n Expected children for node {3}: {0}\n got: {1}'
                                 .format(s, real_children, children, v))

        e = edges[rng.choice(len(edges))]

        g.remove_edge(*e)
        nx_g.remove_edge(*e)
        edges.remove(e)

    print('... pass')
