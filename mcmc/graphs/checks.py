import networkx as nx

from mcmc.graphs.state_space import RestrictionViolation, DAGState


def check_distribution(v, dist, fan_in):
    for ps in dist.parent_sets:
        if len(ps) > fan_in:
            raise RestrictionViolation("Parent set for node {0} is bigger than the fan in restriction")
        if v in ps:
            raise RestrictionViolation("Node {v} is in one of its own parent sets")


def check_consistency(state: DAGState):
    nx_digraph = nx.from_scipy_sparse_matrix(state.adj, create_using=nx.DiGraph())

    for v in state.adj.nodes_iter():
        for u in state.ancestors(v):

            if not state.has_path(u, v):
                # Check has_path for graph
                if nx.has_path(nx_digraph, u, v):
                    raise ValueError('DiGraph.has_path is not consistent')

                # Check ancestor matrix
                raise nx.NetworkXNoPath('No path from {0} to {1} but ancestor matrix says there is'.format(u, v))
            else:
                if not nx.has_path(nx_digraph, u, v):
                    raise ValueError(
                        'DiGraph.has_path is not consistent. Path from {0} to {1} does not exist'.format(u, v))

            if state.has_path(v, u):
                if not nx.has_path(nx_digraph, v, u):
                    raise ValueError(
                        'DiGraph.has_path is not consistent. Path from {0} to {1} does not exist'.format(v, u))

                raise RestrictionViolation(
                    'DAG restriction violation. Cycle detected between node {0} and node {1}'.format(v, u))
            else:
                if nx.has_path(nx_digraph, v, u):
                    raise ValueError(
                        'DiGraph.has_path is not consistent. Path from {0} to {1} does exist'.format(v, u))

        card_pa = len(state.adj.parents(v))
        if len(state.adj.parents(v)) > state.fan_in_:
            raise RestrictionViolation(
                'Fan in restriction violated. |Pa({0})| = {1} > {2}'.format(v, card_pa, state.fan_in_))

    return True
