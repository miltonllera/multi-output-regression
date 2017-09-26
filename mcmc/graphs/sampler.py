import numpy as np
from collections import OrderedDict

from structure.graphs import DiGraph
from core.misc import get_rng
from mcmc.sampling import DebuggerHook, SamplingTrace, metropolis_hastings, Trace


# noinspection PyAttributeOutsideInit
class MHStructureSampler:
    """
    A class that implements sampling in the space of DAGs using the Metropolis-Hastings Sampler

    Parameters
    ----------
    proposal: ProposalDistribution
        The proposal distribution to use. If provided move_prob is ignored and parent_set scores_ are not calculated.
    n_steps: int
        The number of sample networks to keep.
    sample_freq: int
        Haw many iterations until the next sample is saved.
    burn_in: int or None (default)
        The amount of iterations before the algorithm stores samples.
    verbose: bool
        Whether to print the progress of the algorithm

    rng: numpy.random.RandomState
        The random number generator used by the sampler. If none will use the default one.
    """
    def __init__(self, proposal, n_steps=1000, sample_freq=1, burn_in=None, verbose=False, rng=None):
        if burn_in is None:
            burn_in = n_steps // 10

        self.proposal = proposal
        self.n_steps = n_steps
        self.burn_in = burn_in
        self.sample_freq = sample_freq
        self.rng = get_rng(rng)
        self.verbose = verbose

    @property
    def returned_samples(self):
        return (self.n_steps - self.burn_in) // self.sample_freq

    def generate_samples(self, data, return_scores=False, debug=False):
        """
        Sample graphs using Metropolis-Hastings starting at a given value

        Parameters
        ----------
        data: (N, D) array of training samples
            The data used to score the parent sets of the nodes used for sampling

        return_scores: bool (default False)
            If true, will return the score of the graphs sampled.

        debug: bool (default False)
            If true will plot the sampled graphs and check if they are consistent with DAG restrictions at each step.

        Returns
        -------
        list of DiGraphs:
            If return_scores is False will return the list of DiGraphs

        list of Traces:
            If return_scores is True will return the 2-tuples of (DiGraph, float) with the pairs of corresponding
            graphs and scores_

        """

        if self.verbose:
            print('initializing sampling.......')

        self.proposal.initialize(data)

        if self.verbose:
            print('done.\ncollecting samples.......')

        if debug:
            hook = DebuggerHook(self.sample_freq, self.burn_in, verbose=self.verbose)
        else:
            hook = SamplingTrace(self.sample_freq, self.burn_in, verbose=self.verbose)

        s0 = self.proposal.random_state()

        graphs = metropolis_hastings(s0, self.proposal, self.n_steps, self.burn_in, self.sample_freq, hook, self.rng)
        print('done.')

        graphs = [g.adj for g in graphs]

        if return_scores:
            return Trace(graphs, np.asarray(hook.scores_))

        return graphs


class DAGDistribution:
    """
    Model the empirical distribution of DAGs. Samples obtained through some method (right now only MCMC is implemented)
    are stored and used to compute statistics like the probability of edges and parent sets.

    Due to fact that in general there are many more graphs than can be sampled, estimating probabilities of graphs,
    using the joint probability of arcs is not very robust.

    Parameters
    ----------
    samples: list of DiGraphs
        The sampled graphs stored as sparse matrix representing the adjacency list of a graph

    """
    def __init__(self, samples):
        if not all(isinstance(s, DiGraph) for s in samples):
            raise ValueError('Samples must be instances of DiGraph')

        self.samples = samples

    def edge_prob(self, e):
        u, v = e
        return len(list(filter(lambda g: g[u, v], self.samples))) / len(self.samples)

    def edge_conditional_prob(self, e, given):
        u, v = e
        s = list(filter(lambda g: all(g[u, v] for u, v in given), self.samples))
        den = len(s) + 1

        num = len(list(filter(lambda g: g[u, v], s)))

        return num / den

    def graph_probability(self, graph):
        graph = set(graph)
        num = np.sum([graph <= set(s.flatnonzero()) for s in self.samples])

        return num / len(self.samples)

    def get_param_values(self, edges):
        params = []
        for u, v in edges:
            params.append([g[u, v] for g in self.samples])

        kv_pairs = sorted(zip(edges, params), key=lambda kv: kv[0])
        parameter_traces = OrderedDict(kv_pairs)

        return parameter_traces
