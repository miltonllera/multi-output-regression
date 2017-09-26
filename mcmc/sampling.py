import numpy as np
from recordclass.record import recordclass

from mcmc.graphs.checks import check_consistency
from structure.graphs import plot_digraph
from core.misc import get_rng

Trace = recordclass('Trace', ['networks', 'scores_'])


def metropolis_hastings(s0, proposal, n_steps=1000, burn_in=None, save_iter=1, iter_hook=None, rng=None):
    """
    General Metropolis Hastings sampler(Hastings et.at. (1953), Hastings (1970)).
     Based on the pseudo-code found in Murphy (2009).

    The algorithms uses a proposal distribution to propose new states and accepts them with probability dependant on
    it's score as defined by said proposal.

    iter_hook is executed after each iteration to record relevant statistics as defined by the user.

    Parameters:
    ----------
    s0: State
        Starting sample for the sampling run
    proposal: mcmc.proposal.ProposalDistribution
        The distribution used to propose states given the current state in the sampling process
    n_steps: int
        The total number of sampling steps to run
    burn_in: int, dafault
        The number of initial samples to discard before

    Returns
    -------
    list of States:
        The list of recorded States of the underlying Markov Chain

    """

    init_score = proposal.score_fn_(s0.adj)

    s, s_score = s0, init_score
    samples = []

    for i in range(n_steps):

        next_s, acceptance, score_diff = proposal.sample(s)

        r = np.exp(min(0, acceptance))
        p = rng.random_sample()

        if p < r:
            s = next_s
            s_score += score_diff

        if i + 1 > burn_in and (i + 1) % save_iter == 0:
            samples.append(s)

        if iter_hook is not None:
            iter_hook(i, n_steps, s, s_score, p, r)

    return samples


class ProposalDistribution:
    def __init__(self, prior=None, random_state=None):
        if prior is None:
            prior = lambda x: 1

        self.prior = prior
        self.rng = get_rng(random_state)

    def initialize(self, data, **kwargs):
        raise NotImplementedError()

    def sample(self, state):
        raise NotImplementedError()

    def random_state(self):
        raise NotImplementedError()


class IterationHook:
    def __call__(self, i, n_steps, s, score, p, r):
        raise NotImplementedError


class SamplingTrace(IterationHook):
    def __init__(self, save_iter=1, discard=0, verbose=False):
        self.save_iter = save_iter
        self.discard = discard
        self.verbose = verbose

    def __call__(self, i, n_steps, s, score, p, r):
        if i == 0:
            self.scores_ = []
            self.accept_ratios_ = []
            self.accepted_ = 0

        i += 1
        if p < r:
            self.accepted_ += 1
            if self.verbose:
                print('Iteration {0}/{1}... accepted'.format(i, n_steps))
        else:
            print('Iteration {0}/{1}... rejected'.format(i, n_steps))

        print('\tCurrent graph score: {:.4f} \n\tAcceptance ratio: {:.2f}'.format(score, self.accepted_ / i))

        if i > self.discard and i % self.save_iter == 0:
            self.scores_.append(score)
            self.accept_ratios_.append(self.accepted_ / i)


class StatePlotter(IterationHook):
    def __call__(self, i, n_steps, s, score, p, r):
        plot_digraph(s.adj)


class DebuggerHook(SamplingTrace, StatePlotter):
    def __call__(self, i, n_steps, s, score, p, r):
        SamplingTrace.__call__(self, i, n_steps, s, score, p, r)
        StatePlotter.__call__(self, i, n_steps, s, score, p, r)
        check_consistency(s)
