import numpy as np
from core.misc import get_rng
from recordclass.record import recordclass

from mcmc.hooks import SamplingTrace, DebuggerHook


Trace = recordclass('Trace', ['networks', 'scores'])


def metropolis_hastings(s0, proposal, n_steps=1000, burn_in=None, save_iter=1, iter_hook=None, rng=None):
    """
    General Metropolis Hastings sampler(Hastings et.at. (1953), Hastings (1970)).
    Based on the pseudocode found in Murphy (2009).

    Parameters:
    ----------
    s0: object
        Starting sample for the sampling run
    proposal: mcmc.proposal.ProposalDistribution
        The distribution used to propose states given the current state in the sampling process
    n_steps: int
        The total number of sampling steps to run
    burn_in: int, dafault
        The number of initial samples to discard before
    """

    init_score = proposal.score_fn(s0)

    s, s_score = s0, init_score
    samples = []

    for i in range(n_steps):

        next_s, acceptance, score_diff = proposal.sample(s)

        r = np.exp(min(0, acceptance))
        p = rng.random_sample()

        if p < r:
            s = next_s

        if i + 1 > burn_in and (i + 1) % save_iter == 0:
            samples.append(s)

        if iter_hook is not None:
            iter_hook(i, s, s_score, p, r)

    return samples


# noinspection PyAttributeOutsideInit
class MHStructureSampler:
    """
    A class that implements sampling in the space of DAGs using the Metropolis-Hastings Sampler

    Parameters
    ----------
    proposal: ProposalDistribution
        The proposal distribution to use. If provided move_prob is ignored and parent_set scores are not calculated.
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

    def generate_samples(self, data, s0, return_scores=False, debug=True):
        """
        Sample networks using Metropolis Hastings starting at a given value

        :param data:
        :param s0:
        :param return_scores:
        :param debug:
        :return:
        """

        if self.verbose:
            print('initializing sampling.......')

        self.proposal.initialize(data, verbose=self.verbose)

        if self.verbose:
            print('done.\ncollecting samples.......')

        if debug:
            hook = DebuggerHook(self.sample_freq, self.burn_in)
        else:
            hook = SamplingTrace(self.sample_freq, self.burn_in)

        networks = metropolis_hastings(s0, self.proposal, self.n_steps, self.burn_in, self.sample_freq, hook, self.rng)
        print('done.')

        if return_scores:
            return Trace(networks, hook.scores)

        return networks
