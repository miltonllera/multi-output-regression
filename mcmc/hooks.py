from structure.graphs import plot_digraph
from mcmc.state_space import check_consistency


class IterationHook:
    def __call__(self, i, s, score, p, r):
        raise NotImplementedError


class SamplingTrace(IterationHook):
    def __init__(self, save_iter=1, discard=0):
        self.scores = []
        self.accept_ratios = []
        self.save_iter = save_iter
        self.discard = discard
        self.accepted = 0

    def __call__(self, i, s, score, p, r):
        i += 1
        if i > self.discard and (i + 1) % self.save_iter == 0:
            self.scores.append(score)
            if p < r:
                self.accepted += 1
                self.accept_ratios.append(self.accepted / (i + 1))


class StatePlotter(IterationHook):
    def __call__(self, i, s, score, p, r):
        plot_digraph(s)


class DebuggerHook(SamplingTrace, StatePlotter):
    def __call__(self, i, s, score, p, r):
        SamplingTrace.__call__(self, i, s, score, p, r)
        StatePlotter.__call__(self, i, s, score, p, r)
        check_consistency(s)