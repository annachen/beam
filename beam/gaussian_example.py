__doc__ = """Try the 2D Gaussian examples in the paper."""

import numpy as np

from beam.rbm import RBM
from beam.beam import BEAM


rbm_params = {
    'nv': 2,
    'nh': 50,
    'batch_size': 1000,
}

rbm_training_params = {
    'learning_rate': 1e-3,
}

beam_params = {
    'nv': 2,
    'nh': 50,
    'batch_size': 1000,
}

beam_training_params = {
    'learning_rate': 1e-3,
    'adversary_weight': 0.5,
    'k': 5
}


class Gaussian2D25(object):
    def __init__(self, seed=0):
        self.random_state = np.random.RandomState(seed=0)
        self.n_gaussians = 25
        ys, xs = np.meshgrid(range(5), range(5), indexing='ij')
        self.means = np.stack([ys, xs]).reshape((2, 25)).T - 2. # (25, 2)
        self.scale = 0.2

    def sample(self, n):
        gaussian_ids = np.floor(
            self.random_state.rand(n) / (1. / self.n_gaussians)).astype(np.int32)
        samples = self.random_state.normal(
            self.means[gaussian_ids].flatten(), self.scale).reshape((n, 2))
        return samples


def get_network(typ):
    if typ == 'rbm':
        # prepare RBM
        net = RBM(rbm_params['nv'],
                  rbm_params['nh'],
                  rbm_params['batch_size'],
                  seed=0)
    elif typ == 'beam':
        # prepare BEAM
        net = BEAM(beam_params['nv'],
                   beam_params['nh'],
                   beam_params['batch_size'],
                   seed=0)
    return net


def init_network(typ):
    net = get_network(typ)
    net.initialize()
    return net


def train(net, sampler, train_params, n_steps=1000):
    errs = []
    for _ in range(n_steps):
        batch = sampler.sample(net._batch_size)
        net.train_step(batch, **train_params, sample=True)
        errs.append(net.reconstruction_error(batch))
    return errs
