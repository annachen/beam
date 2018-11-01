import numpy as np
import torch
from torchvision.datasets import mnist
from torchvision import transforms

from .rbm import RBM
from .beam import BEAM
from .gbrbm import GaussianBernoulliRBM


MNIST_ROOT = '~/data/mnist'

rbm_params = {
    'nv': 28 * 28,
    'nh': 50,
    'batch_size': 20,
}

rbm_training_params = {
    'learning_rate': 1e-3,
}

gbrbm_params = {
    'nv': 28 * 28,
    'nh': 50,
    'batch_size': 20,
    'sigma': 0.2,
    'sparsity_coef': 0.1,
    'h_given_v_entropy_coef': 0.01,
}

gbrbm_training_params = {
    'learning_rate': 1e-3
}

beam_params = {
    'nv': 28 * 28,
    'nh': 50,
    'batch_size': 20,
}

beam_training_params = {
    'learning_rate': 1e-3,
    'adversary_weight': 0.5,
    'k_neighbors': 5
}


def get_iterators(batch_size):
    # prepare data
    trainset = mnist.MNIST(
        root=MNIST_ROOT,
        train=True,
        download=True,
        transform=transforms.ToTensor())
    testset = mnist.MNIST(
        root=MNIST_ROOT,
        train=False,
        transform=transforms.ToTensor())

    train_iter = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_iter = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_iter, test_iter


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
    elif typ == 'gbrbm':
        net = GaussianBernoulliRBM(gbrbm_params['nv'],
                                   gbrbm_params['nh'],
                                   gbrbm_params['batch_size'],
                                   gbrbm_params['sigma'],
                                   sparsity_coef=gbrbm_params['sparsity_coef'],
                                   h_given_v_entropy_coef=gbrbm_params['h_given_v_entropy_coef'],
                                   seed=0)
    return net


def init_network(typ):
    net = get_network(typ)
    net.initialize()
    return net


def train(net, iterator, train_params, n_steps=1000):
    errs = []
    for batch in iterator:
        # I guess in pytorch it's (N, C, H, W)
        # just gonna convert to numpy now since i'm not really using pytorch
        imgs, _ = batch
        imgs = np.asarray(imgs).reshape((imgs.shape[0], -1))

        # run training
        net.train_step(imgs, **train_params, sample=True)
        errs.append(net.reconstruction_error(imgs))
        if len(errs) == n_steps:
            break
    return errs
