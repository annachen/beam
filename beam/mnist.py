import numpy as np
import torch
from torchvision.datasets import mnist
from torchvision import transforms

from rbm import RBM
from beam import BEAM


MNIST_ROOT = '~/data/mnist'

rbm_params = {
    'nv': 28 * 28,
    'nh': 50,
    'batch_size': 20,
}

rbm_training_params = {
    'learning_rate': 1e-3,
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
    return net


def train(typ):
    net = get_network(typ)
    net.initialize()

    train_iter, test_iter = get_iterators(net._batch_size)

    train_params = eval('{}_train_params'.format(typ))

    errs = []
    for batch in train_iter:
        # for now only works with batch_size=1
        imgs, _ = batch
        # I guess in pytorch it's (N, C, H, W)
        # just gonna convert to numpy now since i'm not really using pytorch
        imgs = np.asarray(imgs)

        # make it flat
        imgs = imgs.reshape((imgs.shape[0], -1))
        net.train_step(imgs, *train_params, sample=True)
        errs.append(net.reconstruction_error(imgs))
