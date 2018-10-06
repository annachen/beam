import numpy as np
import torch
from torchvision.datasets import mnist
from torchvision import transforms

from rbm import RBM


MNIST_ROOT = '~/data/mnist'

rbm_params = {
    'nv': 28 * 28,
    'nh': 50,
    'batch_size': 1,
}

def train_mnist(batch_size=1, learning_rate=0.1):
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

    # prepare RBM
    rbm = RBM(rbm_params['nv'], 
              rbm_params['nh'],
              rbm_params['batch_size'])

    for batch in train_iter:
        # for now only works with batch_size=1
        imgs, _ = batch
        # I guess in pytorch it's (N, C, H, W)
        # just gonna convert to numpy now since i'm not really using pytorch
        imgs = np.asarray(imgs)

        # make it flat
        imgs = imgs.reshape((imgs.shape[0], -1))
       
        # for now pass it into RBM one by one, since it doesn't support batch yet
        for img in imgs:
            rbm.train_step(im, learning_rate)


