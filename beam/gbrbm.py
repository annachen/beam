import numpy as np
import joblib
from .rbm import RBM
from .utils import sigmoid


# TODO(anna): add sparsity constraint
# TODO(anna): add entroty loss term
# TODO(anna): add monitoring kl divergence (and reverse kl divergence)
# TODO(anna): run on the paper examples again
# TODO(anna): try unit test case? say in a 3x3 patch, only 1 pixel is on

class GaussianBernoulliRBM(RBM):
    additional_losses = [
        'sparsity',
        'h_given_v_entropy',
    ]

    def __init__(self,
                 nv, nh,
                 batch_size,
                 sigma,
                 sparsity_coef=0.,
                 h_given_v_entropy_coef=0.,
                 random_state=None):
        super(GaussianBernoulliRBM, self).__init__(
            nv, nh, batch_size, random_state=random_state)
        self.sparsity_coef = sparsity_coef
        self.h_given_v_entropy_coef = h_given_v_entropy_coef
        self.sigma = sigma

    def p_h_given_v(self, v):
        # v: (batch_size, nv)
        # output: (batch_size, nh)
        return sigmoid(self.hb[np.newaxis] + np.matmul(v, self.W) / self.sigma)

    def p_h_given_v_logits(self, v):
        return self.hb[np.newaxis] + np.matmul(v, self.W) / self.sigma

    def mean_p_v_given_h(self, h):
        # h: (batch_size, nh)
        # output: (batch_size, nv)
        return self.vb[np.newaxis] + np.matmul(h, self.W.T)

    def sample_p_v_given_h(self, h):
        # h: (batch_size, nh)
        # output: (batch_size, nv)
        center = self.vb[np.newaxis] + np.matmul(h, self.W.T)
        return self.random_state.normal(loc=center, scale=self.sigma)

    def par_nll_par_W(self, v, h):
        return np.matmul(v.T, h) / self._batch_size / self.sigma

    def par_nll_par_hb(self, h):
        return np.mean(h, axis=0)

    def par_nll_par_vb(self, v):
        return np.mean(v - self.vb, axis=0) / (self.sigma ** 2)

    def par_l1_par_W(self):
        return np.sign(self.W)

    def updates_from_sparsity(self,
                              v, p_h_given_v0, h0,
                              vn, p_h_given_vn, hn,
                              sample=False):
        return -self.par_l1_par_W(), 0, 0

    def updates_from_h_given_v_entropy(self,
                                         v, p_h_given_v0, h0,
                                         vn, p_h_given_vn, hn,
                                         sample=False):
        logits = self.p_h_given_v_logits(v)  # (batch_size, nh)
        h_term = logits * sigmoid(logits) * (1. - sigmoid(logits))
        delta_W_entropy = -np.matmul(v.T, h_term) / self.sigma
        delta_hb_entropy = -np.mean(h_term, axis=0)
        return -delta_W_entropy, 0, -delta_hb_entropy

    # TODO: make these better by making the class picklable, maybe
    def save(self, path):
        if not path.endswith('.pkl'):
            path += '.pkl'
        model = {
            'W': self.W,
            'hb': self.hb,
            'vb': self.vb,
            'params': {
                'nv': self._nv,
                'nh': self._nh,
                'batch_size': self._batch_size
            },
            'random_state': self._random_state,
        }
        joblib.dump(model, path, protocol=2)

    def load(cls, path):
        model = joblib.load(path)
        self.W = model['W']
        self.hb = model['hb']
        self.vb = model['vb']
        self._nv = model['params']['nv']
        self._nh = model['params']['nh']
        self._batch_size = model['params']['batch_size']
        self._random_state = model['random_state']


class GaussianBernoulliRBMOld(RBM):
    def __init__(self, nv, nh, batch_size,
                 sigma,
                 sparsity_coef=0.,
                 h_given_v_entropy_coef=0.,
                 seed=None):
        super(GaussianBernoulliRBM, self).__init__(nv, nh, batch_size, seed=seed)
        self.sparsity_coef = sparsity_coef
        self.h_given_v_entropy_coef = h_given_v_entropy_coef
        self.sigma = sigma

    def p_h_given_v(self, v):
        # v: (batch_size, nv)
        # output: (batch_size, nh)
        return sigmoid(self.hb[np.newaxis] + np.matmul(v, self.W) / self.sigma)

    def p_h_given_v_logits(self, v):
        return self.hb[np.newaxis] + np.matmul(v, self.W) / self.sigma

    def mean_p_v_given_h(self, h):
        # h: (batch_size, nh)
        # output: (batch_size, nv)
        return self.vb[np.newaxis] + np.matmul(h, self.W.T)

    def sample_p_v_given_h(self, h):
        # h: (batch_size, nh)
        # output: (batch_size, nv)
        center = self.vb[np.newaxis] + np.matmul(h, self.W.T)
        return self.random_state.normal(loc=center, scale=self.sigma)

    def par_nll_par_W(self, v, h):
        return np.matmul(v.T, h) / self._batch_size / self.sigma

    def par_nll_par_hb(self, h):
        return np.mean(h, axis=0)

    def par_nll_par_vb(self, v):
        return np.mean(v - self.vb, axis=0) / (self.sigma ** 2)

    def par_l1_par_W(self):
        return np.sign(self.W)

    def train_step(self, v, learning_rate, sample=False, n_gibbs=1):
        """
        v: np.array (batch_size, nv)
        learning_rate: float
        sample: boolean
          if True, sample at every v->h or h->v step. if False, use probability instead
          of sampling as advised
        n_gibbs: int
          number of up-down passes to run sampling to get model statistics

        """
        # train with gradient descent
        # compute gradient:
        # 1st component of gradient is total energy average over data distribution
        # this is computable depending on h being binary
        p_h_given_v = self.p_h_given_v(v)
        h = (self._random_state.rand(self._batch_size, self._nh) < p_h_given_v)\
            .astype(np.float32)
        if sample:
            # v: (batch_size, nv)
            # h: (batch_size, nh)
            # par_nll_par_W_data: (nv, nh) = mean(tmp, axis=0)
            par_nll_par_W_data = self.par_nll_par_W(v, h)
            par_nll_par_hb_data = self.par_nll_par_hb(h)
        else:
            par_nll_par_W_data = self.par_nll_par_W(v, p_h_given_v)
            par_nll_par_hb_data = self.par_nll_par_hb(p_h_given_v)
        par_nll_par_vb_data = self.par_nll_par_vb(v)

        # TODO: start here
        # 2nd component of gradient is total energy average over model distribution
        for _ in range(n_gibbs):
            if sample:
                v_ = self.sample_p_v_given_h(h)
            else:
                v_ = self.mean_p_v_given_h(h)

            p_h_given_v_ = self.p_h_given_v(v_)
            # for hidden layer, always sample (unless calculating updates below)
            h_ = (self._random_state.rand(self._batch_size, self._nh) < p_h_given_v_)\
                     .astype(np.float32)
            # set to h so the loop can repeat
            h = h_

        if sample:
            par_nll_par_W_model = self.par_nll_par_W(v_, h_)
            par_nll_par_hb_model = self.par_nll_par_hb(h_)
            par_nll_par_vb_model = self.par_nll_par_vb(v_)
        else:
            par_nll_par_W_model = self.par_nll_par_W(v_, p_h_given_v_)
            par_nll_par_hb_model = self.par_nll_par_hb(p_h_given_v_)
            par_nll_par_vb_model = self.par_nll_par_vb(v_)

        # now the update is just the <...>data - <...>model
        # or <...>model - <...>data, if using -= in the update
        delta_W = par_nll_par_W_data - par_nll_par_W_model
        delta_vb = par_nll_par_vb_data - par_nll_par_vb_model
        delta_hb = par_nll_par_hb_data - par_nll_par_hb_model

        # sparsity constraint
        # this seems to make W sparse, but not h
        # on the other hand, since each feature is very small and local now,
        # we need more h to be activated (?)
        delta_W_l1 = self.par_l1_par_W()

        # entropy constraint
        logits = self.p_h_given_v_logits(v)  # (batch_size, nh)
        h_term = logits * sigmoid(logits) * (1. - sigmoid(logits))
        delta_W_entropy = -np.matmul(v.T, h_term) / self.sigma

        # update it!
        self.W += learning_rate *\
                  (delta_W - self.sparsity_coef * delta_W_l1\
                           - self.h_given_v_entropy_coef * delta_W_entropy)
        self.vb += learning_rate * delta_vb
        self.hb += learning_rate * delta_hb

    def reconstruction_error(self, v):
        # v: (batch_size, nv)
        h = (self._random_state.rand(self._batch_size, self._nh) < self.p_h_given_v(v))\
            .astype(np.float32)
        v_ = self.mean_p_v_given_h(h)
        return np.mean((v - v_) ** 2)

    # TODO: make these better by making the class picklable, maybe
    def save(self, path):
        if not path.endswith('.pkl'):
            path += '.pkl'
        model = {
            'W': self.W,
            'hb': self.hb,
            'vb': self.vb,
            'params': {
                'nv': self._nv,
                'nh': self._nh,
                'batch_size': self._batch_size
            },
            'random_state': self._random_state,
        }
        joblib.dump(model, path, protocol=2)

    def load(cls, path):
        model = joblib.load(path)
        self.W = model['W']
        self.hb = model['hb']
        self.vb = model['vb']
        self._nv = model['params']['nv']
        self._nh = model['params']['nh']
        self._batch_size = model['params']['batch_size']
        self._random_state = model['random_state']
