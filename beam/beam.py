import numpy as np
import joblib
import sklearn.neighbors


class BEAM(object):
    def __init__(self, nv, nh, batch_size, seed=None):
        self._nv = nv
        self._nh = nh
        self._batch_size = batch_size
        self._random_state = np.random.RandomState(seed=seed)

    def initialize(self, scale=0.01):
        self.W = self._random_state.normal(0, scale, (self._nv, self._nh))
        self.vb = np.zeros((self._nv))
        self.hb = np.zeros((self._nh))

        # for adversarial training
        self._memory = None

    def p_h_given_v(self, v):
        # technically, p(h_j = 1 | v) for all j
        # p(h_j = 1 | v) = 1 / (1 + exp(b_j + sum_i(v_i * W_ij)))
        # v: (batch_size, nv)
        # output: (batch_size, nh)
        return 1. / (1. + np.exp(self.hb[np.newaxis] + np.matmul(v, self.W)))

    def p_v_given_h(self, h):
        # technically, p(v_i = 1 | h) for all i
        # p(v_i = 1 | h) = 1 / (1 + exp(a_i + sum_j(h_j * W_ij)))
        # h: (batch_size, nh)
        # output: (batch_size, nv)
        return 1. / (1. + np.exp(self.vb[np.newaxis] + np.matmul(h, self.W.T)))

    def train_step(self,
                   v,
                   adversary_weight,
                   k,
                   learning_rate,
                   sample=False,
                   n_gibbs=1):
        """
        v: np.array (batch_size, nv)
        adversary_weight: float
        k: int
          number of nearest neighbors to look at
        learning_rate: float
        sample: boolean
          if True, sample at every v->h or h->v step. if False, use probability instead
          of sampling as advised
        n_gibbs: int
          number of up-down passes to run sampling to get model statistics

        """
        # this part is copied from rbm.py
        p_h_given_v = self.p_h_given_v(v)
        h = (self._random_state.rand(self._batch_size, self._nh) < p_h_given_v)\
            .astype(np.float32)
        h_data = h
        if sample:
            par_E_par_W_data = -np.matmul(v.T, h) / self._batch_size
            par_E_par_hb_data = -np.mean(h, axis=0)
        else:
            par_E_par_W_data = -np.matmul(v.T, p_h_given_v) / self._batch_size
            par_E_par_hb_data = -np.mean(p_h_given_v, axis=0)
        par_E_par_vb_data = -np.mean(v, axis=0)

        for _ in range(n_gibbs):
            p_v_given_h = self.p_v_given_h(h)
            if sample:
                v_ = (self._random_state.rand(self._batch_size, self._nv) < p_v_given_h)\
                     .astype(np.float32)
            else:
                v_ = p_v_given_h
            p_h_given_v_ = self.p_h_given_v(v_)
            if sample:
                h_ = (self._random_state.rand(self._batch_size, self._nh) < p_h_given_v_)\
                     .astype(np.float32)
            else:
                h_ = p_h_given_v_
            h = h_

        # modify this to retain the value from each sample in batch
        # TODO(anna): optimize this
        par_E_par_W_model = []
        par_E_par_hb_model = []
        par_E_par_vb_model = []
        for vt, ht in zip(v_, h_):
            par_E_par_W_model.append(-np.matmul(vt[..., np.newaxis], ht[np.newaxis]))
            par_E_par_hb_model.append(-h_)
            par_E_par_vb_model.append(-v_)
        par_E_par_W_model = np.array(par_E_par_W_model)  # (batch_size, nv, nh)
        par_E_par_hb_model = np.array(par_E_par_hb_model)  # (batch_size, nh)
        par_E_par_vb_model = np.array(par_E_par_vb_model)  # (batch_size, nv)
        #par_E_par_W_model = -np.matmul(v_.T, h_) / self._batch_size
        #par_E_par_hb_model = -np.mean(h_, axis=0)
        #par_E_par_vb_model = -np.mean(v_, axis=0)

        par_L_par_W = -par_E_par_W_data + np.mean(par_E_par_W_model, axis=0)
        par_L_par_vb = -par_E_par_vb_data + np.mean(par_E_par_vb_model, axis=0)
        par_L_par_hb = -par_E_par_hb_data + np.mean(par_E_par_hb_model, axis=0)

        # for the first iteration, there's no previous particles
        # initialize memory and skip calculating adverserial loss
        if self._memory is None:
            # I need to put in h_data and h_
            self._memory = sklearn.neighbors.KDTree(
                np.concatenate([h_data, h_], axis=0),
                leaf_size=self.batch_size / 10)
            # note that h_data formed the first half
            par_A_par_W = np.zeros_like(self.W)
            par_A_par_vb = np.zeros_like(self.vb)
            par_A_par_hb = np.zeros_like(self.hb)

        else:
            # new part: adversary loss
            # par_A_par_theta = Cov_theta[T(h), -par_E_par_theta]
            # need to calculate covariance with the model distribution
            # Cov[X, Y] = E[XY] - E[X]E[Y]

            # let's compute T(h) drawing from model distribution
            # so at this point h_ is the minibatch that's drawn from the model distribution
            # so I'd calcuate XY, X, Y and average over the minibatch
            # so E[X] = mean(T(h))
            critic_values = self.knn_critic(h_, k) # (batch_size,)
            mean_critic = np.mean(critic_values)
            # then E[Y] = mean(-par_E_par_theta)
            # hey, i have these calculated up there (without the - sign)
            # now E[XY]
            prod_W = np.mean(
                critic_values[..., np.newaxis, np.newaxis] * (-par_E_par_W_model),
                axis=0)
            prod_vb = np.mean(
                critic_values[..., np.newaxis] * (-par_E_par_vb_model),
                axis=0)
            prod_hb = np.mean(
                critic_values[..., np.newaxis] * (-par_E_par_hb_model),
                axis=0)

            # put together to get cov
            par_A_par_W = prod_W - mean_critic * np.mean(-par_E_par_W_model, axis=0)
            par_A_par_vb = prod_vb - mean_critic * np.mean(-par_E_par_vb_model, axis=0)
            par_A_par_hb = prod_hb - mean_critic * np.mean(-par_E_par_hb_model, axis=0)

        # update them
        def _combine_with_weight(x, y):
            return (1. - adversary_weight) * x + adversary_weight * y

        delta_W = _combine_with_weight(par_L_par_W, par_A_par_W)
        delta_vb = _combine_with_weight(par_L_par_vb, par_A_par_vb)
        delta_hb = _combine_with_weight(par_L_par_hb, par_A_par_hb)

        self.W -= learning_rate * delta_W
        self.vb -= learning_rate * delta_vb
        self.hb -= learning_rate * delta_hb

    def knn_critic(h, k):
        # h: (batch_size, nh)
        # self._memory: points from previous iteration
        _, ind = self._memory.query(h, k=k)  # ind: (batch_size, k)
        return np.sum(ind < self.batch_size, axis=1) / self.batch_size - 1.

    def reconstruction_error(self, v):
        # v: (batch_size, nv)
        h = (self._random_state.rand(self._batch_size, self._nh) < self.p_h_given_v(v))\
            .astype(np.float32)
        v_ = self.p_v_given_h(h)
        return np.mean((v - v_) ** 2)

    def save(self, path):
        if not path.endswith('.pkl'):
            path += '.pkl'
        model = {
            'W': self.W,
            'hb': self.hb,
            'vb': self.vb
        }
        joblib.dump(path, model, protocol=2)

    def load(self, path):
        model = joblib.load(path)
        self.W = model['W']
        self.hb = model['hb']
        self.vb = model['vb']
