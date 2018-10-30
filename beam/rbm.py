import numpy as np
import joblib


# TODO(anna): add sparsity constraint
# TODO(anna): add entroty loss term
# TODO(anna): add monitoring kl divergence (and reverse kl divergence)
# TODO(anna): try unit test case? say in a 3x3 patch, only 1 pixel is on

class RBM(object):
    def __init__(self, nv, nh, batch_size, seed=None):
        self._nv = nv
        self._nh = nh
        self._batch_size = batch_size
        self._random_state = np.random.RandomState(seed=seed)

    def initialize(self, scale=0.01):
        # TODO: extent to handle batch input
        # NOTE(anna): very sensitive to initial W range; smaller seems better
        self.W = self._random_state.normal(0, scale, (self._nv, self._nh))
        # self.W = (self._random_state.rand(self._nv, self._nh) - 0.5) * 0.1
        self.vb = np.zeros((self._nv))
        self.hb = np.zeros((self._nh))

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
            # wouldn't this make deep network hard to train? Seems lots of terms get
            # set to 0
            # v: (batch_size, nv)
            # h: (batch_size, nh)
            # tmp: (batch_size, nv, nh)
            # par_E_par_W_data: (nv, nh) = mean(tmp, axis=0)
            # TODO(anna): use einsum to optimize?
            par_E_par_W_data = -np.matmul(v.T, h) / self._batch_size
            par_E_par_hb_data = -np.mean(h, axis=0)
            #par_E_par_W_data = -np.matmul(v[..., np.newaxis], h[np.newaxis])
            #par_E_par_hb_data = -h
        else:
            # NOTE(anna): haven't successfully run sample=False. All weights seem
            # to be converging to the same pattern in this case.
            par_E_par_W_data = -np.matmul(v.T, p_h_given_v) / self._batch_size
            par_E_par_hb_data = -np.mean(p_h_given_v, axis=0)
            #par_E_par_W_data = -np.matmul(v[..., np.newaxis], p_h_given_v[np.newaxis])
            #par_E_par_hb_data = -p_h_given_v
        par_E_par_vb_data = -np.mean(v, axis=0)
        #par_E_par_vb_data = -v

        # 2nd component of gradient is total energy average over model distribution
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

        if sample:
            par_E_par_W_model = -np.matmul(v_.T, h_) / self._batch_size
            par_E_par_hb_model = -np.mean(h_, axis=0)
            par_E_par_vb_model = -np.mean(v_, axis=0)
            #par_E_par_W_model = -np.matmul(v_[..., np.newaxis], h_[np.newaxis])
            #par_E_par_hb_model = -h_
            #par_E_par_vb_model = -v_
        else:
            par_E_par_W_model = -np.matmul(v_.T, p_h_given_v_) / self._batch_size
            par_E_par_hb_model = -np.mean(p_h_given_v_, axis=0)
            par_E_par_vb_model = -np.mean(p_v_given_h, axis=0)
            #par_E_par_W_model = -np.matmul(v_[..., np.newaxis], p_h_given_v_[np.newaxis])
            #par_E_par_hb_model = -p_h_given_v_
            #par_E_par_vb_model = -p_v_given_h

        # now the update is just the <-...>data - <-...>model
        delta_W = -par_E_par_W_data + par_E_par_W_model
        delta_vb = -par_E_par_vb_data + par_E_par_vb_model
        delta_hb = -par_E_par_hb_data + par_E_par_hb_model

        # update it!
        # NOTE: range of W
        # theoretically no real bound; make the probability closer to 0 or 1
        # turns out I was runing gradient ascent, not gradient descent :P
        # after learning run for quite a while, the weights start to look interesting
        self.W -= learning_rate * delta_W
        self.vb -= learning_rate * delta_vb
        self.hb -= learning_rate * delta_hb

    def reconstruction_error(self, v):
        # v: (batch_size, nv)
        h = (self._random_state.rand(self._batch_size, self._nh) < self.p_h_given_v(v))\
            .astype(np.float32)
        v_ = self.p_v_given_h(h)
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
