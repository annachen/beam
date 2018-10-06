import numpy as np


class RBM(object):
    def __init__(self, nv, nh, batch_size, seed=None):
        self._nv = nv
        self._nh = nh
        self._batch_size = batch_size
        self._random_state = np.random.RandomState(seed=seed)

    def initialize(self):
        # TODO: extent to handle batch input
        self.W = (self._random_state.rand(self._nv, self._nh ) - 0.5) * 0.1
        self.vb = np.zeros((self._nv))
        self.hb = np.zeros((self._nh))
    
    def p_h_given_v(self, v):
        # technically, p(h_j = 1 | v) for all j
        # p(h_j = 1 | v) = 1 / (1 + exp(b_j + sum_i(v_i * W_ij)))
        return 1. / (1. + np.exp(self.hb + np.matmul(v, self.W)))

    def p_v_given_h(self, h):
        # technically, p(v_i = 1 | h) for all i
        # p(v_i = 1 | h) = 1 / (1 + exp(a_i + sum_j(h_j * W_ij)))
        return 1. / (1. + np.exp(self.vb + np.matmul(self.W, h)))
    
    def train_step(self, v, learning_rate):
        #print('W', self.W)
        # train with gradient descent
        # compute gradient:
        # 1st component of gradient is total energy average over data distribution
        # this is computable depending on h being binary
        # do i need to sample here? 
        #print('p_h_given_v', self.p_h_given_v(v))
        p_h_given_v = self.p_h_given_v(v)
        h = (self._random_state.rand(self._nh) < p_h_given_v).astype(np.float32)
        # or can i just use the probability anyway?
        #partial_E_partial_W_data = -np.matmul(v[..., np.newaxis], p_h_given_v[np.newaxis])
        partial_E_partial_W_data = -np.matmul(v[..., np.newaxis], h[np.newaxis])
        #print('pEpW_data', np.any(partial_E_partial_W_data != 0))
        # wouldn't this make deep network hard to train?
        # basically setting the update = 0?
        partial_E_partial_vb_data = -v
        #partial_E_partial_hb_data = -p_h_given_v
        partial_E_partial_hb_data = -h

        # 2nd component of gradient is total energy average over model distribution
        # well then this should be straight forward too; why do we need gibbs sampling?
        # what is gibbs sampling anyway?
        # isn't it just conditioning on almost everything and sample one thing at the time
        # to deal with the independency?
        #print('p_v_given_h', self.p_v_given_h(h))
        v_ = (self._random_state.rand(self._nv) < self.p_v_given_h(h)).astype(np.float32)
        # v_ = self.p_v_given_h(h)
        #print('p_h_given_v', self.p_h_given_v(v_))
        p_h_given_v_ = self.p_h_given_v(v_)
        h_ = (self._random_state.rand(self._nh) < p_h_given_v_).astype(np.float32)
        #partial_E_partial_W_model = -np.matmul(v_[..., np.newaxis], p_h_given_v_[np.newaxis])
        partial_E_partial_W_model = -np.matmul(v_[..., np.newaxis], h_[np.newaxis])
        #print('pEpW_model', np.any(partial_E_partial_W_model != 0))
        partial_E_partial_vb_model = -v_
        #partial_E_partial_hb_model = -p_h_given_v_
        partial_E_partial_hb_model = -h_

        # now the update is just the <-...>data - <-...>model
        delta_W = -partial_E_partial_W_data + partial_E_partial_W_model
        delta_vb = -partial_E_partial_vb_data + partial_E_partial_vb_model
        delta_hb = -partial_E_partial_hb_data + partial_E_partial_hb_model
        #print('delta_W', delta_W, delta_W.min(), delta_W.max())
        #print('vb', delta_vb)
        #print('hb', delta_hb)

        # update it!
        self.W += learning_rate * delta_W
        self.vb += learning_rate * delta_vb
        self.hb += learning_rate * delta_hb
