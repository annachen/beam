import numpy as np
import joblib
from .utils import sigmoid


# TODO(anna): add sparsity constraint
# TODO(anna): add entroty loss term
# TODO(anna): add monitoring kl divergence (and reverse kl divergence)
# TODO(anna): try unit test case? say in a 3x3 patch, only 1 pixel is on

class RBM(object):
    additional_losses = []

    def __init__(self, nv, nh, random_state=None):
        self._nv = nv
        self._nh = nh
        self._random_state = random_state or np.random.RandomState()

    def initialize(self, scale=0.01):
        # NOTE(anna): very sensitive to initial W range; smaller seems better
        self.W = self._random_state.normal(0, scale, (self._nv, self._nh))
        self.vb = np.zeros((self._nv))
        self.hb = np.zeros((self._nh))

    def p_h_given_v_logits(self, v):
        raise NotImplementedError()

    def p_h_given_v(self, v):
        raise NotImplementedError()

    def p_v_given_h(self, h):
        raise NotImplementedError()

    def sample_p_v_given_h(self, h):
        raise NotImplementedError()

    def mean_p_v_given_h(self, h):
        raise NotImplementedError()

    def par_nll_par_W(self, v, h):
        raise NotImplementedError()

    def par_nll_par_hb(self, h):
        raise NotImplementedError()

    def par_nll_par_vb(self, v):
        raise NotImplementedError()

    def train_step(self, v, learning_rate, sample=False, n_gibbs=1):
        # train with gradient descent
        # first run a few steps of the chain
        p_h_given_v0, h0, vn, p_h_given_vn, hn = self.run_chain(
            v, sample=sample, n_gibbs=n_gibbs)

        delta_W, delta_vb, delta_hb = self.updates_from_nll(
            v, p_h_given_v0, h0, vn, p_h_given_vn, hn, sample=sample)

        for loss in self.additional_losses:
            updates_fn = eval('self.updates_from_{}'.format(loss))
            delta_W_tmp, delta_vb_tmp, delta_hb_tmp = updates_fn(
                v, p_h_given_v0, h0, vn, p_h_given_vn, hn)
            loss_weight = eval('self.{}_coef'.format(loss))
            delta_W += loss_weight * delta_W_tmp
            delta_vb += loss_weight * delta_vb_tmp
            delta_hb += loss_weight * delta_hb_tmp

        # finally, update for real
        self.W += learning_rate * delta_W
        self.vb += learning_rate * delta_vb
        self.hb += learning_rate * delta_hb

    def run_chain(self, v, sample=False, n_gibbs=1):
        batch_size = len(v)
        # first step, v -> h
        p_h_given_v0 = self.p_h_given_v(v)
        h0 = (self._random_state.rand(batch_size, self._nh) < p_h_given_v0)\
             .astype(np.float32)

        # now, do n_gibbs times of h -> v, v -> h
        hn = np.copy(h0)
        for _ in range(n_gibbs):
            if sample:
                vn = self.sample_p_v_given_h(hn)
            else:
                vn = self.mean_p_v_given_h(hn)

            p_h_given_vn = self.p_h_given_v(vn)
            # for hidden layer, always sample (unless calculating updates)
            hn = (self._random_state.rand(batch_size, self._nh) < p_h_given_vn)\
                 .astype(np.float32)
        return p_h_given_v0, h0, vn, p_h_given_vn, hn

    def updates_from_nll(self, v, p_h_given_v0, h0, vn, p_h_given_vn, hn, sample=False):
        # calculate data stats based on v and h0 (p_h_given_v0)
        if sample:
            # v: (batch_size, nv)
            # h: (batch_size, nh)
            # par_nll_par_W_data: (nv, nh)
            par_nll_par_W_data = self.par_nll_par_W(v, h0)
            par_nll_par_hb_data = self.par_nll_par_hb(h0)
        else:
            par_nll_par_W_data = self.par_nll_par_W(v, p_h_given_v0)
            par_nll_par_hb_data = self.par_nll_par_hb(p_h_given_v0)
        par_nll_par_vb_data = self.par_nll_par_vb(v)

        # calculate model stats based on vn and hn (p_h_given_vn)
        if sample:
            par_nll_par_W_model = self.par_nll_par_W(vn, hn)
            par_nll_par_hb_model = self.par_nll_par_hb(hn)
        else:
            par_nll_par_W_model = self.par_nll_par_W(vn, p_h_given_vn)
            par_nll_par_hb_model = self.par_nll_par_hb(p_h_given_vn)
        par_nll_par_vb_model = self.par_nll_par_vb(vn)
        return (par_nll_par_W_data - par_nll_par_W_model,
                par_nll_par_vb_data - par_nll_par_vb_model,
                par_nll_par_hb_data - par_nll_par_hb_model)

    def reconstruction_error(self, v):
        batch_size = len(v)
        h = (self._random_state.rand(batch_size, self._nh) < self.p_h_given_v(v))\
            .astype(np.float32)
        v_ = self.mean_p_v_given_h(h)
        return np.mean((v - v_) ** 2)

    def save(self, filename):
        joblib.dump(self, filename)

    @staticmethod
    def load(filename):
        return joblib.load(filename)


class BernoulliRBM(RBM):
    typ = 'bernoulli'

    def p_h_given_v_logits(self, v):
        return self.hb[np.newaxis] + np.matmul(v, self.W)

    def p_h_given_v(self, v):
        return sigmoid(self.hb[np.newaxis] + np.matmul(v, self.W))

    def p_v_given_h(self, h):
        return sigmoid(self.vb[np.newaxis] + np.matmul(h, self.W.T))

    def sample_p_v_given_h(self, h):
        batch_size = len(h)
        p_v_given_h = self.p_v_given_h(h)
        return (self._random_state.rand(batch_size, self._nv) < p_v_given_h)\
               .astype(np.float32)

    def mean_p_v_given_h(self, h):
        return self.p_v_given_h(h)

    def par_nll_par_W(self, v, h):
        batch_size = len(v)
        return np.matmul(v.T, h) / batch_size

    def par_nll_par_hb(self, h):
        return np.mean(h, axis=0)

    def par_nll_par_vb(self, v):
        return np.mean(v, axis=0)

    # TODO: add saving and loading
    # TODO: add reconstruction error
    # TODO: add KL divergence based on particles
    # TODO: test this implementation



# old implementation of BernoulliRBM.
class BernoulliRBMOld(RBM):
    def p_h_given_v(self, v):
        # technically, p(h_j = 1 | v) for all j
        # p(h_j = 1 | v) = 1 / (1 + exp(b_j + sum_i(v_i * W_ij)))
        # v: (batch_size, nv)
        # output: (batch_size, nh)
        # NOTE: there's a sign change making this a sigmoid function
        return sigmoid(self.hb[np.newaxis] + np.matmul(v, self.W))

    def p_v_given_h(self, h):
        # technically, p(v_i = 1 | h) for all i
        # p(v_i = 1 | h) = 1 / (1 + exp(a_i + sum_j(h_j * W_ij)))
        # h: (batch_size, nh)
        # output: (batch_size, nv)
        # NOTE: there's a sign change making this a sigmoid function
        return sigmoid(self.vb[np.newaxis] + np.matmul(h, self.W.T))

    def par_nll_par_W(self, v, h):
        return np.matmul(v.T, h) / self._batch_size

    def par_nll_par_hb(self, h):
        return np.mean(h, axis=0)

    def par_nll_par_vb(self, v):
        return np.mean(v, axis=0)

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
