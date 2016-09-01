import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from collections import OrderedDict

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def random_weights(shape, name=None):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01), name=name)


def zeros(shape, name=""):
    return theano.shared(floatX(np.zeros(shape)), name=name)


def softmax(X, temperature=1.0):
    e_x = T.exp((X - X.max(axis=1).dimshuffle(0, 'x'))/temperature)
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def sigmoid(X):
    return 1 / (1 + T.exp(-X))


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
        return X


def rectify(X):
    return T.maximum(X, 0.)


def clip(X, epsilon):

    return T.maximum(T.minimum(X, epsilon), -1*epsilon)

def scale(X, max_norm):
    curr_norm = T.sum(T.abs_(X))
    return ifelse(T.lt(curr_norm, max_norm), X, max_norm * (X/curr_norm))



def get_params(layers):
    params = []
    for layer in layers:
        params += layer.get_params()
    return params


def make_caches(params):
    caches = []
    for p in params:
        caches.append(theano.shared(floatX(np.zeros(p.get_value().shape))))

    return caches

##Training code
def SGD (cost, params, eta, lambda2 = 0.0):
    updates = []
    grads = T.grad(cost=cost, wrt=params)

    for p,g in zip(params, grads):
        updates.append([p, p - eta *( g + lambda2*p)])

    return updates

def momentum(cost, params, caches, eta, rho=.1, clip_at=0.0, scale_norm=0.0, lambda2=0.0):
    updates = []
    grads = T.grad(cost=cost, wrt=params)

    for p, c, g in zip(params, caches, grads):
        if clip_at > 0.0:
            grad = clip(g, clip_at)
        else:
            grad = g

        if scale_norm > 0.0:
            grad = scale(g, scale_norm)

        delta = rho * g + (1-rho) * c
        updates.append([c, delta])
        updates.append([p, p - eta * ( delta + lambda2 * p)])

    return updates


def create_optimization_updates(cost, params, updates=None, max_norm=5.0,
                                lr=0.01, eps=1e-6, rho=0.95,
                                method = "adadelta", gradients = None):
    """
    Get the updates for a gradient descent optimizer using
    SGD, AdaDelta, or AdaGrad.
    Returns the shared variables for the gradient caches,
    and the updates dictionary for compilation by a
    theano function.
    Inputs
    ------
    cost     theano variable : what to minimize
    params   list            : list of theano variables
                               with respect to which
                               the gradient is taken.
    max_norm float           : cap on excess gradients
    lr       float           : base learning rate for
                               adagrad and SGD
    eps      float           : numerical stability value
                               to not divide by zero
                               sometimes
    rho      float           : adadelta hyperparameter.
    method   str             : 'adagrad', 'adadelta', or 'sgd'.
    Outputs:
    --------
    updates  OrderedDict   : the updates to pass to a
                             theano function
    gsums    list          : gradient caches for Adagrad
                             and Adadelta
    xsums    list          : gradient caches for AdaDelta only
    lr       theano shared : learning rate
    max_norm theano_shared : normalizing clipping value for
                             excessive gradients (exploding).
    """
    lr = theano.shared(np.float64(lr).astype(theano.config.floatX))
    eps = np.float64(eps).astype(theano.config.floatX)
    rho = theano.shared(np.float64(rho).astype(theano.config.floatX))
    if max_norm is not None and max_norm is not False:
        max_norm = theano.shared(np.float64(max_norm).astype(theano.config.floatX))

    gsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) if (method == 'adadelta' or method == 'adagrad') else None for param in params]
    xsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) if method == 'adadelta' else None for param in params]

    gparams = T.grad(cost, params) if gradients is None else gradients

    if updates is None:
        updates = OrderedDict()

    for gparam, param, gsum, xsum in zip(gparams, params, gsums, xsums):
        # clip gradients if they get too big
        if max_norm is not None and max_norm is not False:
            grad_norm = gparam.norm(L=2)
            gparam = (T.minimum(max_norm, grad_norm)/ (grad_norm + eps)) * gparam

        if method == 'adadelta':
            updates[gsum] = T.cast(rho * gsum + (1. - rho) * (gparam **2), theano.config.floatX)
            dparam = -T.sqrt((xsum + eps) / (updates[gsum] + eps)) * gparam
            updates[xsum] = T.cast(rho * xsum + (1. - rho) * (dparam **2), theano.config.floatX)
            updates[param] = T.cast(param + dparam, theano.config.floatX)
        elif method == 'adagrad':
            updates[gsum] =  T.cast(gsum + (gparam ** 2), theano.config.floatX)
            updates[param] =  T.cast(param - lr * (gparam / (T.sqrt(updates[gsum] + eps))), theano.config.floatX)
        else:
            updates[param] = param - gparam * lr

    if method == 'adadelta':
        lr = rho

    return updates, gsums, xsums, lr, max_norm
