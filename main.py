
import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng

theano.config.floatX = 'float32'

import lasagne

'''
Start with two hidden layer MLP.  

x -> h1 -> h2 -> s

'''
def init_params():

    params = {}

    params["h1_W"] = 0.1 * rng.normal(size = (1,256)).astype('float32')
    params["h1_b"] = 0.0 * rng.normal(size = (256,)).astype('float32')
    params["h2_W"] = 0.1 * rng.normal(size = (256,256)).astype('float32')
    params["h2_b"] = 0.0 * rng.normal(size = (256,)).astype('float32')
    params["s_W"] = 0.1 * rng.normal(size = (256,1)).astype('float32')

    return params

def params_shared(params):
    shared = {}
    for param in params:
        shared[param] = theano.shared(params[param])
    return shared

def score_network(params, x):

    h1 = T.maximum(0.0, T.dot(x, params["h1_W"]) + params['h1_b'])

    h2 = T.maximum(0.0, T.dot(h1, params['h2_W']) + params['h2_b'])

    s = T.dot(h2, params['s_W'])

    return s

if __name__ == "__main__":


    params = params_shared(init_params())

    print len(params)

    xl = T.matrix('xl')
    xq = T.matrix('xq')

    s_xl = score_network(params, xl)
    s_xq = score_network(params, xq)

    xn = xl + 1.0 * T.grad(T.sum(s_xl), xl)

    s_xn = score_network(params, xn)

    loss = T.maximum(0.0, s_xl - s_xn + 0.1) + T.maximum(0.0, s_xl - s_xq + 0.1)
    
    loss = T.mean(loss)

    updates = lasagne.updates.adam(loss, params.values(), 0.0001)

    train = theano.function([xl,xq], outputs = {'loss' : loss, 'xn' : xn, 's_xl' : s_xl, 's_xq' : s_xq}, updates = updates)    

    get_s = theano.function([xl], s_xl)

    for iteration in range(0,20000):

        xu = rng.uniform(1.9,2.0,size=(16,1)).astype('float32')

        if iteration % 100 == 0:
            print "===================================="

        for step in range(0,50):

            if iteration % 100 == 0:
                print step, xu[0]
                print "s", s_xl[0]

            xq = rng.uniform(-5.0,5.0,size=(16,1)).astype('float32')
            out = train(xu,xq)
            xu = out['xn']
            s_xl = out['s_xl']


        if iteration % 100 == 0:
            print "SHOWING S VALUES OVER RANGE"
            v = np.arange(-5,5,0.1).reshape(100,1).astype('float32')

            s = get_s(v)

            for i in range(0, v.shape[0]):
                print i, v[i], s[i]

