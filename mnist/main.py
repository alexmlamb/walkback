
import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng

theano.config.floatX = 'float32'

import lasagne

import matplotlib.pyplot as plt

from viz import plot_images

class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()

import cPickle as pickle
import gzip
from random import randint
data = pickle.load(gzip.open("mnist.pkl.gz", "r"))

trx = data[0][0]


#output is 16 x 784
def sample_data():
    return trx[0:16]





'''
Start with two hidden layer MLP.  

x -> h1 -> h2 -> s

'''
def init_params():

    params = {}

    params["h1_W"] = 0.1 * rng.normal(size = (28*28,256)).astype('float32')
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

    xn = xl + 0.1 * T.grad(T.sum(s_xl), xl)

    xb = xl - 0.1 * T.grad(T.sum(s_xl), xl)

    s_xn = score_network(params, xn)

    loss = T.maximum(0.0, s_xl - s_xn + 0.1) + T.maximum(0.0, s_xl - s_xq + 0.1)
    
    loss = T.mean(loss)

    updates = lasagne.updates.adam(loss, params.values(), 0.0001)

    train = theano.function([xl,xq], outputs = {'loss' : loss, 'xn' : xn, 's_xl' : s_xl, 's_xq' : s_xq}, updates = updates)    

    get_s = theano.function([xl], s_xl)

    reverse_step = theano.function([xl], [xb, s_xl])

    num_steps = 150

    for iteration in range(0,200000):

        xu = sample_data()

        if iteration % 100 == 0:
            print "===================================="

        for step in range(0,num_steps):

            xq = rng.uniform(-1.5,1.5,size=(16,28*28)).astype('float32')
            out = train(xu,xq)
            xu = out['xn']
            s_xl = out['s_xl']

            if iteration % 100 == 0:
                print step, "s", s_xl[0]
                if step == num_steps-1:
                    print "s for prior", get_s(xq)
                    print "xu shape", xu.shape
                plot_images(xu.reshape(16,1,28,28), "plots/imagewalk_step_" + str(step))


        if iteration % 2000 == 0:
            print "DOING GENERATION"
            for step in range(0, num_steps):
                res = reverse_step(xu)
                print step, "score", res[1][0]
                xu = res[0]

                plot_images(xu.reshape(16,1,28,28), "plots/gen/step_" + str(step))
