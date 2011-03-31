from __future__ import division
import numpy as np

import data
import ab

class HMM(object):
    slots = ['t', 'e']

class Fenome(HMM):
    def __init__(self, noutputs, focal_sym = None):
        self.t = np.array([0.8, 0.1, 0.1])
        if focal_sym is None:
            pert = np.sqrt(np.random.random([2, noutputs]))
            pert = np.transpose(np.transpose(pert) / np.sum(pert, 1))
            self.e = (np.ones([2, noutputs])/(noutputs)*3+pert*1)/4
        else:
            self.e = np.ones([2, noutputs])/(2*(noutputs-1))
            self.e[:, focal_sym] = 0.5

class Silence(HMM):
    def __init__(self, noutputs):
        self.t = np.ones(12)/2
        self.e = np.ones([9, noutputs])/noutputs

def make_fenomes(data):
    fenomes = []
    nlbls = len(data.labels)
    if data.nfenomes == nlbls:
        # Exhaustive
        for i in xrange(data.nfenomes):
            fenomes.append(Fenome(nlbls, i))
    else:
        for i in xrange(data.nfenomes):
            fenomes.append(Fenome(nlbls))
    fenomes.append(Silence(nlbls))
    return fenomes

def iterate(fenomes, data):
    p, qnet, tnet, qnet_sil, tnet_sil, bnet_sil = \
            ab.train(fenomes, data.baseforms, data.observations)

    for i in xrange(len(fenomes)-1):
        fenomes[i].e = qnet[2*i:(2*i+1), :]
        fenomes[i].t = tnet[i, :]
    fenomes[-1].e = qnet_sil
    fenomes[-1].t[0:9] = tnet_sil
    fenomes[-1].t[9:12] = bnet_sil

    # Flatten the silence model slightly
    a = 1e-20   
    fenomes[i].e = (fenomes[i].e + 
            np.ones(fenomes[i].e.shape)/np.prod(fenomes[i].e.shape))/(1+a)
    fenomes[i].t = (fenomes[i].t + 
            np.ones(fenomes[i].t.shape)/np.prod(fenomes[i].t.shape))/(1+a)

    return p, fenomes
