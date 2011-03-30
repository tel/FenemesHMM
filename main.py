from __future__ import division
import numpy as np

import data

class HMM(object):
    slots = ['t', 'e']

class Fenome(HMM):
    def __init__(self, noutputs, focal_sym = None):
        self.t = np.array([0.8, 0.1, 0.1])
        if focal_sym is None:
            pert = np.sqrt(np.random.random([2, noutputs]))
            pert = np.transpose(np.transpose(pert) / np.sum(pert, 1))
            self.e = (np.ones([2, noutputs])/(noutputs)+pert)/2
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

d = data.Data(alphabetic_baseforms = True)
fenomes = make_fenomes(d)
