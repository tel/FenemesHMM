#!/usr/bin/env python
import numpy as np
import data
import main
import jumps, ab

import shutil, os, sys

def note(s): sys.stderr.write(str(s) + '\n'); sys.stderr.flush()

def do():
    global d 
    d = data.Data(alphabetic_baseforms = False)
    global fenomes 
    fenomes = main.make_fenomes(d)

    global M 
    M = []
    for i in range(4):
        ps, fenomes = main.iterate(fenomes, d)
        note("%s -- %s" % (i, np.mean(ps)))
        M.append(ps)

    global P, K, decode, conf
    P = []
    for i, bf in enumerate(d.baseforms):
        ps = ab.forwards(fenomes, bf, d.test_instances)
        P.append(ps)
    P = np.transpose(P)
    none = np.invert(np.all(np.isinf(P), 1))
    vocmod = d.vocab
    vocmod.append("<unknown>")
    K = np.argmax(P, 1) * none - np.invert(none)
    decode = map(lambda x: vocmod[x], K)
    Ps = np.sort(P, 1)
    conf = np.diff(Ps[:, -2:]).squeeze()
    def stripnan(x):
        if np.isnan(x):
            return 0
        else:
            return 1/(1+np.exp(-x))
    conf = map(stripnan, conf)

    for word, c in zip(decode, conf):
        print (word, c)
        #print word


    global Q
    Q = []
    for x in P:
        m = np.argmax(x)
        Q.append((m, x[m]/np.sum(filter(lambda e: not np.isinf(e), x))))

def test_jumps():
    foo = lambda: jumps.jump_matrices(main.d.baseforms[0], main.fenomes)
    a, b, c, d, e = foo()
    #Image.fromarray(np.uint8(255*e/np.max(e))).save('out.png')

if __name__ == '__main__':
    do()
