#!/usr/bin/env python
import numpy as np
import data
import main
import jumps

import sys

def note(s): sys.stderr.write(str(s) + '\n'); sys.stderr.flush()

def do():
    global d 
    d = data.Data(alphabetic_baseforms = False)
    global fenomes 
    fenomes = main.make_fenomes(d)

    global P 
    P = []
    for i in range(4):
        note(i)
        ps, fenomes = main.iterate(fenomes, d)
        note( np.mean(ps) )
        P.append(ps)

def test_jumps():
    foo = lambda: jumps.jump_matrices(main.d.baseforms[0], main.fenomes)
    a, b, c, d, e = foo()
    Image.fromarray(np.uint8(255*e/np.max(e))).save('out.png')

if __name__ == '__main__':
    do()
