#!/usr/bin/env python
import numpy as np
import main
import jumps
import ab
import Image

def do():
    ab.train(main.d.baseforms[0], main.fenomes, main.d.observations[0])

def test_jumps():
    foo = lambda: jumps.jump_matrices(main.d.baseforms[0], main.fenomes)
    a, b, c, d, e = foo()
    Image.fromarray(np.uint8(255*e/np.max(e))).save('out.png')


if __name__ == '__main__':
    do()
