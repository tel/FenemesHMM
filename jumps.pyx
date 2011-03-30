import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def jump_matrices(baseform, fenomes):
    cdef:
        Py_ssize_t nfene, nt, nb, nlbl
        Py_ssize_t it = 0
        Py_ssize_t ib = 0
        Py_ssize_t s, t, j
        np.ndarray[np.float64_t, ndim=1] ptrans, pnull, ftrans
        np.ndarray[np.int_t, ndim=2] strans, snull
        np.ndarray[np.float64_t, ndim=2] qemiss, femiss

    nfene = len(baseform)
    nt = 18 + 2*nfene
    nb = 6 + nfene
    nlbl = fenomes[0].e.shape[1]

    ptrans = np.empty(nt)
    pnull  = np.empty(nb)
    strans = np.ones([nt, 2], dtype = np.int)
    snull  = np.ones([nb, 2], dtype = np.int)
    qemiss = np.empty([nt, nlbl])

    # load the fenome transitions
    for t in range(nfene):
        it = 9 + 2*t
        ib = 3 + t
        s = 6 + t

        fene = fenomes[baseform[t]]
        femiss = fene.e
        ftrans = fene.t
        # Insert the emission densities
        for j in range(nlbl):
            qemiss[it  , j] = femiss[0, j]
            qemiss[it+1, j] = femiss[1, j]

        # Insert the self transition
        ptrans[it   ] = ftrans[1]
        strans[it, 0] = s
        strans[it, 1] = s

        # Insert the stepping transition
        ptrans[it+1   ] = ftrans[0]
        strans[it+1, 0] = s
        strans[it+1, 1] = s+1

        # Insert the null transition
        pnull[ib   ] = ftrans[2]
        snull[ib, 0] = s
        snull[ib, 1] = s+1

    # load the silence transitions
    silence = fenomes[-1]
    for j in range(9):
        ptrans[j] = silence.t[j]
        ptrans[2+it+j] = silence.t[j]
    for j in range(3):
        pnull[j] = silence.t[9+j]
        pnull[1+ib+j] = silence.t[9+j]

    strans[0:9, :] = np.array(
            [0, 1, 1, 1, 1, 2, 2, 2, 2, 
                6, 0, 3, 3, 4, 4, 5, 5, 6]
            ).reshape([9, 2])
    strans[(2+it):nt, :] = strans[0:9, :] + 6+nfene

    snull[0:3, :] = np.array(
            [3, 6, 4, 6, 5, 6]
            ).reshape([3, 2])
    snull[(1+ib):nb, :] = snull[0:3, :] + 6+nfene

    qemiss[0:9, :] = silence.e
    qemiss[(2+it):nt, :] = silence.e

    return ptrans, pnull, strans, snull, qemiss
