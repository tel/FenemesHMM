import numpy as np
cimport numpy as np
cimport cython

import jumps

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def train(np.ndarray[Py_ssize_t, ndim=1] baseform, fenomes, instances):
    cdef:
        np.ndarray[DTYPE_t, ndim=2] alphas, betas
        np.ndarray[DTYPE_t, ndim=1] As, Bs

        np.ndarray[np.float64_t, ndim=1] ptrans, pnull
        np.ndarray[np.int_t, ndim=2] strans, snull
        np.ndarray[np.float64_t, ndim=2] qemiss

        np.ndarray[Py_ssize_t, ndim=1] inst

        Py_ssize_t i, j, k, s, t, o
        Py_ssize_t ns, nt, nb, nlbl, nfene

        np.float64_t acc, total
        Py_ssize_t nobs

    # Build the net baseform HMM model
    ptrans, pnull, strans, snull, qemiss = \
            jumps.jump_matrices(baseform, fenomes)

    # Initialize some constants for the whole iteration
    nfene = baseform.shape[0]
    ns = 2*6 + nfene + 1
    nt = ptrans.shape[0]
    nb = pnull.shape[0]
    nlbl = qemiss.shape[1]

    # Begin the training
    for i in range(len(instances)):

        inst = instances[i]
        nobs = inst.shape[0]
    
        # Begin computing the alphas
        alphas = np.zeros([ns, nobs+1])
        As     = np.zeros([nobs+1])
        alphas[0, 0] = 1
        As[0] = 1

        for t in range(1, nobs+1):
            total = 0
            o = inst[t-1]
            # Real transitions
            for j in range(nt):
                acc = alphas[strans[j, 0], t-1] * ptrans[j] * qemiss[j, o]
                total += acc
                alphas[strans[j, 1], t] += acc
            # Null transitions
            for j in range(nb):
                acc = alphas[snull[j, 0], t] * pnull[j]
                total += acc
                alphas[snull[j, 1], t] += acc
            # Renormalization
            As[t] = total
            for s in range(ns):
                alphas[s, t] /= As[t]

        # Begin computing the betas
        betas = np.zeros([ns, nobs+1])
        Bs    = np.zeros([nobs+1])
        betas[:, nobs] = 1/<np.float64_t>ns
        Bs[nobs] = 1
                    

    #return counts, alphas, betas
