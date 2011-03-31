import numpy as np
cimport numpy as np
cimport cython
#import Image
import jumps
import sys

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def train(fenomes, baseforms, observations):
    cdef:
        np.ndarray[Py_ssize_t, ndim=1] baseform

        np.ndarray[DTYPE_t, ndim=2] alphas, betas
        np.ndarray[DTYPE_t, ndim=1] As, Bs
        np.ndarray[DTYPE_t, ndim=1] Pv

        np.ndarray[np.float64_t, ndim=1] ptrans, pnull
        np.ndarray[np.int_t, ndim=2] strans, snull
        np.ndarray[np.float64_t, ndim=2] qemiss

        np.ndarray[Py_ssize_t, ndim=1] inst

        np.ndarray[np.float64_t, ndim=2] qcount
        np.ndarray[np.float64_t, ndim=2] qnet, qnet_sil, arrownet
        np.ndarray[np.float64_t, ndim=1] tcount, bcount
        np.ndarray[np.float64_t, ndim=1] tnet, tnet_sil, bnet, bnet_sil

        Py_ssize_t idx, i, j, k, s, t, o, fe
        Py_ssize_t ns, nt, nb, nlbl, nfene, nfene_total

        np.float64_t acc, total, z
        Py_ssize_t nobs, nvoc, ninst

    # A constant throughout the entire process
    nlbl = fenomes[0].e.shape[1]
    nfene_total = len(fenomes)
    nvoc = len(baseforms)

    # Initialize the net counters
    qnet = np.zeros([2*nfene_total, nlbl])
    qnet_sil = np.zeros([9, nlbl])
    tnet = np.zeros([2*nfene_total])
    tnet_sil = np.zeros([9])
    bnet = np.zeros([nfene_total])
    bnet_sil = np.zeros([3])
    Pv = np.zeros([len(baseforms) * 10])

    # Begin cycling through the words
    for idx in range(nvoc):
        baseform = baseforms[idx]
        instances = observations[idx]

        # Build the net baseform HMM model
        ptrans, pnull, strans, snull, qemiss = \
                jumps.jump_matrices(baseform, fenomes)

        # Initialize some constants for the whole iteration
        nfene = baseform.shape[0]
        ns = 2*6 + nfene + 1
        nt = ptrans.shape[0]
        nb = pnull.shape[0]

        # Initialize the counters
        qcount = np.zeros([nt, nlbl])
        tcount = np.zeros([nt])
        bcount = np.zeros([nb])

        ninst = len(instances)
        # Begin the training
        for i in range(ninst):
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

            # Save this forward probability
            print As
            Pv[idx*ninst + i] = np.mean(np.log(As))

            # Begin computing the betas
            betas = np.zeros([ns, nobs+1])
            Bs    = np.zeros([nobs+1])
            betas[:, nobs] = 1/<np.float64_t>ns
            Bs[nobs] = 1
                        
            for t from nobs > t >= 0:
                total = 0
                o = inst[t]
                # Real transtions
                for j from nt > j >= 0:
                    acc = betas[strans[j, 1], t+1] * ptrans[j] * qemiss[j, o]
                    total += acc
                    betas[strans[j, 0], t] += acc
                # Null transitions
                for j from nb > j >= 0:
                    acc = betas[snull[j, 1], t] * pnull[j]
                    total += acc
                    betas[snull[j, 0], t] += acc
                # Renormalization
                Bs[t] = total
                for s in range(ns):
                    betas[s, t] /= Bs[t]

            # Accumulate the soft counts
            for j in range(nt):
                z = 1
                for t in range(nobs):
                    z *= As[t]/Bs[t]
                    o = inst[t]
                    acc = alphas[strans[j, 0], t] * betas[strans[j, 1], t+1] \
                            * ptrans[j] * qemiss[j, o]
                    tcount[j] += acc
                    qcount[j, o] += acc
            for j in range(nb):
                z = 1
                for t in range(nobs):
                    z *= As[t]/Bs[t]
                    bcount[j] += alphas[snull[j, 0], t] * betas[snull[j, 1], t] \
                           * pnull[j] * Bs[t]

            #Image.fromarray(np.uint8(255*np.isnan(qnet))).save(
                    #'exp/emiss{idx}_{i}.png'.format( idx = idx, i = i))
            #Image.fromarray(np.uint8(255*(np.r_[alphas, betas]/np.max(alphas)))).save(
                    #'exp/ab{idx}_{i}.png'.format( idx = idx, i = i))

            # Shuffle into the net counters
            for j in range(len(baseform)):
                fe = baseform[j]
                
                for k in range(nlbl):
                    qnet[2*fe  , k] += qcount[9 + 2*j    , k]
                    qnet[2*fe+1, k] += qcount[9 + 2*j + 1, k]

                tnet[2*fe  ] += tcount[9 + 2*j    ]
                tnet[2*fe+1] += tcount[9 + 2*j + 1]
                bnet[fe  ]   += bcount[3 + j      ]
                bnet[fe+1]   += bcount[3 + j + 1  ]

            qnet_sil += qcount[  :9, :]
            qnet_sil += qcount[-9: , :]
            tnet_sil += tcount[:9] + tcount[-9:]
            bnet_sil += bcount[:3] + bcount[-3:]

            #Image.fromarray(np.uint8(255*(np.c_[np.reshape(tnet, [nfene_total, 2]),
                #np.reshape(bnet, [nfene_total, 1])]/np.max(tnet)))).save(
                    #'exp/trans{idx}_{i}.png'.format( idx = idx, i = i))

    #Renormalize the distributions
    for j in range(qnet.shape[0]):
        total = 0
        for k in range(nlbl):
            total += qnet[j, k]
        for k in range(nlbl):
            qnet[j, k] /= total

    for j in range(qnet_sil.shape[0]):
        total = 0
        for k in range(nlbl):
            total += qnet_sil[j, k]
        for k in range(nlbl):
            qnet_sil[j, k] /= total

    arrownet = np.c_[np.reshape(tnet, [nfene_total, 2]),
                        np.reshape(bnet, [nfene_total, 1])]
    for j in range(nfene_total):
        total = 0
        for k in range(3):
            total += arrownet[j, k]
        for k in range(3):
            arrownet[j, k] /= total

    return Pv, qnet, arrownet, qnet_sil, tnet_sil, bnet_sil
