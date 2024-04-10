# -*- coding: utf-8 -*-
# doDMRG_MPO.py
import numpy as np
from numpy import linalg as LA
from ncon import ncon
from BiorthoLib import left_decomp, right_decomp, eigLR

def doDMRG_lr(M, W, chi_max, numsweeps = 10, dispon = 2, updateon = True, debug = False, which = "SR"):
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 19/1/2019
------------------------
Implementation of DMRG for a 1D chain with open boundaries, using the \
two-site update strategy. Each update is accomplished using a custom \
implementation of the Lanczos iteration to find (an approximation to) the \
ground state of the superblock Hamiltonian. Input 'M' is containing the MPS \
tensors whose length is equal to that of the 1D lattice. The Hamiltonian is \
specified by an MPO with 'ML' and 'MR' the tensors at the left and right \
boundaries, and 'M' the bulk MPO tensor. Automatically grow the MPS bond \
dimension to maximum dimension 'chi'. Outputs 'A' and 'B' are arrays of the \
MPS tensors in left and right orthogonal form respectively, while 'sWeight' \
is an array of the Schmidt coefficients across different lattice positions. \
'Ekeep' is a vector describing the energy at each update step.

Optional arguments:
`numsweeps::Integer=10`: number of DMRG sweeps
`dispon::Integer=2`: print data never [0], after each sweep [1], each step [2]
`updateon::Bool=true`: enable or disable tensor updates
`maxit::Integer=2`: number of iterations of Lanczos method for each diagonalization
`krydim::Integer=4`: maximum dimension of Krylov space in superblock diagonalization
"""

    ##### left-to-right 'warmup', put MPS in right orthogonal form
    # Index of M is: left'' - right'' - physical - physical'
    Nsites = len(M)
    
    # The L[i] operator is the MPO contracted with the MPS and its dagger for sites <= i-1
    # R[i] is contracted for sites >= i+1
    L = [0 for _ in range(Nsites)]; L[0] = np.ones((1,1,1))
    R = [0 for _ in range(Nsites)]; R[Nsites-1] = np.ones((1,1,1))
    A = [0 for _ in range(Nsites)]
    B = [0 for _ in range(Nsites)]
    for p in range(Nsites-1,0,-1): # Do right normalization, from site Nsites-1 to 1

        #print("p = {}".format(p))

        # Shape of M is: left bond - physical bond - right bond
        if np.shape(M[p]) != np.shape(Mb[p]):
            raise Exception("Shapes of M[p] and Mb[p] must match!")
        
        # Set the p-th matrix to right normal form, and multiply the transform matrix to p-1
        B[p], Q = right_decomp(M[p], chi_max=chi_max)
        M[p-1] = ncon([M[p-1],Q], [[-1,-2,1],[1,-3]])

        # Construct R[p-1]. The indices of R is: left'' - left - left'
        R[p-1] = ncon([R[p], W[p], B[p], B[p].conj()],[[3,1,5],[-1,3,2,4],[-2,2,1],[-3,4,5]])

    # Normalize M[0] and Mb[0] so that the trial wave functions are bi-normalized
    ratio = 1/np.sqrt(ncon([M[0],M[0].conj()],[[1,2,3],[1,2,3]]))
    M[0] *= ratio

    # At this point we have turned M[1:] to right normal form, and constructed R[1:]
    # We start the sweep at site 0
    # The effective Hamiltonian at site [i] is the contraction of L[i], R[i], and W[i]
    
    Ekeep = np.array([])
    Hdifs = np.array([])
    for k in range(1,numsweeps+2):
        
        ##### final sweep is only for orthogonalization (disable updates)
        if k == numsweeps+1:
            updateon = False
            dispon = 0
        
        ###### Optimization sweep: left-to-right
        for p in range(Nsites-1):

            # Optimize at this step
            # Move the pointer one site to the right, and left-normalize the matrices at the currenter pointer
            if updateon:
                E, tM, tMb = eigLR(L[p], R[p], W[p], M[p], M[p].conj(), which = which)
                rho = 
                Ekeep = np.append(Ekeep, E)

            else:
                A[p], Q = left_decomp_svd(M[p], chi_max=chi_max)

            M[p+1] = ncon([Q,B[p+1]], [[-1,1],[1,-2,-3]])

            # Construct L[p+1]
            L[p+1] = ncon([L[p], W[p], Y[p], Yb[p]], [[3,1,5],[3,-1,2,4],[1,2,-2],[5,4,-3]])
        
            ##### display energy
            if dispon == 2:
                print('Sweep: {} of {}, Loc: {},Energy: {:.3f}'.format(k, numsweeps, p, Ekeep[-1]))

        # Set Y[-1]
        Y[-1] = M[-1]
        Yb[-1] = Mb[-1]
        
        ###### Optimization sweep: right-to-left
        for p in range(Nsites-1,0,-1):

            #print("p = {}".format(p))

            # Optimize at this step
            if updateon:
                E, M[p], Mb[p] = eigLR(L[p], R[p], W[p], M[p], Mb[p], which = which)
                Ekeep = np.append(Ekeep, E)

            # Move the pointer one site to the left, and right-normalize the matrices at the currenter pointer
            Z[p], Zb[p], I, Ib = right_decomp(M[p], Mb[p], chi_max=chi_max, timing=debug)
            M[p-1] = ncon([Y[p-1],I], [[-1,-2,1],[1,-3]])
            Mb[p-1] = ncon([Yb[p-1],Ib], [[-1,-2,1],[1,-3]])

            # Construct R[p-1]. The indices of R is: left'' - left - left'
            R[p-1] = ncon([R[p], W[p], Z[p], Zb[p]],[[3,1,5],[-1,3,2,4],[-2,2,1],[-3,4,5]])
        
            ##### display energy
            if dispon == 2:
                print('Sweep: {} of {}, Loc: {},Energy: {:.3f}'.format(k, numsweeps, p, Ekeep[-1]))

        # Set Z[0]
        Z[0] = M[0]
        Zb[0] = Mb[0]
        
        if dispon >= 1:

            RR = np.ones((1,1,1,1))
            for p in range(Nsites-1,-1,-1):
                RR = ncon([RR,W[p],W[p],Z[p],Zb[p]], [[5,3,1,6],[-1,5,4,7],[-2,3,2,4],[-3,2,1],[-4,7,6]])

            Hdifs = np.append(Hdifs, RR.flatten()[0])

            print('Sweep: {} of {}, Energy: {:.3f}, H dif: {}, Bond dim: {}'.format(k, numsweeps, Ekeep[-1], RR.flatten()[0]-Ekeep[-1]**2, chi_max))
            
    return Ekeep, Hdifs, Y, Yb, Z, Zb

def left_decomp_svd (M, chi_max = 0):

    if len(np.shape(M)) == 2:
        M = M.reshape(1,np.shape(M)[0], np.shape(M)[1])
    dL, dS, _ = np.shape(M)
    U, S, V = LA.svd(M.reshape(dL*dS,-1))
    if 0 < chi_max < np.size(S):
        U = U[:,:chi_max]
        S = S[:,:chi_max]
        V = V[:chi_max,:]
    return U.reshape(np.shape(M)), np.diag(S)@V