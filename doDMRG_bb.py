# -*- coding: utf-8 -*-
# doDMRG_MPO.py
import numpy as np
from numpy import linalg as LA
from ncon import ncon
from BiorthoLib import left_decomp, right_decomp, eigLR

<<<<<<< Updated upstream
def doDMRG_bb(M, Mb, W, chi_max, numsweeps = 10, dispon = 2, updateon = True, debug = False, which = "SR", method = "biortho"):
=======
def doDMRG_excited(M, Mb, W, chi_max, k=1, which = "SM", expected_gap = 1, numsweeps = 10, dispon = 2, debug = False, method = "biortho", cut = 1e-8, stop_if_not_converge = True, log_write = print):

    # which should be "SM" or "LR"
    
    Ms = []
    Mbs = []
    Es = []

    if not isinstance(W, MPO):
        W = MPO(W)

    for thisk in range(k):
        log_write(f"Finding eigenvalue #{thisk+1}")

        Ekeep, Hdifs, Y, Yb, Z, Zb = doDMRG_IncChi(M, Mb, W, chi_max, which = which,
            normalize_against = [(Ms[i],Mbs[i],-expected_gap*(thisk-i)) for i in range(thisk)],
            vt_amp=4, chi_start=16,
            numsweeps=numsweeps,dispon=dispon,debug=debug,method=method,cut=cut,log_write=log_write)
        if np.abs(Hdifs[-1]) < 1e-3:
            log_write(f"Found eigenvalue #{thisk+1}")
        else:
            if stop_if_not_converge:
                raise Exception(f"Failed to converge for eigenvalue #{thisk+1}: <Delta H^2> = {Hdifs[-1]}")
            else:
                log_write(f"ERROR: Failed to converge for eigenvalue #{thisk+1}: <Delta H^2> = {Hdifs[-1]}")

        M = [Zi.copy() for Zi in Z]
        Mb = [Zbi.copy() for Zbi in Z]

        Es.append(Ekeep[-1])
        Ms.append(Y)
        
        if method == "biortho":
            Mbs.append(Yb)

        else:
            Ekeep, Hdifs, Y, _, _, _ = doDMRG_IncChi(Zb, Z, W.transpose(), chi_max, which = Es[-1],
            normalize_against = [(Mbs[i],Ms[i],-expected_gap*(thisk-i)) for i in range(thisk)],
            numsweeps=numsweeps,dispon=dispon,debug=debug,method=method,cut=cut,log_write=log_write)
            
            if np.abs(Hdifs[-1]) > 1e-3 or np.abs(Ekeep[-1]-Es[-1]) > 1e-3:
                if stop_if_not_converge:
                    raise Exception(f"Failed to converge for eigenvalue #{thisk+1}: <Delta H^2> = {Hdifs[-1]}, Delta E = {np.abs(Ekeep[-1]-Es[-1])}")
                else:
                    log_write(f"Failed to converge for eigenvalue #{thisk+1}: <Delta H^2> = {Hdifs[-1]}, Delta E = {np.abs(Ekeep[-1]-Es[-1])}")

            Mbs.append(Y)

    return Ms, Mbs, Es
    

def doDMRG_IncChi (M, Mb, W, chi_max, chi_inc = 10, chi_start = 20, init_sweeps = 5, inc_sweeps = 2, tol_start = 1e-3, tol_end = 1e-6, vt_amp = 3, vt_sweeps = 3, numsweeps = 10, dispon = 2, debug = False, which = "SR", method = "biortho", cut = 1e-8, normalize_against = [], log_write = print):

    _,_,M,Mb,_,_ = doDMRG_bb(M, Mb, W, chi_start, tol=tol_start,numsweeps=init_sweeps,dispon=dispon,updateon=True,debug=debug,which=which,method=method,normalize_against=normalize_against,log_write=log_write)
    
    chi = chi_start
    while True:
        chi += chi_inc
        if chi >= chi_max:
            break
        _,_,M,Mb,_,_ = doDMRG_bb(M, Mb, W, chi, tol=tol_start,numsweeps=inc_sweeps,dispon=dispon,updateon=True,debug=debug,which=which,method=method,normalize_against=normalize_against,log_write=log_write)

    chi = chi_max
    tol = tol_start
    while tol > tol_end:
        _,_,M,Mb,_,_ = doDMRG_bb(M, Mb, W, chi, tol=tol,numsweeps=vt_sweeps,dispon=dispon,updateon=True,debug=debug,which=which,method=method,normalize_against=normalize_against,log_write=log_write)
        tol *= 10**(-vt_amp)

    return doDMRG_bb(M, Mb, W, chi_max, tol=tol_end, numsweeps=numsweeps,dispon=dispon,updateon=True,debug=debug,which=which,method=method,normalize_against=normalize_against,log_write=log_write)


def doDMRG_bb(M, Mb, W, chi_max, numsweeps = 10, dispon = 2, updateon = True, debug = False, which = "SR", method = "biortho", tol=0, normalize_against = [], log_write = print):
>>>>>>> Stashed changes
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 19/1/2019
------------------------
Implementation of DMRG for a 1D chain with open boundaries. Input 'M' is containing the MPS \
tensors whose length is equal to that of the 1D lattice, and 'Mb' is the corresponding left \
vector. The Hamiltonian is specified by an MPO with entries 'W'. Automatically grow the MPS bond \
dimension to maximum dimension 'chi_max'.

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
    if len(Mb) != Nsites:
        raise Exception("Length of M and Mb must match!")
    
    # The L[i] operator is the MPO contracted with the MPS and its dagger for sites <= i-1
    # R[i] is contracted for sites >= i+1
    L = [0 for _ in range(Nsites)]; L[0] = np.ones((1,1,1))
    R = [0 for _ in range(Nsites)]; R[Nsites-1] = np.ones((1,1,1))
    Y = [0 for _ in range(Nsites)]
    Yb = [0 for _ in range(Nsites)]
    Z = [0 for _ in range(Nsites)]
    Zb = [0 for _ in range(Nsites)]
    for p in range(Nsites-1,0,-1): # Do right normalization, from site Nsites-1 to 1

        # Shape of M is: left bond - physical bond - right bond
        if np.shape(M[p]) != np.shape(Mb[p]):
            raise Exception("Shapes of M[p] and Mb[p] must match!")
        
        # Set the p-th matrix to right normal form, and multiply the transform matrix to p-1
        Z[p], Zb[p], I, Ib = right_decomp(M[p], Mb[p], chi_max=chi_max, timing=debug, method=method)
        M[p-1] = ncon([M[p-1],I], [[-1,-2,1],[1,-3]])
        Mb[p-1] = ncon([Mb[p-1],Ib], [[-1,-2,1],[1,-3]])

        # Construct R[p-1]. The indices of R is: left'' - left - left'
        R[p-1] = ncon([R[p], W[p], Z[p], Zb[p]],[[3,1,5],[-1,3,2,4],[-2,2,1],[-3,4,5]])

    # Normalize M[0] and Mb[0] so that the trial wave functions are bi-normalized
    ratio = 1/np.sqrt(ncon([M[0],Mb[0]],[[1,2,3],[1,2,3]]))
    M[0] *= ratio
    Mb[0] *= ratio

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
            if updateon:
                E, M[p], Mb[p] = eigLR(L[p], R[p], W[p], M[p], Mb[p], which = which)
                Ekeep = np.append(Ekeep, E)

            # Move the pointer one site to the right, and left-normalize the matrices at the currenter pointer
            Y[p], Yb[p], I, Ib = left_decomp(M[p], Mb[p], chi_max=chi_max, timing=debug, method=method)

            M[p+1] = ncon([I,Z[p+1]], [[-1,1],[1,-2,-3]])
            Mb[p+1] = ncon([Ib,Zb[p+1]], [[-1,1],[1,-2,-3]])

            # Construct L[p+1]
            L[p+1] = ncon([L[p], W[p], Y[p], Yb[p]], [[3,1,5],[3,-1,2,4],[1,2,-2],[5,4,-3]])
        
            ##### display energy
            if dispon == 2:
                log_write('Sweep: {} of {}, Loc: {},Energy: {:.3f}'.format(k, numsweeps, p, Ekeep[-1]))

        # Set Y[-1]
        Y[-1] = M[-1]
        Yb[-1] = Mb[-1]
        
        ###### Optimization sweep: right-to-left
        for p in range(Nsites-1,0,-1):

            # Optimize at this step
            if updateon:
                E, M[p], Mb[p] = eigLR(L[p], R[p], W[p], M[p], Mb[p], which = which)
                Ekeep = np.append(Ekeep, E)

            # Move the pointer one site to the left, and right-normalize the matrices at the currenter pointer
            Z[p], Zb[p], I, Ib = right_decomp(M[p], Mb[p], chi_max=chi_max, timing=debug, method=method)
            M[p-1] = ncon([Y[p-1],I], [[-1,-2,1],[1,-3]])
            Mb[p-1] = ncon([Yb[p-1],Ib], [[-1,-2,1],[1,-3]])

            # Construct R[p-1]. The indices of R is: left'' - left - left'
            R[p-1] = ncon([R[p], W[p], Z[p], Zb[p]],[[3,1,5],[-1,3,2,4],[-2,2,1],[-3,4,5]])
        
            ##### display energy
            if dispon == 2:
                log_write('Sweep: {} of {}, Loc: {},Energy: {:.3f}'.format(k, numsweeps, p, Ekeep[-1]))

        # Set Z[0]
        Z[0] = M[0]
        Zb[0] = Mb[0]
        
        if dispon >= 1:

            """
            # Calculate <1>
            R0 = np.ones((1,1))
            for p in range(Nsites-1,-1,-1):
                R0 = ncon([R0,Z[p],Zb[p]], [[1,2],[-1,3,1],[-2,3,2]])
            norm = R0.flatten()[0]
            """

<<<<<<< Updated upstream
            # Calculate <H^2>-<H>^2
            RR = np.ones((1,1,1,1))
            for p in range(Nsites-1,-1,-1):
                RR = ncon([RR,W[p],W[p],Z[p],Zb[p]], [[5,3,1,6],[-1,5,4,7],[-2,3,2,4],[-3,2,1],[-4,7,6]])

            Hdif = RR.flatten()[0]-Ekeep[-1]**2
            Hdifs = np.append(Hdifs, Hdif)
=======
            log_write('Sweep: {} of {}, Energy: {:.3f}, H dif: {}, Bond dim: {}, tol: {}'.format(k, numsweeps, Ekeep[-1], Hdif, chi_max, tol))

        cut = max(tol, np.finfo(float).eps) * 10
        # Early termination if converged
        if np.abs(np.std(Ekeep[-2*Nsites:])) < cut and np.abs(Hdif) < cut:
            log_write("Converged")
            k = numsweeps+1
>>>>>>> Stashed changes

            print('Sweep: {} of {}, Energy: {:.3f}, H dif: {}, Bond dim: {}'.format(k, numsweeps, Ekeep[-1], Hdif, chi_max))
            
    return Ekeep, Hdifs, Y, Yb, Z, Zb