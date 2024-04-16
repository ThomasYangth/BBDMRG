# -*- coding: utf-8 -*-
# doDMRG_bb.py

from Config import *
from BiorthoLib import left_decomp, right_decomp, eigLR
from MPSlib import *

def doDMRG_excited(M, Mb, W, chi_max, k=1, which = "SM", expected_gap = 1, numsweeps = 10, dispon = 2, debug = False, method = "biortho", cut = 1e-8, stop_if_not_converge = True):

    # which should be "SM" or "LR"
    
    Ms = []
    Mbs = []
    Es = []

    if not isinstance(W, MPO):
        W = MPO(W)

    for thisk in range(k):
        print(f"Finding eigenvalue #{thisk+1}")

        Ekeep, Hdifs, Y, Yb, Z, Zb = doDMRG_IncChi(M, Mb, W, chi_max, which = which,
            normalize_against = [(Ms[i],Mbs[i],-expected_gap*(thisk-i)) for i in range(thisk)],
            vt_amp=4,
            numsweeps=numsweeps,dispon=dispon,debug=debug,method=method,cut=cut)
        if np.abs(Hdifs[-1]) < 1e-3:
            print(f"Found eigenvalue #{thisk+1}")
        else:
            if stop_if_not_converge:
                raise Exception(f"Failed to converge for eigenvalue #{thisk+1}: <Delta H^2> = {Hdifs[-1]}")
            else:
                print(f"ERROR: Failed to converge for eigenvalue #{thisk+1}: <Delta H^2> = {Hdifs[-1]}")

        M = [Zi.copy() for Zi in Z]
        Mb = [Zbi.copy() for Zbi in Z]

        Es.append(Ekeep[-1])
        Ms.append(Y)
        
        if method == "biortho":
            Mbs.append(Yb)

        else:
            Ekeep, Hdifs, Y, _, _, _ = doDMRG_IncChi(Zb, Z, W.transpose(), chi_max, which = Es[-1],
            normalize_against = [(Mbs[i],Ms[i],-expected_gap*(thisk-i)) for i in range(thisk)],
            numsweeps=numsweeps,dispon=dispon,debug=debug,method=method,cut=cut)
            
            if np.abs(Hdifs[-1]) > 1e-3 or np.abs(Ekeep[-1]-Es[-1]) > 1e-3:
                if stop_if_not_converge:
                    raise Exception(f"Failed to converge for eigenvalue #{thisk+1}: <Delta H^2> = {Hdifs[-1]}, Delta E = {np.abs(Ekeep[-1]-Es[-1])}")
                else:
                    print(f"Failed to converge for eigenvalue #{thisk+1}: <Delta H^2> = {Hdifs[-1]}, Delta E = {np.abs(Ekeep[-1]-Es[-1])}")

            Mbs.append(Y)

    return Ms, Mbs, Es
    

def doDMRG_IncChi (M, Mb, W, chi_max, chi_inc = 10, chi_start = 20, inc_sweeps = 2, tol_start = 1e-3, vt_amp = 3, vt_sweeps = 3, numsweeps = 10, dispon = 2, debug = False, which = "SR", method = "biortho", cut = 1e-8, normalize_against = []):

    chi = chi_start
    while chi < chi_max:
        _,_,M,Mb,_,_ = doDMRG_bb(M, Mb, W, chi, tol=tol_start,numsweeps=inc_sweeps,dispon=dispon,updateon=True,debug=debug,which=which,method=method,normalize_against=normalize_against)
        chi = min(chi+chi_inc, chi_max)

    tol = tol_start
    tol_end = np.finfo(float).eps
    while tol > tol_end:
        _,_,M,Mb,_,_ = doDMRG_bb(M, Mb, W, chi, tol=tol,numsweeps=vt_sweeps,dispon=dispon,updateon=True,debug=debug,which=which,method=method,normalize_against=normalize_against)
        tol *= 10**(-vt_amp)

    return doDMRG_bb(M, Mb, W, chi_max, numsweeps=numsweeps,dispon=dispon,updateon=True,debug=debug,which=which,method=method,normalize_against=normalize_against)


def doDMRG_bb(M, Mb, W, chi_max, numsweeps = 10, dispon = 2, updateon = True, debug = False, which = "SR", method = "biortho", tol=0, normalize_against = []):
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 19/1/2019
------------------------
modified by Tian-Hua Yang
------------------------
Implementation of DMRG for a 1D chain with open boundaries. Input 'M' is containing the MPS \
tensors whose length is equal to that of the 1D lattice, and 'Mb' is the corresponding left \
vector. The Hamiltonian is specified by an MPO with entries 'W'. Automatically grow the MPS bond \
dimension to maximum dimension 'chi_max'.

Optional arguments:
`numsweeps::Integer=10`: number of DMRG sweeps
`dispon::Integer=2`: print data never [0], after each sweep [1], each step [2]
`updateon::Bool=true`: enable or disable tensor updates
`debug::Bool=False`: enable debugging messages
`which::str="SR"`: which eigenvalue to choose, "SR" indicates smallest real part
`method::str="biortho"`: method for truncation of density matrix; 'biortho' is for bbDMRG, \
    'lrrho' for using the density matrix rho=(psiL psiL + psiR psiR)/2
"""

    ##### left-to-right 'warmup', put MPS in right orthogonal form
    # Index of W is: left'' - right'' - physical' - physical
    # Index notation: no prime = ket, one prime = bra, two primes = operator link
    Nsites = len(M)
    if len(Mb) != Nsites:
        raise Exception("Length of M and Mb must match!")

    # Each element in normalize_against should be a tuple (Mi, Mib, amp)
    # Corresponding to adding a term amp * Mi*Mib to the Hamiltonian
    # For this we record LNA, RNA, LNAb, RNAb
    # LNA[i] corresponds to the product of Mib with the current M at site i
    # Simialr for the other three
    NumNA = len(normalize_against)
    LNA = []
    RNA = []
    LNAb = []
    RNAb = []
    Namp = []
    MN = []
    MNb = []
    for i,item in enumerate(normalize_against):
        LNA.append([np.ones((1,1))]+[0 for _ in range(Nsites-1)])
        RNA.append([0 for _ in range(Nsites-1)]+[np.ones((1,1))])
        LNAb.append([np.ones((1,1))]+[0 for _ in range(Nsites-1)])
        RNAb.append([0 for _ in range(Nsites-1)]+[np.ones((1,1))])
        MN.append(item[0])
        MNb.append(item[1])
        Namp.append(item[2])
    
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

        # Construct R[p-1]. The indices of R is: left'' - left' - left
        R[p-1] = ncon([R[p], W[p], Zb[p], Z[p]],[[3,1,5],[-1,3,2,4],[-2,2,1],[-3,4,5]])

        for i in range(NumNA):
            RNA[i][p-1] = ncon([RNA[i][p], MNb[i][p], Z[p]], [[1,2],[-1,3,1],[-2,3,2]])
            RNAb[i][p-1] = ncon([RNAb[i][p], Zb[p], MN[i][p]], [[1,2],[-1,3,1],[-2,3,2]])

    # Normalize M[0] and Mb[0] so that the trial wave functions are bi-normalized
    ratio = 1/np.sqrt(ncon([M[0],Mb[0]],[[1,2,3],[1,2,3]]))
    M[0] *= ratio
    Mb[0] *= ratio

    # At this point we have turned M[1:] to right normal form, and constructed R[1:]
    # We start the sweep at site 0
    # The effective Hamiltonian at site [i] is the contraction of L[i], R[i], and W[i]
    
    Ekeep = np.array([])
    Hdifs = np.array([])

    k = 1

    while k <= numsweeps+1:
        
        ##### final sweep is only for orthogonalization (disable updates)
        if k == numsweeps+1:
            updateon = False
            dispon = 0
        
        ###### Optimization sweep: left-to-right
        for p in range(Nsites-1):

            # Optimize at this step
            if updateon:
                E, M[p], Mb[p] = eigLR(L[p], R[p], W[p], M[p], Mb[p], which = which, tol=tol,
                    normalize_against = [(LNA[i][p],RNA[i][p],MNb[i][p],LNAb[i][p],RNAb[i][p],MN[i][p],Namp[i]) for i in range(NumNA)])
                Ekeep = np.append(Ekeep, E)

            # Move the pointer one site to the right, and left-normalize the matrices at the currenter pointer
            Y[p], Yb[p], I, Ib = left_decomp(M[p], Mb[p], chi_max=chi_max, timing=debug, method=method)

            M[p+1] = ncon([I,Z[p+1]], [[-1,1],[1,-2,-3]])
            Mb[p+1] = ncon([Ib,Zb[p+1]], [[-1,1],[1,-2,-3]])

            # Construct L[p+1]
            L[p+1] = ncon([L[p], W[p], Yb[p], Y[p]], [[3,1,5],[3,-1,2,4],[1,2,-2],[5,4,-3]])

            for i in range(NumNA):
                LNA[i][p+1] = ncon([LNA[i][p], MNb[i][p], Y[p]], [[1,2],[1,3,-1],[2,3,-2]])
                LNAb[i][p+1] = ncon([LNAb[i][p], Yb[p], MN[i][p]], [[1,2],[1,3,-1],[2,3,-2]])
        
            ##### display energy
            if dispon == 2:
                print('Sweep: {} of {}, Loc: {},Energy: {:.3f}'.format(k, numsweeps, p, Ekeep[-1]))

        # Set Y[-1]
        Y[-1] = M[-1]
        Yb[-1] = Mb[-1]
        
        ###### Optimization sweep: right-to-left
        for p in range(Nsites-1,0,-1):

            # Optimize at this step
            if updateon:
                E, M[p], Mb[p] = eigLR(L[p], R[p], W[p], M[p], Mb[p], which = which, tol=tol,
                    normalize_against = [(LNA[i][p],RNA[i][p],MNb[i][p],LNAb[i][p],RNAb[i][p],MN[i][p],Namp[i]) for i in range(NumNA)])
                Ekeep = np.append(Ekeep, E)

            # Move the pointer one site to the left, and right-normalize the matrices at the currenter pointer
            Z[p], Zb[p], I, Ib = right_decomp(M[p], Mb[p], chi_max=chi_max, timing=debug, method=method)
            M[p-1] = ncon([Y[p-1],I], [[-1,-2,1],[1,-3]])
            Mb[p-1] = ncon([Yb[p-1],Ib], [[-1,-2,1],[1,-3]])

            # Construct R[p-1]. The indices of R is: left'' - left - left'
            R[p-1] = ncon([R[p], W[p], Zb[p], Z[p]],[[3,1,5],[-1,3,2,4],[-2,2,1],[-3,4,5]])

            for i in range(NumNA):
                RNA[i][p-1] = ncon([RNA[i][p], MNb[i][p], Z[p]], [[1,2],[-1,3,1],[-2,3,2]])
                RNAb[i][p-1] = ncon([RNAb[i][p], Zb[p], MN[i][p]], [[1,2],[-1,3,1],[-2,3,2]])
        
            ##### display energy
            if dispon == 2:
                print('Sweep: {} of {}, Loc: {},Energy: {:.3f}'.format(k, numsweeps, p, Ekeep[-1]))

        # Set Z[0]
        Z[0] = M[0]
        Zb[0] = Mb[0]
        
        # Calculate <H^2>-<H>^2
        RR = np.ones((1,1,1,1))
        for p in range(Nsites-1,-1,-1):
            RR = ncon([Zb[p],RR,W[p],W[p],Z[p]], [[-3,2,1],[5,3,1,6],[-1,5,2,4],[-2,3,4,7],[-4,7,6]])

        Hdif = RR.flatten()[0]-Ekeep[-1]**2
        Hdifs = np.append(Hdifs, Hdif)

        if dispon >= 1:

            """
            # Calculate <1>
            R0 = np.ones((1,1))
            for p in range(Nsites-1,-1,-1):
                R0 = ncon([R0,Z[p],Zb[p]], [[1,2],[-1,3,1],[-2,3,2]])
            norm = R0.flatten()[0]
            """

            print('Sweep: {} of {}, Energy: {:.3f}, H dif: {}, Bond dim: {}, tol: {}'.format(k, numsweeps, Ekeep[-1], Hdif, chi_max, tol))

        cut = max(tol, np.finfo(float).eps) * 10
        # Early termination if converged
        if np.abs(np.std(Ekeep[-2*Nsites:])) < cut and np.abs(Hdif) < cut:
            print("Converged")
            k = numsweeps+1

        k += 1
            
    return Ekeep, Hdifs, Y, Yb, Z, Zb