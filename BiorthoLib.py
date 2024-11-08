from Config import *
from scipy import linalg as sLA
from scipy import sparse
from time import time
import numpy
from numbers import Number
import traceback

def left_decomp (M, Mb, chi_max = 0, timing = False, method = "biortho"):
    if method == "biortho":
        return left_decomp_biortho(M, Mb, chi_max=chi_max, timing=timing)
    elif method == "lrrho":
        return left_decomp_lrrho(M, Mb, chi_max=chi_max, timing=timing)
    
def right_decomp (M, Mb, chi_max = 0, timing = False, method = "biortho"):

    M = M.transpose()
    Mb = Mb.transpose()
    Y, Yb, I, Ib = left_decomp(M, Mb, chi_max=chi_max, timing=timing, method=method)
    return Y.transpose(), Yb.transpose(), I.transpose(), Ib.transpose()

"""
Given two MPS blocks M (left - physical - right) and Mb (left' - physical' - right')
Returns Y (left, physical, temp), Yb (left' - physical' - temp'), eta (temp - right), etab (temp' - right')
such that Y contracted with Yb on left and physical indices yields identity.

The process is done by the two-step partial diagonalization of the density matrix rho = (M Mb contracted on right index), 
as described in 2401.15000. This brings M and Mb into "left-canonical" form, which eta and etab are to be multiplied onto 
the site on the right.
"""

def left_decomp_biortho (M, Mb, chi_max = 0, unitarize = False, timing = False):

    t1 = time()

    def timestamp(msg):
        if timing:
            print("Timestamp {}: {:.2f}s".format(msg, time()-t1))

    timestamp("Begin decomposition")


    if np.shape(M) != np.shape(Mb):
        raise Exception("The shape of M and Mb must be identical!")
    
    # Deal with the case where M is the lefter-most tensor
    leftmost = False
    if len(np.shape(M)) == 2:
        M = M.reshape(1,np.shape(M)[0], np.shape(M)[1])
        Mb = Mb.reshape(1,np.shape(M)[0], np.shape(M)[1])
        leftmost = True

    # Construct Density Matrix
    dL, dS, _ = np.shape(M)
    rho = ncon([M, Mb], [[-1,-2,1],[-3,-4,1]]).reshape((dL*dS, dL*dS))

    timestamp("Rho Construction")

    # If rho is normal, just do diagonalization
    rhoH = rho.conj().T
    if np.allclose(rho@rhoH-rhoH@rho, 0):

        w, u = LA.eigh(rho+rhoH)
        if 0 < chi_max < dL*dS:
            args = np.argsort(-np.abs(w))[:chi_max]
            u = u[:,args]
        Ys = u
        Ysb = u.conj()

    # Generic case
    else: 
        # Two-step block diagonalization
        A, C, D, Ss, Sd = schur_sorted(rho, chi_max, doprint = timing) # Schur decomposition

        timestamp("Schur Decomposition")

        # If truncation happened, do Roth removal
        Ys = Ss
        Ysb = Ss.conj()
        if np.size(D) > 0:
            X = sLA.solve_sylvester(A, -C, D)
            Ysb += Sd.conj() @ X.T

        timestamp("Roth Removal")

        # Bi-orthonormalization
        Ys, Ysb = GSbiortho(Ys, Ysb)

        timestamp("Biorthonormalization")

        # Unitarization
        if unitarize:
            U,_,_ = LA.svd(Ys)
            Ys = U
            Ysb = U.conj()

    Y = Ys.reshape(dL, dS, -1)
    Yb = Ysb.reshape(dL, dS, -1)

    I = ncon([Yb,M],[[1,2,-1],[1,2,-2]])
    Ib = ncon([Y,Mb],[[1,2,-1],[1,2,-2]])

    timestamp("Final")

    return Y, Yb, I, Ib

# Make v1@v2=1 and norm(v1)=norm(v2)
def unitize (v1, v2, return_ratios = False):
    ipsr = 1/np.sqrt(np.inner(v1, v2))
    ratio = np.sqrt(LA.norm(v1)/LA.norm(v2))

    if return_ratios:
        return v1*ipsr/ratio, v2*ipsr*ratio, ratio/ipsr, 1/(ipsr*ratio)
    else:
        return v1*ipsr/ratio, v2*ipsr*ratio
    
# Gram-Shimidt Biorthonormalization
# The goal is to make Yb.T@Y = eye
def GSbiortho (Y, Yb):
    L = np.shape(Y)[1]
    for i in range(L):
        # Orthogonalize the i-th vector with all those before it
        for j in range(i):
            Yb[:,i] -= Yb[:,j] * np.inner(Yb[:,i],Y[:,j])/np.inner(Yb[:,j],Y[:,j])
            Y[:,i] -= Y[:,j] * np.inner(Yb[:,j],Y[:,i])/np.inner(Yb[:,j],Y[:,j])
        # Normalize the i-th vector
        Yb[:,i], Y[:,i] = unitize(Yb[:,i], Y[:,i])
    return Y, Yb

# Get the null space of a matrix M
# Returns Y and Yb, with M@Y=Yb@M=0, and Yb@Y=Id
def nullspace (M):
    U, S, V = LA.svd(M)
    cut = (1e-16)*np.max(np.shape(M))
    U = U[:,S<cut]
    V = V[S<cut,:]
    T = V @ U
    E, C = LA.eig(T)
    Ep = np.diag(E**(-1/2))
    return U@C@Ep, Ep@LA.inv(C)@V


# Returns A, C, D, Ss, Sd
def schur_sorted (M, chi, doprint = False):

    if doprint:
        print("Schur sorted called")

    L = np.shape(M)[0]
    T, Z = sLA.schur(M)

    if doprint:
        print("Bare decomposition done")

    if chi <= 0 or L <= chi:
        return T, np.zeros((L,0)), np.zeros((0,0)), Z, np.zeros((L,0))
    args = np.argsort(-np.abs(np.diag(T)))

    if doprint:
        print("Entering permutation")

    def permute (r1, r2):
        if r1 == r2:
            return
        T[r1,:], T[r2,:] = T[r2,:], T[r1,:]
        T[:,r1], T[:,r2] = T[:,r2], T[:,r1]
        Z[:,r1], Z[:,r2] = Z[:,r2], Z[:,r1]

    # Exchange eigenvalues at position r and r+1
    def exchange_at (r):
        a = T[r,r]
        b = T[r+1, r+1]
        c = T[r,r+1]
        x = c/(a-b)
        y = np.sqrt(1+np.abs(x)**2)
        Q = np.array([[-np.conj(x), 1],[1, x]])/y
        Qd = Q.conj().T
        T[:,r:r+2] = T[:,r:r+2] @ Qd
        Z[r:r+2,:] = Q @ Z[r:r+2,:]
        Z[:,r:r+2] = Z[:,r:r+2] @ Qd
    
    # Iteratively, find the largest eigenvalue in the lower half and the smallest eigenvalue in the upper half,
    # permute them to adjacent positions, and exchange them.
    # We generate two pointers that travel from the two ends of [0,L-1]
    pL = 0
    pR = L-1
    while pL < chi:
        # Find pL such that the eigenvalue at pL is a small eigenvalue
        if args[pL] < chi:
            pL += 1
            continue
        # Find pR such that the eigenvalue at pR is a large eigenvalue
        while True:
            if args[pR] < chi:
                break
            else:
                if pR <= chi:
                    raise Exception("Unpaired out-of-order eigenvalue!")
                pR -= 1
                continue
        # Permute pL and pR
        permute(pL, chi-1)
        permute(pR, chi)
        exchange_at(chi-1)
        permute(pL, chi-1)
        permute(pR, chi)
        pL += 1
        pR -= 1

    
    if doprint:
        print("Permutation done")

    return T[:chi,:chi], T[chi:,chi:], T[:chi,chi:], Z[:,:chi], Z[:,chi:]

"""
Similar to left_decomp

Input:
M (left - physical - right)
Mb (left' - physical' - right')

Output:
Z (temp - physical - right)
Zb (temp' - physical' - right')
eta (left - temp)
etab (left' - temp')
"""


def eigLR (L, R, M, A, Ab, which = 'SR', use_sparse = True, tol = 0, normalize_against = [], log_write = print, ncv = 50, library = "ARPACK"):

    t1 = time()

    Ham = ncon([L,R,M],[[1,-1,-4],[2,-3,-6],[1,2,-2,-5]])
    log_write(f"In eigLR, bond dimensions: {np.shape(Ham)[:3]}...", end=" ")
    dim = np.array(np.shape(Ham)[:3]).prod()
    dim = dim.item()
    Ham = np.reshape(Ham, (dim,dim))
    ncv = min(ncv, dim)

    which_is_number = isinstance(which, (Number, np.number))

    for norm_tuple in normalize_against:
        Ln,Rn,Mnb,Lnb,Rnb,Mn,amp = norm_tuple
        Hb = ncon([Ln,Rn,Mnb],[[1,-1],[2,-3],[1,-2,2]])
        H = ncon([Lnb,Rnb,Mn],[[-1,1],[-3,2],[1,-2,2]])
        Ham += amp * np.tensordot(H.flatten(), Hb.flatten(), axes=0)

    def select_arg (w):
        if which == "SR":
            return np.argsort(np.real(w))[0]
        elif which == "SM":
            return np.argsort(np.abs(w))[0]
        elif which == "LR":
            return np.argsort(np.real(w))[-1]
        elif which_is_number:
            return np.argsort(np.abs(w-which))[0]
        else:
            raise Exception("Invalid instance which = {}".format(which))
        
    select_arg(np.array([0]))

    def sparse_eig (H, which, v0, tol):

        tncv = ncv
    
        while True:
            try:
                if isinstance(which, str):
                    return sparse.linalg.eigs(H, k=1, which=which, v0=v0, maxiter=10000, tol=tol, ncv=tncv)
                else:
                    return sparse.linalg.eigs(H, k=1, sigma=which, which="LM", v0=v0, maxiter=10000, tol=tol, ncv=tncv)
            except sparse.linalg.ArpackNoConvergence:
                tncv += ncv
            if tncv >= dim / 5:
                raise Exception(f"No convergence at ncv = {tncv}")

    try:

        if not use_sparse:
            raise Exception("This exception jumps the code to the non-sparse method")
        
        if not isinstance(Ham, numpy.ndarray):
            Ham = Ham.get()

        log_write(f"Using sparse algorithm...", end=" ")

        Hspr = sparse.csr_matrix(Ham)
        #sparse_ratio = Hspr.count_nonzero()/np.size(Hspr.toarray())
        #log_write("Using sparse: "+str(sparse_ratio))
        #log_write("Eigen solving step 1")
        w, v = sparse_eig(Hspr, which, A.flatten(), tol)
        w = w[0]
        v = v[:,0]
        #log_write("Eigen solving step 2, w = {}".format(w))
        w1, vL = sparse.linalg.eigs(Hspr.transpose(), k=1, sigma=w, which="LM", v0=Ab.flatten(), tol=tol, ncv=ncv)
        w1 = w1[0]
        vL = vL[:,0]
        #log_write("Eigen solving step 3, w1 = {}".format(w1))

    except Exception as e:

        print(e)
        print(traceback.format_exc())
        log_write("Using non-sparse algorithm...", end=" ")

        #log_write("Eigen solving step 1")
        w, v = LA.eig(Ham)
        a = select_arg(w)
        w = w[a]
        v = v[:,a]
        #log_write("Eigen solving step 2, w = {}".format(w))
        w1, vL = LA.eig(Ham.transpose())
        a = select_arg(w1)
        w1 = w1[a]
        vL = vL[:,a]
        #log_write("Eigen solving step 3, w1 = {}".format(w1))

    if np.abs(w-w1) > max(10*tol, 1e-14):
        log_write(f"L-R Eigenvalue error {np.abs(w-w1)} / tol {tol}...", end=" ")
    
    # Normalize the left- and right- eigenvectors
    v, vL = unitize(v, vL)

    log_write(f"Done in {time()-t1}s")

    return (w+w1)/2, np.reshape(v, np.shape(A)), np.reshape(vL, np.shape(Ab))

def left_decomp_lrrho (M, Mb, chi_max = 0, timing = False):

    if timing:
        t1 = time()
        print("In decomp_lrrho")

    if len(np.shape(M)) == 2:
        M = M.reshape(1,np.shape(M)[0], np.shape(M)[1])

    if np.shape(M) != np.shape(Mb):
        raise Exception("The shape of M and Mb must be identical!")
        
    dL, dS, dR = np.shape(M)
    M = M.reshape(dL*dS,-1)
    Mb = Mb.reshape(dL*dS,-1)
    rho = (M@M.conj().T + Mb.conj()@Mb.T)/2
    S, U = LA.eigh(rho)
    if 0 < chi_max < np.size(S):
        args = np.argsort(np.abs(S))[-chi_max:]
        S = S[args]
        U = U[:,args]

    I = U.conj().T @ M
    Ib = U.T @ Mb

    U = U.reshape(dL, dS, -1)

    if timing:
        print("lrrho cost {}s".format(time()-t1))

    return U, U.conj(), I, Ib
    







# Gram-Shimidt Biorthonormalization
# The goal is to make Yb@Y = eye
# All transforms are carried on to a matrix A so that Yb@A@Y remains unchanged
def GSbiortho_old (Y, Yb, A):
    L = np.shape(Y)[0]
    for i in range(L):
        # Orthogonalize the i-th vector with all those before it
        for j in range(i):
            tmp = np.inner(Yb[:,i],Y[j,:])/np.inner(Yb[:,j],Y[j,:])
            Yb[:,i] -= Yb[:,j] * tmp
            A[j,:] += A[i,:] * tmp
            tmp = np.inner(Yb[:,j],Y[i,:])/np.inner(Yb[:,j],Y[j,:])
            Y[i,:] -= Y[j,:] * tmp
            A[:,j] += A[:,i] * tmp
        # Normalize the i-th vector
        Yb[:,i], Y[i,:], r1, r2 = unitize(Yb[:,i], Y[i,:], return_ratios=True)
        A[i,:] *= r1
        A[:,i] *= r2
    return Y, Yb, A