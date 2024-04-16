import numpy as np
from numpy import linalg as LA
from scipy import linalg as sLA
from scipy import sparse
from ncon import ncon
from time import time

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
            log_write("Timestamp {}: {:.2f}s".format(msg, time()-t1))

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
    rho = ncon([M, Mb], [[-1,-2,1],[-3,-4,1]]).reshape(dL*dS, dL*dS)

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
        log_write("Schur sorted called")

    L = np.shape(M)[0]
    T, Z = sLA.schur(M)

    if doprint:
        log_write("Bare decomposition done")

    if chi <= 0 or L <= chi:
        return T, np.zeros((L,0)), np.zeros((0,0)), Z, np.zeros((L,0))
    args = np.argsort(-np.abs(np.diag(T)))

    if doprint:
        log_write("Entering permutation")

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
        log_write("Permutation done")

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


<<<<<<< Updated upstream
def eigLR (L, R, M, A, Ab, which = 'SR'):
    Ham = ncon([L,R,M],[[1,-4,-1],[2,-6,-3],[1,2,-5,-2]])
    Ham = Ham.reshape((np.prod(np.shape(Ham)[:3]),-1))
=======
def eigLR (L, R, M, A, Ab, which = 'SR', use_sparse = True, tol = 0, normalize_against = [], log_write = print):

    t1 = time()

    Ham = ncon([L,R,M],[[1,-1,-4],[2,-3,-6],[1,2,-2,-5]])
    log_write(f"In eigLR, bond dimensions: {np.shape(Ham)[:3]}...", end=" ")
    dim = np.array(np.shape(Ham)[:3]).prod()
    dim = dim.item()
    Ham = np.reshape(Ham, (dim,dim))

    for norm_tuple in normalize_against:
        Ln,Rn,Mnb,Lnb,Rnb,Mn,amp = norm_tuple
        Hb = ncon([Ln,Rn,Mnb],[[1,-1],[2,-3],[1,-2,2]])
        H = ncon([Lnb,Rnb,Mn],[[-1,1],[-3,2],[1,-2,2]])
        Ham += amp * np.tensordot(H.flatten(), Hb.flatten(), axes=0)
>>>>>>> Stashed changes

    def select_arg (w):
        if which == "SR":
            return np.argsort(np.real(w))[0]
        elif which == "SM":
            return np.argsort(np.abs(w))[0]
        elif isinstance(which, complex):
            return np.argsort(np.abs(w-which))[0]
        else:
            raise Exception("Invalid instance which = {}".format(which))

    if False:

<<<<<<< Updated upstream
        Ham = sparse.csr_matrix(Ham)
        print(Ham.count_nonzero()/np.size(Ham.toarray()))
        print("Eigen solving step 1")
        w, v = sparse.linalg.eigs(Ham, k=1, which="SM", v0=A.flatten(), maxiter=10000)
=======
        if not use_sparse:
            raise Exception("This exception jumps the code to the non-sparse method")
        
        if not isinstance(Ham, numpy.ndarray):
            Ham = Ham.get()

        log_write(f"Using sparse algorithm...", end=" ")

        Hspr = sparse.csr_matrix(Ham)
        #sparse_ratio = Hspr.count_nonzero()/np.size(Hspr.toarray())
        #print("Using sparse: "+str(sparse_ratio))
        #print("Eigen solving step 1")
        if isinstance(which, str):
            w, v = sparse.linalg.eigs(Hspr, k=1, which=which, v0=A.flatten(), maxiter=1000, tol=tol)
        elif isinstance(which, complex):
            #print(f"EigLR which = {which}")
            w, v = sparse.linalg.eigs(Hspr, k=1, sigma=which, which="LM", v0=A.flatten(), maxiter=1000, tol=tol)
        else:
            raise Exception()
>>>>>>> Stashed changes
        w = w[0]
        v = v[:,0]
        print("Eigen solving step 2, w = {}".format(w))
        w1, vL = sparse.linalg.eigs(Ham.transpose(), k=1, sigma=w, which="LM", v0=Ab.flatten())
        w1 = w1[0]
        vL = vL[:,0]
<<<<<<< Updated upstream
        print("Eigen solving step 3, w1 = {}".format(w1))

    else:
=======
        #log_write("Eigen solving step 3, w1 = {}".format(w1))

    except Exception as e:

        log_write(f"{e}, Using non-sparse algorithm...", end=" ")
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
    if np.abs(w-w1) > 1e-6:
        print("Fails to converge to the same eigenvalue: right eigenvalue {:.8f}, left eigenvalue {:.8f}".format(w, w1))
=======
    if np.abs(w-w1) > max(10*tol, 1e-14):
        log_write(f"L-R Eigenvalue error {np.abs(w-w1)} / tol {tol}...", end=" ")
>>>>>>> Stashed changes
    
    # Normalize the left- and right- eigenvectors
    v, vL = unitize(v, vL)

<<<<<<< Updated upstream
=======
    log_write(f"Done in {time()-t1}s")

>>>>>>> Stashed changes
    return (w+w1)/2, np.reshape(v, np.shape(A)), np.reshape(vL, np.shape(Ab))

def left_decomp_lrrho (M, Mb, chi_max = 0, timing = False):

    if timing:
        t1 = time()
        log_write("In decomp_lrrho")

    if len(np.shape(M)) == 2:
        M = M.reshape(1,np.shape(M)[0], np.shape(M)[1])

    if np.shape(M) != np.shape(Mb):
        raise Exception("The shape of M and Mb must be identical!")
        
    dL, dS, _ = np.shape(M)
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
        log_write("lrrho cost {}s".format(time()-t1))

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