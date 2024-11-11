using Dates
using ITensors
import Arpack:eigs as arpack_eigs
include("Config.jl")

function decomp(M::ITensor, Mb::ITensor, link::Index; chi_max = 0, timing = false, method::DM_Method = LR)
    if method == BB
        return decomp_biortho(M, Mb, link; chi_max=chi_max, timing=timing)
    elseif method == LR
        return decomp_lrrho(M, Mb, link; chi_max=chi_max, timing=timing)
    else
        throw(ArgumentError("Unrecognized DM_Method = $method."))
    end
end

function decomp(M::ITensor, Mb::ITensor, linkind::Int; chi_max = 0, timing = false, method::DM_Method = LR)
    return decomp(M, Mb, inds(M)[linkind>0 ? linkind : end-linkind]; chi_max=chi_max, timing=timing, method=method)
end

"""
decomp_biortho(M::ITensor, Mb::ITensor, link::Index, linkb::Index; chi_max = 0, unitarize = false, timing = false)

Given two MPS blocks M (left - physical - right) and Mb (left' - physical' - right')
Returns Y (left, physical, temp), Yb (left' - physical' - temp'), eta (temp - right), etab (temp' - right')
such that Y contracted with Yb on left and physical indices yields identity.

The process is done by the two-step partial diagonalization of the density matrix rho = (M Mb contracted on right index), 
as described in 2401.15000. This brings M and Mb into "left-canonical" form, which eta and etab are to be multiplied onto 
the site on the right.
"""
function decomp_biortho(M::ITensor, Mb::ITensor, link::Index; chi_max = 0, unitarize = false, timing = false)

    if timing
        t1 = now()
    end

    function timestamp(msg)
        if timing
            fprintln("Timestamp :: $(Dates.value(now()-t1)/1000)s :: $(msg)")
        end
    end

    timestamp("Begin decomposition")

    if size(M) != size(Mb)
        throw(ArgumentError("The shape of M and Mb must be identical!"))
    end

    # Construct Density Matrix
    dL, dS = size(M)[1:2]
    rho = reshape(ncon([M, Mb], [[-1,-2,1],[-3,-4,1]]), (dL*dS, dL*dS))

    timestamp("Rho Construction")

    # If rho is normal, just do diagonalization
    rhoH = rho'
    if isapprox(rho*rhoH, rhoH*rho; rtol=1e-6)

        rho_eig = eigen(rho+rhoH)
        u = rho_eig.vectors
        if 0 < chi_max < dL*dS
            u = u[:,sortperm(-abs.(rho_eig.values))[1:chi_max]]
        end
        Ys = u
        Ysb = conj.(u)

    # Generic case
    else

        # Two-step block diagonalization
        A, C, D, Ss, Sd = schur_sorted(rho, chi_max; doprint = timing) # Schur decomposition

        timestamp("Schur Decomposition")

        # If truncation happened, do Roth removal
        Ys = Ss
        Ysb = conj.(Ss)
        if length(D) > 0
            X = sLA.solve_sylvester(A, -C, D)
            Ysb += conj.(Sd) * transpose(X)
        end

        timestamp("Roth Removal")

        # Bi-orthonormalization
        Ys, Ysb = GSbiortho(Ys, Ysb)

        timestamp("Biorthonormalization")

        # Unitarization
        if unitarize
            Ys = svd(Ys).U
            Ysb = conj.(U)
        end

    end

    Y = ireshape(Ys, (dL, dS, -1))
    Yb = ireshape(Ysb, (dL, dS, -1))

    I = ncon([Yb,M],[[1,2,-1],[1,2,-2]])
    Ib = ncon([Y,Mb],[[1,2,-1],[1,2,-2]])

    timestamp("Final")

    return Y, Yb, I, Ib

end

# Make v1@v2=1 and norm(v1)=norm(v2)
function unitize(v1, v2; return_ratios = false)

    # Notice that v1 is already the left eigenvector, so it NEED NOT BE CONJUGATED
    # DO NOT USE dot, because that would conjugate the first vector.
    ipsr = 1/sqrt(plaindot(v1,v2)) 
    ratio = sqrt(norm(v1)/norm(v2))

    if return_ratios
        return v1.*(ipsr/ratio), v2.*(ipsr*ratio), ratio/ipsr, 1/(ipsr*ratio)
    else
        return v1.*(ipsr/ratio), v2.*(ipsr*ratio)
    end

end
    
# Gram-Shimidt Biorthonormalization
# The goal is to make Yb.T@Y = eye
function GSbiortho(Y, Yb)
    for i = 1:size(Y)[2]
        # Orthogonalize the i-th vector with all those before it
        for j = 1:i-1
            Yb[:,i] -= Yb[:,j] .* (plaindot(Yb[:,i],Y[:,j])/plaindot(Yb[:,j],Y[:,j]))
            Y[:,i] -= Y[:,j] .* (plaindot(Yb[:,j],Y[:,i])/plaindot(Yb[:,j],Y[:,j]))
        end
        # Normalize the i-th vector
        Yb[:,i], Y[:,i] = unitize(Yb[:,i], Y[:,i])
    end
    return Y, Yb
end

# Get the null space of a matrix M
# Returns Y and Yb, with M@Y=Yb@M=0, and Yb@Y=Id
function nullspace(M)
    Msvd = svd(M)
    cut = (1e-16)*max(size(M)...)
    U = Msvd.U[:,Msvd.S.<cut]
    V = Msvd.V[Msvd.S.<cut,:]
    E, C = eigen(V * U)
    Ep = diagm(E.^(-1/2))
    return U*C*Ep, Ep*inv(C)*V
end

# Returns A, C, D, Ss, Sd
function schur_sorted(M, chi; doprint = false)

    if doprint
        fprint("Schur sorted called")
    end

    L = size(M)[1]
    T, Z = schur(M)

    if doprint
        fprint("Bare decomposition done")
    end

    if chi <= 0 || L <= chiz
        return T, zeros((L,0)), zeros((0,0)), Z, zeros((L,0))
    end

    args = sortperm(-abs.(diagm(T)))

    if doprint
        fprint("Entering permutation")
    end

    function permute(r1, r2)
        if r1 == r2
            return
        end
        T[r1,:], T[r2,:] = T[r2,:], T[r1,:]
        T[:,r1], T[:,r2] = T[:,r2], T[:,r1]
        Z[:,r1], Z[:,r2] = Z[:,r2], Z[:,r1]
    end

    # Exchange eigenvalues at position r and r+1
    function exchange_at(r)
        a = T[r, r]
        b = T[r+1, r+1]
        c = T[r, r+1]
        x = c/(a-b)
        y = sqrt(1+abs(x)^2)
        Q = [-conj(x) 1; 1 x]./y
        Qd = Q'
        T[:,r:r+1] = T[:,r:r+1] * Qd
        Z[r:r+1,:] = Q * Z[r:r+1,:]
        Z[:,r:r+1] = Z[:,r:r+1] * Qd
    end
    
    # Iteratively, find the largest eigenvalue in the lower half and the smallest eigenvalue in the upper half,
    # permute them to adjacent positions, and exchange them.
    # We generate two pointers that travel from the two ends of [1,L]
    pL = 1
    pR = L
    while pL <= chi
        # Find pL such that the eigenvalue at pL is a small eigenvalue
        if args[pL] <= chi
            pL += 1
            continue
        end
        # Find pR such that the eigenvalue at pR is a large eigenvalue
        while true
            if args[pR] <= chi
                break
            else
                if pR <= chi
                    throw(ErrorException("Unpaired out-of-order eigenvalue!"))
                end
                pR -= 1
                continue
            end
        end
        # Permute pL and pR
        permute(pL, chi)
        permute(pR, chi+1)
        exchange_at(chi)
        permute(pL, chi)
        permute(pR, chi+1)
        pL += 1
        pR -= 1
    end
    
    if doprint
        print("Permutation done")
    end

    return T[1:chi,1:chi], T[chi+1:end,chi+1:end], T[1:chi,chi+1:end], Z[:,1:chi], Z[:,chi+1:end]

end

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

function doeig(Ham, v0; sigma=0, tol=0, ncv0=50, use_sparse=true)

    Hdim = size(Ham)[1]

    ncv = ncv0

    try

        if !use_sparse
            throw(ErrorException("This exception jumps the code to the non-sparse method"))
        end

        while true

            try
                fprintln("Trying sprase diagonalization with ncv = $ncv")
                eigvalues, eigvectors = arpack_eigs(Ham; nev=1, which=:LM, sigma=sigma, tol=tol, v0=v0, ncv=ncv)
                fprintln("Converged eigenvalue: ", eigvalues[1])
                return eigvalues[1], eigvectors[:,1]
            catch e
                if isa(e, ErrorException) && occursin("No convergence", e.msg)
                    # Handle non-convergence (e.g., adjust parameters or retry)
                    fprintln("Warning: Eigenvalue computation did not converge at ncv = $ncv.")
                    if ncv < Hdim/2
                        ncv += ncv0
                        fprintln("Redoing with ncv = $ncv.")
                    else
                        throw(ErrorException("Maximal ncv bar reached."))
                    end
                else
                    rethrow(e)  # Rethrow if it's a different error
                end
            end

        end

    catch e

        if hasfield(typeof(e), :msg)
            fprintln(e.msg)
        else
            fprintln(e)
        end
        fprintln("Switching to dense algorithm.")

        w, v = eigen(Ham)
        arg = argmin(abs.(w.-sigma))
        return w[arg], v[:,arg]

    end

end

function eigLR(L::ITensor, R::ITensor, M::ITensor, A::ITensor, Ab::ITensor; sigma::ComplexF64 = ComplexF64(0), use_sparse = true, tol = 0, normalize_against = [], ncv0 = 50, timing = false)

    if timing
        t1 = now()
    end

    function timestamp(msg)
        if timing
            fprintln("Timestamp :: $(Dates.value(now()-t1)/1000)s :: $(msg)")
        end
    end

    Ham = L*M*R
    bonds = [i for i in inds(Ham) if plev(i)==0]
    comb = combiner(bonds...)
    mind = inds(comb)[1]
    oind = inds(comb)[2:end]
    Hdim = dim(mind)
    Ham = Ham*comb*comb'
    timestamp("In eigLR, bond dimensions: $bonds")
    ncv0 = min(ncv0, Hdim)

    for norm_tuple in normalize_against
        Ln,Rn,Mnb,Lnb,Rnb,Mn,amp = norm_tuple
        Ham += amp * vec(Array(Lnb*Mn*Rnb, oind'...)) * transpose(vec(Array(Ln*Mnb*Rn, oind...)))
    end

    Ham = Array(Ham, mind', mind)
    w, v = doeig(Ham, vec(Array(A, oind...));sigma=sigma, tol=tol, ncv0=ncv0, use_sparse=use_sparse)
    w1, vL = doeig(transpose(Ham), vec(Array(Ab, oind'...)); sigma=w, tol=tol, ncv0=ncv0, use_sparse=use_sparse)
    # Right here, the second diagonalization is to find the right eigenvector of HT.
    # The eigenvalue of H^T should be the same as the eigenvalue of H.
    # Ab should contain the left eigenvector, which is the transpose of the right eigenvector of H^T.

    if abs(w-w1) > max(10*tol, 1e-14)
        fprintln("L-R Eigenvalue error $(abs(w-w1)) / tol $tol...")
    end
    
    # Normalize the left- and right- eigenvectors
    v, vL = unitize(v, vL)

    if timing
        timestamp("Done.")
    end

    return (w+w1)/2, ITensor(v, mind)*comb, ITensor(vL, mind')*comb'

end

"""


Given tensors M(link, others) and Mb(linkb, others'), 

"""
function decomp_lrrho(M::ITensor, Mb::ITensor, link::Index; chi_max = 0, timing = false)

    if timing
        t1 = now()
        fprintln("In decomp_lrrho")
    end

    idm = inds(M)

    if !(link in idm)
        throw(ArgumentError("link !in inds(M)!\ninds(M)=$(idm), link=$link"))
    end

    if idm' != inds(Mb)
        throw(ArgumentError("inds(M)' != inds(Mb)! inds(M) = $idm, inds(Mb) = $(inds(Mb))"))
    end
    
    # Combine the non-link indices into one index
    Mcomb = combiner([i for i in idm if i != link])
    Mci = inds(Mcomb)[1]
    msize = dim(Mci)
    M *= Mcomb
    Mb *= Mcomb'
    mind = Index(msize, "matind")

    # Construct density matrix as |psi_R><psi_R|+|psi_L><psi_L|. A factor 1/2 is neglected since we only need the eigenvectors.
    rho = Array((M*delta(Mci,mind))*(conj(M)*delta(Mci,mind')) + (Mb*delta(Mci',mind'))*(conj(Mb)*delta(Mci',mind)), mind', mind)
    S, U = eigen(rho)
    if 0 < chi_max < length(S)
        args = sortperm(-abs.(S))[1:chi_max]
        # S = S[args]
        U = U[:,args]
    end

    # Create a new link
    newlink = Index(size(U)[2], tags(link))

    U = ITensor(U, Mci, newlink)
    I = conj(U)*M
    Ib = U'*Mb

    if timing
        fprintln("lrrho cost $(Dates.value(now()-t1)/1000)s")
    end

    return U*Mcomb, conj(U)'*Mcomb', I, Ib
    
end







# Gram-Shimidt Biorthonormalization
# The goal is to make Yb@Y = eye
# All transforms are carried on to a matrix A so that Yb@A@Y remains unchanged
# def GSbiortho_old (Y, Yb, A):
#     L = size(Y)[0]
#     for i in range(L):
#         # Orthogonalize the i-th vector with all those before it
#         for j in range(i):
#             tmp = np.inner(Yb[:,i],Y[j,:])/np.inner(Yb[:,j],Y[j,:])
#             Yb[:,i] -= Yb[:,j] * tmp
#             A[j,:] += A[i,:] * tmp
#             tmp = np.inner(Yb[:,j],Y[i,:])/np.inner(Yb[:,j],Y[j,:])
#             Y[i,:] -= Y[j,:] * tmp
#             A[:,j] += A[:,i] * tmp
#         # Normalize the i-th vector
#         Yb[:,i], Y[i,:], r1, r2 = unitize(Yb[:,i], Y[i,:], return_ratios=True)
#         A[i,:] *= r1
#         A[:,i] *= r2
#     return Y, Yb, A