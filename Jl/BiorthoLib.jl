using Dates
using ITensors
import Arpack:eigs as arpack_eigs
using MatrixEquations: sylvc
using LinearAlgebra

@doc raw"""
    decomp(M::ITensor, Mb::ITensor, link::Index; chi_max::Int = 0, timing::Bool = false, method::DM_Method = LR)

Does left/right-normaliztion of an MPS block at a given position. Given ITensors M, with indices (link, others),
and Mb, with indices (link', others'), returns ITensors Y (newlink, others), Yb (newlink', others'), such that
$|i\rangle$=`Y[i,:]` and $\langle i|$=`Yb[i,:]` forms a biorthonormal set. Also returns the transformation matrices
I (link, newlink), Ib (link', newlink'), such that `I*Y` $\approx$ M and `Ib*Yb` $\approx$ Mb.

# Arguments
- `M`: ITensor, the right-vector MPS block to be decomposed. Has indices (link, others).
- `Mb`: ITensor, the left-vector MPS block to be decomposed. M and Mb should have the same shape,
    with the indices of Mb being the primes of the indices of M, i.e., (link', others').
- `link`: Index, the link with respect to which the decomposed blocks are orthogonalized. 
    The link should be in the indices of M, and link' should be in the indices of Mb.
- `chi_max`: Int, maximal bond dimension.
- `timing`: Bool, default false. Whether to print timing information.
- `method`: DM_Method, default LR. The method to use for the decomposition. Options:
    LR (**L**eft- and **R**ight-vector density matrix method, see `?decomp_lrrho`)
    or BB (**B**iorthonormal-**B**lock method, see `?decomp_biortho`).
- `return_newlink`: Bool, default false. Whether to return the newlink index.

# Returns
- `Y`: ITensor, the right-vector MPS block after decomposition. Has indices (newlink, others).
- `Yb`: ITensor, the left-vector MPS block after decomposition. Has indices (newlink', others').
- `I`: ITensor, the matrix that transforms M to Y, has indices (link, newlink).
- `Ib`: ITensor, the matrix that transforms Mb to Yb, has indices (link', newlink').
- `newlink`: Index, returned only if return_newlink=true. The newlink index.

"""
function decomp(M::ITensor, Mb::ITensor, link::Index; chi_max::Int = 0, timing = false, method::DM_Method = LR, return_newlink::Bool=false)
    if method == BB
        return decomp_biortho(M, Mb, link; chi_max=chi_max, timing=timing, return_newlink=return_newlink)
    elseif method == LR
        return decomp_lrrho(M, Mb, link; chi_max=chi_max, timing=timing, return_newlink=return_newlink)
    else
        throw(ArgumentError("Unrecognized DM_Method: $method."))
    end
end

# """
#     decomp(M::ITensor, Mb::ITensor, linkind::Int; chi_max::Int = 0, timing::Bool = false, method::DM_Method = LR)

# Does left/right-normaliztion of an MPS block at a given position. See decomp(M::ITensor, Mb::ITensor, link::Index; chi_max::Int = 0, timing::Bool = false, method::DM_Method = LR)

# # Arguments
# - `linkind`, Int, the index of `link` in `inds(M)`.
#     If positive, `link=inds(M)[linkind]`. If negative, `link=inds(M)[end-linkind]`.
# """
# function decomp(M::ITensor, Mb::ITensor, linkind::Int; chi_max::Int = 0, timing::Bool = false, method::DM_Method = LR, return_newlink::Bool=false)
#     return decomp(M, Mb, inds(M)[linkind>0 ? linkind : end-linkind]; chi_max=chi_max, timing=timing, method=method, return_newlink=return_newlink)
# end

@enum Direction Left Right

"""
    decomp(M::MPS, Mb::MPS, dir::Direction; chi_max::Int=0, open_end::Bool=false, timing::Bool=false, method::DM_Method=LR)

Decompose a MPS into left or right normal form, depending on "dir".

# Arguments
- `M`: MPS, the right-vector MPS to be decomposed.
- `Mb`: MPS, the left-vector MPS to be decomposed.
- `dir`: Direction, the direction of the decomposition. Options: `Left` or `Right`.
- `chi_max`: Int, default 0. The maximal bond dimension.
- `open_end`: Bool, default false. Whether to leave the end open. If true, the last site will be decomposed with an imaginary new link, and the new link will be returned.
- `timing`: Bool, default false. Whether to print timing information.
- `method`: DM_Method, default LR. The method to use for the decomposition. Options:
    LR (**L**eft- and **R**ight-vector density matrix method, see `?decomp_lrrho`)
    or BB (**B**iorthonormal-**B**lock method, see `?decomp_biortho`).

# Returns
- `Y`: MPS, the right-vector MPS after decomposition.
- `Yb`: MPS, the left-vector MPS after decomposition.
- `newlink`: Index, returned only if open_end=true. The new link index.

"""
function decomp(M::MPS, Mb::MPS, dir::Direction; chi_max::Int=0, open_end::Bool=false, timing::Bool=false, method::DM_Method=LR)

    Nsites = length(M)
    sites, links = find_sites_and_links(M)
    Y = ITensor[]
    Yb = ITensor[]

    sweep_indices = (dir==Left) ? (1:Nsites) : (Nsites:-1:1)
    link_offset = (dir==Left) ? 0 : -1
    M_offset = (dir==Left) ? 1 : -1

    for i = 1:Nsites-1
        isweep = sweep_indices[i]
        Mi, Mbi, I, Ib = decomp(M[isweep], Mb[isweep], links[isweep+link_offset]; chi_max=chi_max, timing=timing, method=method)
        M[i+M_offset] *= I
        Mb[i+M_offset] *= Ib
        push!(Y, Mi)
        push!(Yb, Mbi)
    end
    if open_end
        # Append an imaginary new link to the end
        endlink = Index(1, "EndLink")
        endtensor = ITensor(1, endlink)
        endindex = sweep_indices[end]
        Mi, Mbi, _, _, endlink = decomp(M[endindex]*endtensor, Mb[endindex]*endtensor', endlink; chi_max=chi_max, timing=timing, method=method, return_newlink=true)
        push!(Y, Mi)
        push!(Yb, Mbi)
        return MPS(Y[sweep_indices]), MPS(Yb[sweep_indices]), endlink
    else
        return MPS(Y[sweep_indices]), MPS(Yb[sweep_indices])
    end

end

@doc raw"""
    decomp(M::MPS, Mb::MPS, pos::Int; chi_max::Int=0, timing::Bool=false, method::DM_Method=LR)

Decompose a set of MPS blocks, such that [1:pos] are in left normal form, and [pos+1:end] are in right normal form.
"""
function decomp(M::MPS, Mb::MPS, pos::Int; chi_max::Int=0, timing::Bool=false, method::DM_Method=LR)

    Nsites = length(M)
    sites, links = find_sites_and_links(M)
    Y = fill(ITensor(1), Nsites)
    Yb = fill(ITensor(1), Nsites)

    if !(1 <= pos <= Nsites)
        throw(ArgumentError("pos should be in [1,$Nsites]!"))
    end

    for i = 1:pos-1
        Mi, Mbi, I, Ib = decomp(M[i], Mb[i], links[i]; chi_max=chi_max, timing=timing, method=method)
        M[i+1] *= I
        Mb[i+1] *= Ib
        Y[i] = Mi
        Yb[i] = Mbi
    end

    for i = Nsites:-1:pos+1
        Mi, Mbi, I, Ib = decomp(M[i], Mb[i], links[i-1]; chi_max=chi_max, timing=timing, method=method)
        M[i-1] *= I
        Mb[i-1] *= Ib
        Y[i] = Mi
        Yb[i] = Mbi
    end

    Y[pos] = M[pos]
    Yb[pos] = Mb[pos]

    return MPS(Y), MPS(Yb)

end

"""
    function doubleMPSSize(M::MPS, Mb::MPS; chi_max::Int=0, timing::Bool=false, method::DM_Method=LR)

Given a pair of MPS, double the spatial size by combining two copies of them together.
"""
function doubleMPSSize(M::MPS, Mb::MPS; chi_max::Int=0, timing::Bool=false, method::DM_Method=LR)

    Y, Yb = decomp(M, Mb, Left; chi_max=chi_max, timing=timing, method=method)

    L = length(M)
    sites,_ = find_sites_and_links(M)

    newsites = [Index(dim(s), "Site $(i+L)") for (i,s) in enumerate(sites)]
    for i in eachindex(sites)
        M *= delta(sites[i],newsites[i])
        Mb *= delta(sites[i]',newsites[i]')
    end
    Z, Zb = decomp(M, Mb, Right; chi_max=chi_max, timing=timing, method=method)

    N = ITensor[]
    Nb = ITensor[]
    for i = 1:length(Y)-1
        push!(N, Y[i])
        push!(Nb, Yb[i])
    end

    # Add a trivial link to connect the two halves
    newlink = Index(1, "$L-link-$(L+1)")
    push!(N, Y[end]*ITensor(1, newlink))
    push!(Nb, Yb[end]*ITensor(1, newlink'))
    push!(N, ITensor(1, newlink)*Z[1])
    push!(Nb, ITensor(1, newlink')*Zb[1])

    for i = 2:length(Z)
        push!(N, Z[i])
        push!(Nb, Zb[i])
    end
    
    return MPS(N), MPS(Nb)
end

@doc raw"""
    decomp(M::ITensor, Mb::ITensor, sites::Sites; chi_max::Int=0, timing::Bool=false, method::DM_Method=LR)

Decompose a pair of ITensors into MPS. Will produce left-normal form assuming sites is in ascending order.

# Arguments
- `M`: ITensor, the right-vector tensor to be decomposed. Should have indices `sites`.
- `Mb`: ITensor, the left-vector tensor to be decomposed. Should have indices `sites'`.
- `sites`: Sites, the sites of the MPS. Should be in ascending order if left-normal form is desired, and vice versa.
- `chi_max`: Int, default 0. The maximal bond dimension.
- `open_end`: Bool, default false. Whether to leave the end open. If true, the last site will be decomposed with an imaginary new link, and the new link will be returned.
- `timing`: Bool, default false. Whether to print timing information.
- `method`: DM_Method, default LR. The method to use for the decomposition. Options:
    LR (**L**eft- and **R**ight-vector density matrix method, see `?decomp_lrrho`)
    or BB (**B**iorthonormal-**B**lock method, see `?decomp_biortho`).

# Returns
- `Y`: MPS, the right-vector MPS after decomposition.
- `Yb`: MPS, the left-vector MPS after decomposition.
- `newlink`: Index, returned only if open_end=true. The new link index.
"""
function decomp(M::ITensor, Mb::ITensor, sites::Sites; chi_max::Int=0, open_end::Bool=false, timing::Bool=false, method::DM_Method=LR)

    Nsites = length(sites)

    Y = ITensor[]
    Yb = ITensor[]

    for i = 1:Nsites-1

        # M should have indices (links[i-1],sites[i],sites[i+1:end])
        # We define Li = (links[i-1],sites[i]), Ri = (sites[i+1:end])
        # And then do a decomposition with link=Ri, others=Li
        # The resulting Y would be (Li,newlink), which we take to be the site tensor in the MPS
        # I would be (newlink,Ri), which we take to be the new M

        remsites = sites[i+1:end]
        rcomb = combiner(remsites)
        Ri = inds(rcomb)[1]

        M *= rcomb
        Mb *= rcomb'
        Yi, Ybi, M, Mb = decomp(M, Mb, Ri; chi_max=chi_max, timing=timing, method=method)
        push!(Y, Yi)
        push!(Yb, Ybi)
        M *= rcomb
        Mb *= rcomb'

    end

    if open_end
        # Append an imaginary new link to the end
        endlink = Index(1, "EndLink")
        endtensor = ITensor(1, endlink)
        Mi, Mbi, _, _, endlink = decomp(M*endtensor, Mb*endtensor', endlink; chi_max=chi_max, timing=timing, method=method, return_newlink=true)
        push!(Y, Mi)
        push!(Yb, Mbi)
        return MPS(Y), MPS(Yb), endlink
    else
        # The last M, having indices (links[Nsites-1],sites[Nsites]), is just the last Y.
        push!(Y, M)
        push!(Yb, Mb)
    
        return MPS(Y), MPS(Yb)
    end
end

@doc raw"""
    decomp_biortho(M::ITensor, Mb::ITensor, link::Index; chi_max::Int = 0, unitarize::Bool = false, timing::Bool = false)

Realizes `decomp()` with the biorthonormal-block method. See `?decomp` for argument definition and function description.
This method follows *arXiv*:2401.15000.
"""
function decomp_biortho(M::ITensor, Mb::ITensor, link::Index; chi_max::Int = 0, unitarize::Bool = false, timing::Bool = false, return_newlink::Bool=false)

    if timing
        t1 = now()
    end

    function timestamp(msg)
        if timing
            fprintln("Timestamp :: $(Dates.value(now()-t1)/1000)s :: $(msg)")
        end
    end

    idm = inds(M)

    if !(link in idm)
        throw(ArgumentError("link !in inds(M)!\ninds(M)=$(idm), link=$link"))
    end

    if idm' != inds(Mb)
        throw(ArgumentError("inds(M)' != inds(Mb)! inds(M) = $idm, inds(Mb) = $(inds(Mb))"))
    end
    
    # Combine the non-link indices into one index
    Mcomb = combiner([i for i in idm if i != link]) # Mcomb combiners others to Mci
    Mci = inds(Mcomb)[1]
    msize = dim(Mci)
    M *= Mcomb
    Mb *= Mcomb'

    timestamp("Begin decomposition")

    # Construct Density Matrix, \rho=\sum_i |i_{Mb}\rangle \langle i_M |.
    rho = Array(M*delta(link,link')*Mb, Mci', Mci)
    timestamp("Rho Construction")

    # In the following, we try to find matrices Y(Mci,newlink), Yb(Mci',newlink')
    # such that Y^T * Yb = identity, and newlink has bond dimension <= chi_max.

    # If rho is normal, just do diagonalization
    rhoH = rho'
    if isapprox(rho*rhoH, rhoH*rho; rtol=1e-6)

        w, u = eigen(rho+rhoH)
        if 0 < chi_max < msize
            u = u[:,sortperm(-abs.(w))[1:chi_max]]
        end
        
        # Here since u has orthonormal columns, Ys and Ysb naturally satisfy the condition.
        Ys = u
        Ysb = conj.(u)

    # Generic case
    else

        Ys, Ysb = decomp_biortho_on_rho(rho; unitarize=unitarize, chi_max=chi_max, timestamp=timestamp)

    end

    newlink = Index(size(Ys)[2], tags(link))

    Y = ITensor(Ys, Mci, newlink)
    Yb = ITensor(Ysb, Mci', newlink')

    I = Yb*delta(Mci',Mci)*M
    Ib = Y*delta(Mci,Mci')*Mb

    Y = Y*Mcomb # Switch back from Mci to others
    Yb = Yb*Mcomb'

    timestamp("Final")

    if return_newlink
        return Y, Yb, I, Ib, newlink
    else
        return Y, Yb, I, Ib
    end

end

@doc raw"""
    function decomp_biortho_on_rho(rho; unitarize::Bool=false, chi_max::Int=0; timestamp::Function=((args...)->nothing))

# Arguments
- `rho`: Matrix, the density matrix to be decomposed.
- `unitarize`: Bool, default false. Whether to unitarize the eigenvectors.
- `chi_max`: Int, default 0. The maximal bond dimension.
- `timestamp`: Function, default `((args...)->nothing)`. A function to print timestamps.

# Returns
- `Ys`: Matrix, the right eigenvectors in its columns.
- `Ysb`: Matrix, the left eigenvectors in its columns.
"""
function decomp_biortho_on_rho(rho; unitarize::Bool=false, chi_max::Int=0, timestamp::Function=((args...)->nothing))

    # First pick out the zero singular values of rho
    Yn, Ynb, Z, Zb = nullspace(rho; full_basis=true)
    # Right here, (Yn,Z) and (Ynb,Zb) form a set of biorthonormal basis
    # Whereas under this basis rho is non-zero only in the (Z,Zb) block
    rhoZ = transpose(Zb)*rho*Z
    # In the following, we only do decomposition on rhoZ.
    # Afterwards, we transform back to the original basis by applying Z x rhoZ x Zb.

    # Two-step block diagonalization
    A, C, D, Ss, Sd = schur_sorted(rhoZ, chi_max; timestamp = timestamp) # Schur decomposition

    timestamp("Schur Decomposition")

    # If truncation happened, do Roth removal
    # Originally, rho = (Ss, Sd) * (A, D; 0, C) * (Ss'; Sd')
    # We want to remove the D block, choose X such that AX-XC=D
    # Then we have (1, X; 0, 1) * (A, D; 0, C) * (1, -X; 0, 1) = (A, 0; 0, C)
    # Therefore, have rho = (Ss, Sd-Ss*X) * (A, 0; 0, C) * (Ss'+X*Sd', Sd')
    # In which the middle is block diagonal (also upper triangular),
    # and the left and right matrices are biorthonormal.
    YsZ = Ss
    YsbZ = conj.(Ss)
    if length(D) > 0
        X = sylvc(A, -C, D)
        YsbZ += conj.(Sd) * transpose(X)
    end

    timestamp("Roth Removal")

    # Bi-orthonormalization
    YsZ, YsbZ = GSbiortho(YsZ, YsbZ)

    timestamp("Biorthonormalization")

    # Unitarization
    if unitarize
        YsZ = svd(YsZ).U
        YsbZ = conj.(YsZ)
    end

    # Transform back to the old basis
    Ys = Z*YsZ
    Ysb = Zb*YsbZ

    return Ys, Ysb

end

"""
    function unitize(v1, v2; return_ratios = false)

Make <v1|v2>=1 and norm(v1)=norm(v2). If return_ratios is true, return the ratios that are multiplied to v1 and v2.

NOTICE: v1 is already the left eigenvector, so it NEED NOT BE CONJUGATED.
"""
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

"""
    function GSbiortho(Y, Yb)

Gram-Shimidt Biorthonormalization, the goal is to make Yb.T*Y = I.

# Arguments
- `Y`: Matrix, right eigenvectors in its columns.
- `Yb`: Matrix, left eigenvectors in its columns.

# Returns
- `Y`: Matrix, new right eigenvectors in its columns. Equals to Y times an upper triangonal matrix.
- `Yb`: Matrix, new left eigenvectors in its columns. Equals to Yb times an upper triangonal matrix.
"""
function GSbiortho(Y, Yb)

    if size(Y) != size(Yb)
        throw(ArgumentError("Y and Yb should have the same shape!"))
    end
    if length(size(Y)) != 2
        throw(ArgumentError("Y and Yb should be matrices!"))
    end

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

"""
    function EIGbiortho(Y, Yb)

Biorthonormalization with eigenvalue decomposition. The goal is to make Yb.T*Y = I.
To do this, diagonalize Yb.T*Y, such that inv(C)*Yb.T*Y*C = E.
Then we take the new Y = Y*C*E^(-1/2), and Yb = Yb*inv(C.T)*E^(-1/2).

# Arguments
- `Y`: Matrix, right eigenvectors in its columns.
- `Yb`: Matrix, left eigenvectors in its columns.

# Returns
- `Y`: Matrix, new right eigenvectors in its columns. Equals to Y times a matrix on the right.
- `Yb`: Matrix, new left eigenvectors in its columns. Equals to Yb times a matrix on the right.
"""
function EIGbiortho(Y, Yb)

    if size(Y) != size(Yb)
        throw(ArgumentError("Y and Yb should have the same shape!"))
    end
    if length(size(Y)) != 2
        throw(ArgumentError("Y and Yb should be matrices!"))
    end

    E, C = eigen(transpose(Yb)*Y)
    Ep = Diagonal(E.^(-1/2))

    return Y*C*Ep, Yb*transpose(inv(C))*Ep
end

"""
    function nullspace(M)

Get the null space of a matrix M. Returns Y and Yb, with Y containing the right nullspace of M in its columns,
and Yb its left nullspace in rows. Y and Yb are biorthonormal. I.e., M*Y=Yb*M=0, and Yb*Y=I.

The calculation is realized with svd. Let M = U*S*V', where U and V are unitary, and S is diagonal.
The right null space of M would be V times the null space of S, and the left null space of M would be the null space of S times U'.
Now let R contain the right null space of M in its columns, and L contains left null space of M in its rows.
We proceed to look for matrices P and Q, such that Y=R*P and Yb = Q*L. We want Yb*Y=I, which is Q*(L*R)*P=I.
We diagonalize L*R, producing inv(C)*(L*R)*C = E. Then we have Q = E^(-1/2)*inv(C) and P = C*E^(-1/2).

# Arguments
- `M`: Matrix, the matrix to get the null space of.
- `cutoff`: Float, default 0. The cutoff for singular values. Singular values smaller than this value are considered zero. If zero, use rel_cutoff only.
- `rel_cutoff`: Float, default 1e-16. The relative cutoff for singular values.
    Singular values smaller than this value times the dimension of the matrix are considered zero.
- `full_basis`: Bool, if true, return two more matrices Z and Zb, such that (Y,Z) and (Yb,Zb) form a full biorthonormal basis.

# Returns
- `Y`: Matrix, the right nullspace of M in its columns.
- `Yb`: Matrix, the left nullspace of M in its rows.
- `Z`: Matrix, returned only if full_basis=true. The right basis that are orthogonal to Yb and complements Y.
- `Zb`: Matrix, returned only if full_basis=true. The left basis that are orthogonal to Y and complements Yb, orthonormal to Z.
"""
function nullspace(M; cutoff = 0., rel_cutoff = 1e-16, full_basis::Bool = false)
    Msvd = svd(M)
    cut = rel_cutoff*max(size(M)...)
    if cutoff > 0
        cut = min(cutoff, cut)
    end

    L = (Msvd.U[:,(Msvd.S).<cut])' # Left null space in rows
    R = Msvd.V[:,(Msvd.S).<cut] # Right null space in columns
    Y, Yb = EIGbiortho(R, transpose(L))
    
    if full_basis
        Nnull = size(Y)[2]
        QYb = qr(Yb).Q[:,Nnull+1:end] # Find basis orthogonal to Yb
        QY = qr(Y).Q[:,Nnull+1:end] # Find basis orthogonal to Y
        Z, Zb = EIGbiortho(QYb, QY) # Bi-orthogonalize the remaining basis
        return Y, Yb, Z, Zb
    else
        return Y, Yb
    end
end

@doc raw"""
    function schur_sorted(M, chi::Int; doprint = false)

Does a Schur decomposition followed by a Bartels-Stewart eigenvalue sort.
The final result is
$$
M = (S_s, S_d) \begin{pmatrix} A & D \\ 0 & C \end{pmatrix} \begin{pmatrix} S_s^\dagger \\ S_d^\dagger \end{pmatrix}.
$$
Where the block A has dimension chi or the dimensions of M, whichever is smaller.

To do this, first take a Schur decomposition
$$
M = Z T Z^\dagger,
$$
where $Z$ is unitary, and $T$ is upper triangular. Then, we permute the eigenvalues of $T$
such that the largest chi eigenvalues are in the top-left chi*chi block.

# Arguments
`M`: Matrix, the matrix to be decomposed.

# Returns
`A`, `C`, `D`, `Ss`, `Sd`: Matrices, see descrption above.
"""

# Returns A, C, D, Ss, Sd
function schur_sorted(M::Matrix, chi::Int; timestamp::Function=((args...)->nothing))

    timestamp("Schur sorted called")

    L = size(M)[1]
    Mschur = schur(M) # Calculate schur decomposition of M

    timestamp("Bare decomposition done")

    if chi<=0 || L <= chi
        # In this case no cutoff is needed, A = T, S_s = Z, all others are zero.
        return Mschur.T, zeros(L,0), zeros(0,0), Mschur.Z, zeros(L,0)
    end

    args = sortperm(-abs.(diagm(Mschur.T))) # Sort the eigenvalues of T in decreasing order
    select_pos = fill(false, L)
    select_pos[args[1:chi]] .= true # select_pos is true for the largest chi eigenvalues

    T,Z = ordschur(Mschur, select_pos) # Reorder the Schur decomposition

    timestamp("Permuted")

    return T[1:chi,1:chi], T[chi+1:end,chi+1:end], T[1:chi,chi+1:end], Z[:,1:chi], Z[:,chi+1:end]

end

"""
    function doeig(Ham, v0; sigma::ComplexF64=ComplexF64(0), tol::Float64=0., ncv0::Int=50, use_sparse::Bool=true)

Do sparse diagonalization to find the eigenvalue of a matrix Ham closest to sigma.
If the sparse diagonalization fails to converge, switch to dense diagonalization.
"""
function doeig(Ham, v0; sigma::ComplexF64=ComplexF64(0), tol::Float64=0., ncv0::Int=50, use_sparse::Bool=true, k::Int=1)

    Hdim = size(Ham)[1]

    ncv = ncv0

    try

        if !use_sparse
            throw(ErrorException("This exception jumps the code to the non-sparse method"))
        end

        while true

            try
                fprintln("Trying sprase diagonalization with ncv = $ncv")
                eigvalues, eigvectors = arpack_eigs(Ham; nev=k, which=:LM, sigma=sigma, tol=tol, v0=v0, ncv=ncv)
                fprintln("Converged eigenvalue(s): ", eigvalues)
                if length(eigvalues) < k
                    throw(ErrorException("No convergence: Not enough eigenvalues found, $(length(eigvalues))/$k."))
                end
                if k == 1
                    return eigvalues[1], eigvectors[:,1]
                else
                    return eigvalues, eigvectors
                end
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
        if k == 1
            arg = argmin(abs.(w.-sigma))
            return w[arg], v[:,arg]
        else
            sel = sortperm(abs.(w.-sigma))[1:k]
            return w[sel], v[:,sel]
        end

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

@doc raw"""
    decomp_lrrho(M::ITensor, Mb::ITensor, link::Index; chi_max::Int = 0, timing::Bool = false)

Realizes decomp() with the left- and right-vector density matrix method.
See `?decomp` for argument definition and function description.
This method follows *Phys. Rev. B* **105**, 205125 (2022).

Let M have indices (link, others) and Mb have indices (link', others'). Denote vectors 
$|i_M\rangle$ = M(i,:) and $\langle j_{Mb}|$ = Mb(j,:). Construct the density matrix as
$$
\rho = \frac{1}{2} \left[\sum_i |i_M\rangle\langle i_M| + \sum_j |j_{Mb}\rangle \langle j_{Mb}|\right]
$$
Then we diagonalize this (Hermitian) density matrix as
$$
\rho = \sum_{\lambda} \rho_{\lambda} |\lambda\rangle\langle\lambda|.
$$
We then choose Y($\lambda$,:) = $|\lambda\rangle$ and Yb($\lambda$,:) = $\langle\lambda|$.
In the case where #$\lambda$ > chi_max, we only keep the chi_max largest eigenvalues and eigenvectors.
I is constructed as I($\lambda$,i) = $\langle\lambda|$M(i,:), and similarly for Ib.
In this case I*Y = $\left[\sum_\lambda |\lambda\rangle\langle\lambda| \right]$, which is identity if
we have chosen all the $\lambda$'s, and a projector into the largest-singular-value subspace if we have done a cutoff.
"""
function decomp_lrrho(M::ITensor, Mb::ITensor, link::Index; chi_max::Int = 0, timing::Bool = false, return_newlink::Bool=false)

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

    Y = U*Mcomb
    Yb = conj(U)'*Mcomb'

    if return_newlink
        return Y, Yb, I, Ib, newlink
    else
        return Y, Yb, I, Ib
    end
    
end