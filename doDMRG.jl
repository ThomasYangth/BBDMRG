# -*- coding: utf-8 -*-
# doDMRG.jl

"""

    Non-Hermitian Density Matrix Renormalization Group, by Tian-Hua Yang,
    based on www.tensors.net, (v1.1) by Glen Evenbly.

"""

using Dates
using Statistics # For std
using JLD2 # For saving and processing array

using Printf

export doDMRG_excited, doDMRG_excited_IncL, doDMRG_IncChi, doDMRG

"""
    doDMRG_excited(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
    k::Int=1, expected_gap::Float64=1, tol::Float64 = 1e-15,
    numsweeps::Int = 10, dispon::Int = 2, debug::Bool = false, method::DM_Method = LR,
    cut::Float64 = 1e-8, stop_if_not_converge::Bool = true, savename = nothing, override::Bool = false)

Function does DMRG to find the first k excited states of a Hamiltonian given by the MPO W.

# Arguments
- `M`: MPS, initial guess for right eigenvector.
- `Mb`: MPS, initial guess for left eigenvector.
- `W`: MPO, Hamiltonian, or Liouvillian.
- `chi_max`: Int, maximal bond dimension.
- `k`: Int, number of excited states; default 1, ground state only.
- `expected_gap`: Float, default 1. Expected gap between excited states,
  used to normalize the Hamiltonian when searching for excited states.
- `tol`: Float, default 1e-15. Tolerance when diagonalizing a local block.
- `numsweeps`: Int, default 10. Number of sweeps.
- `savename`: String, default nothing. A temporary file that saves the already-converged states.


# Returns
Description of the return value.

# Examples

"""
function doDMRG_excited(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
    k::Int=1, expected_gap::Float64=1, tol::Float64 = 1e-15,
    numsweeps::Int = 10, dispon::Int = 2, debug::Bool = false, method::DM_Method = LR,
    cut::Float64 = 1e-8, stop_if_not_converge::Bool = true, savename = nothing, override::Bool = false)

    L = length(W)

    Ms = Array{MPS, 1}()
    Mbs = Array{MPS, 1}()
    Es = Array{ComplexF64, 1}()

    chi_start = 30 # 
    vt_amp = 15

    if isnothing(savename)
        savename = Dates.format(now(), "MMddyy-HHMMSS")
    end

    if !isa(W, MPO)
        W = MPO(W)
    end

    filename = "$savename.jld2"

    if !override && isfile(filename)
        fprintln("Loading file $filename")
        jldopen(filename, "r") do file

            # Read the already converged energies
            if haskey(file, "Es")
                Es = file["Es"]
            end

            # Read the already converged eigenstates
            k_loop_1 = false
            for i = 1:k
                thisM = []
                for j = 1:L
                    if !haskey(file, "M$(i)R$(j)")
                        k_loop_1 = true
                        break
                    end
                    push!(thisM, file["M$(i)R$(j)"])
                end
                if k_loop_1
                    break
                end
                push!(Ms, MPS(thisM))
            end

            k_loop_2 = false
            for i = 1:k
                thisM = []
                for j = 1:L
                    if !haskey(file, "M$(i)L$(j)")
                        k_loop_2 = true
                        break
                    end
                    push!(thisM, file["M$(i)L$(j)"])
                end
                if k_loop_2
                    break
                end
                push!(Mbs, MPS(thisM))
            end

        end
    end

    function dosave()
        jldopen(filename, (!override && isfile(filename)) ? "r+" : "w+") do file

            file["Es"] = Es

            for (i,M) in enumerate(Ms)
                for (key,mat) in M.asdict("M$(i)R")
                    if override && !haskey(file, key)
                        file[key] = mat
                    end
                end
            end

            for (i,M) in enumerate(Mbs)
                for (key,mat) in M.asdict("M$(i)L")
                    if override && !haskey(file, key)
                        file[key] = mat
                    end
                end
            end

        end
        fprintln("Saved current data: len(Es)=$(length(Es)), len(Ms)=$(length(Ms)), len(Mbs)=$(length(Mbs)).")
    end
        
    for thisk = 1:k

        fprintln("Finding eigenvalue $(thisk)")

        if thisk == length(Ms) + 1

            sigma = shift_eps*im
            #sigma = length(Es) > 0 ? Es[end] : 0

            if method == BB

                Ekeep, Hdifs, Y, Yb = doDMRG_IncChi(M, Mb, W, chi_max;
                    normalize_against = [(Ms[i],Mbs[i],-expected_gap*(thisk-i)) for i = 1:thisk-1],
                    sigma=sigma, vt_amp=vt_amp, tol_end=tol, chi_start=chi_start,
                    numsweeps=numsweeps, dispon=dispon, debug=debug, method=method)

                if Hdifs[end] < 1e-3
                    fprintln("Found eigenvalue $thisk = $(fmtcpx(Ekeep[-1]))")
                else
                    fprintln("ERROR: Failed to converge for eigenvalue $thisk: <Delta H^2> = $(fmtf(Hdifs[end]))")
                    if stop_if_not_converge
                        throw(ErrorException())
                    end
                end

                push!(Es, Ekeep[-1])
                push!(Ms, Y)
                push!(Mbs, Yb)

            elseif method == LR

                Ekeep, Hdifs, Y, _ = doDMRG_IncChi(M, Mb, W, chi_max;
                    normalize_against = [(Ms[i],Mbs[i],-expected_gap*(thisk-i)) for i = 1:thisk-1],
                    sigma=sigma, vt_amp=vt_amp, tol_end=tol, chi_start=chi_start,
                    numsweeps=numsweeps, dispon=dispon, debug=debug, method=method)

                if Hdifs[end] < 1e-3
                    fprintln("Found eigenvalue $thisk = $(fmtcpx(Ekeep[-1]))")
                else
                    fprintln("ERROR: Failed to converge for eigenvalue $thisk: <Delta H^2> = $(fmtf(Hdifs[end]))")
                    if stop_if_not_converge
                        throw(ErrorException())
                    end
                end

                push!(Es, Ekeep[-1])
                push!(Ms, Y)

            else
                throw(ArgumentError("Unrecognized method: $method"))
            end

            dosave()

        end

        if thisk != length(Ms)
            throw(ErrorException("Unexpected: thisk = $(thisk), length(Ms) = $(length(Ms))."))
        end

        if thisk == length(Mbs) + 1 && method == LR

            # Right-normalize the M solution
            _,_, Z, Zb = doDMRG(Ms[thisk], conj.(Ms[thisk]), W, chi_max; numsweeps=0, updateon=false)

            Ekeep, Hdifs, Y, _ = doDMRG_IncChi(Z, Zb, W, chi_max;
                    normalize_against = [(Ms[i],Mbs[i],-expected_gap*(thisk-i)) for i = 1:thisk-1],
                    sigma = shift_eps*im, vt_amp=vt_amp, tol_end=tol, chi_start=chi_start,
                    numsweeps=numsweeps, dispon=dispon, debug=debug, method=method)

            if Hdifs[end] < 1e-3
                fprintln("Found eigenvalue $thisk = $(fmtcpx(Ekeep[end]))")
            else
                fprintln("ERROR: Failed to converge for eigenvalue $thisk: <Delta H^2> = $(fmtf(Hdifs[end]))")
                if stop_if_not_converge
                    throw(ErrorException())
                end
            end

            push!(Mbs, Y)

        end

        if thisk != length(Mbs)
            throw(ErrorException("Unexpected: thisk = $(thisk), length(Mbs) = $(length(Mbs))."))
        end

        dosave()

    end

    return Ms, Mbs, Es

end

function doDMRG_excited_IncL(W0::Array{<:Number, 4}, chi_max::Int, L0::Int, doubles::Int;
    k::Int=1, expected_gap::Float64=1., tol::Float64 = 1e-15,
    numsweeps::Int = 10, dispon::Int = 2, debug::Bool = false, method::DM_Method = LR,
    cut::Float64 = 1e-8, stop_if_not_converge::Bool = true, savename = nothing, override::Bool = false)

    if isnothing(savename)
        savename = Dates.format(now(), "MMddyy-HHMMSS")
    end

    filename = "$savename.jld2"

    # Ms[i,j] will be the j-th excited state for system size L0*2^i.
    Ms = Array{MPS, 2}(undef, k, doubles)
    Mbs = Array{MPS, 2}(undef, k, doubles)
    Es = Array{ComplexF64, 1}()

    # At length L0, do a sparse diagonalization for the full matrix, to get the initial states.
    sites = [Index(size(W0)[1], "Site $i") for i = 1:L0*(2^doubles)]
    initSites = sites[1:L0]
    W_L0 = MPOonSites(W0, initSites; leftindex=1, rightindex=2)
    W_L0_mat, comb = MPO_to_Matrix(W_L0)
    Mci = inds(comb)[1]
    wR, vR = doeig(W_L0_mat, ComplexF64[randn() + im * randn() for _ in 1:L0]; k=k, use_sparse=false)
    wL, vL = doeig(transpose(W_L0_mat), ComplexF64[randn() + im * randn() for _ in 1:L0]; k=k, use_sparse=false)

    fprintln("Found right eigenvalues: ", wR)
    fprintln("Found left eigenvalues: ", wL)

    if !isapprox(wR, wL; atol=1e-6)
        throw(ErrorException("Eigenvalues of W_L0 are not symmetric!"))
    end

    # Double the states vR and vL to serve as initial guesses for the system size 2*L0.
    for i = 1:k

        Y, Yb = decomp(ITensor(vR[:,i], Mci)*comb, ITensor(vL[:,i], Mci')*comb', initSites; chi_max=chi_max, timing=debug, method=method, linknames=["$j-link-$(j+1)" for j=1:L0-1])

        initSites2 = sites[L0+1:2*L0]
        comb2 = combiner(initSites2...)
        Mci2 = inds(comb2)[1]
        Z, Zb = decomp(ITensor(vR[:,i], Mci2)*comb2, ITensor(vL[:,i], Mci2')*comb2', initSites2[end:-1:1]; chi_max=chi_max, timing=debug, method=method, linknames=["$(j-1)-link-$j" for j=2*L0:-1:L0+2])
    
        M = ITensor[]
        Mb = ITensor[]
        for j = 1:L0-1
            push!(M, Y[j])
            push!(Mb, Yb[j])
        end

        # Add a trivial link to connect the two halves
        newlink = Index(1, "$L0-link-$(L0+1)")
        push!(M, Y[end]*ITensor(1, newlink))
        push!(Mb, Yb[end]*ITensor(1, newlink'))
        push!(M, ITensor(1, newlink)*Z[end])
        push!(Mb, ITensor(1, newlink')*Zb[end])

        for j = 1:L0-1
            push!(M, Z[L0-j])
            push!(Mb, Zb[L0-j])
        end

        Ms[i,1] = MPS(M)
        Mbs[i,1] = MPS(Mb)
    
    end

    for j = 1:doubles

        this_L = L0*(2^j)
        this_sites = sites[1:this_L]
        thisW = MPOonSites(W0, this_sites; leftindex=1, rightindex=2)

        for thisk = 1:k
            # Do DMRG for this size
            _, _, Ms[thisk,j], Mbs[thisk,j] = doDMRG(Ms[thisk,j], Mbs[thisk,j], thisW, chi_max;
                    normalize_against = [(Ms[i,j], Mbs[i,j], -expected_gap*(thisk-i)) for i = 1:thisk-1],
                    sigma = shift_eps*im + 0.1, numsweeps=numsweeps, dispon=dispon, debug=debug, method=method,
                    stop_if_not_converge=stop_if_not_converge)
            # Double the MPS for the next initial guess
            if j < doubles
                Ms[thisk,j+1], Mbs[thisk,j+1] = doubleMPSSize(Ms[thisk,j], Mbs[thisk,j]; chi_max=chi_max, method=method, timing=debug, newsites=sites[this_L+1:2*this_L])
            end
        end

    end

end

function doDMRG_IncChi(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
    chi_inc::Int = 10, chi_start::Int = 20, init_sweeps::Int = 5, inc_sweeps::Int = 2,
    tol_start::Float64 = 1e-3, tol_end::Float64 = 1e-6, vt_amp::Int = 3, vt_sweeps::Int = 3,
    numsweeps::Int = 10, dispon::Int = 2, debug = false, method::DM_Method = LR,
    sigma::ComplexF64 = shift_eps*im, normalize_against = [])

    _,_,M,Mb = doDMRG(M, Mb, W, chi_start;
        tol=tol_start, numsweeps=init_sweeps, dispon=dispon, updateon=true,
        debug=debug, method=method, normalize_against=normalize_against, sigma=sigma)
    
    chi = chi_start + chi_inc
    while chi < chi_max
        _,_,M,Mb = doDMRG(M, Mb, W, chi;
            tol=tol_start, numsweeps=inc_sweeps, dispon=dispon, updateon=true,
            debug=debug, method=method, normalize_against=normalize_against, sigma=sigma)
        chi += chi_inc
    end

    chi = chi_max
    tol = tol_start
    while tol > tol_end
        _,_,M,Mb = doDMRG(M, Mb, W, chi;
            tol=tol, numsweeps=vt_sweeps, dispon=dispon, updateon=true,
            debug=debug, method=method, normalize_against=normalize_against, sigma=sigma)
        tol *= (10.)^(-vt_amp)
    end

    return doDMRG(M, Mb, W, chi_max; tol=tol_end, numsweeps=numsweeps,
        dispon=dispon, updateon=true, debug=debug, method=method, normalize_against=normalize_against, sigma=sigma)

end

# ------------------------
# Implementation of DMRG for a 1D chain with open boundaries. Input 'M' is containing the MPS \
# tensors whose length is equal to that of the 1D lattice, and 'Mb' is the corresponding left \
# vector. The Hamiltonian is specified by an MPO with entries 'W'. Automatically grow the MPS bond \
# dimension to maximum dimension 'chi_max'.

# Optional arguments:
# `numsweeps::Integer=10`: number of DMRG sweeps
# `dispon::Integer=2`: print data never [0], after each sweep [1], each step [2]
# `updateon::Bool=true`: enable or disable tensor updates
# `debug::Bool=False`: enable debugging messages
# `which::str="SR"`: which eigenvalue to choose, "SR" indicates smallest real part
# `method::str="biortho"`: method for truncation of density matrix; 'biortho' is for bbDMRG, \
#     'lrrho' for using the density matrix rho=(psiL psiL + psiR psiR)/2
# """
function doDMRG(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
    numsweeps::Int = 10, sigma::ComplexF64 = shift_eps*im, dispon = 2, updateon = true, debug = false,
    method::DM_Method = LR, tol::Float64=0., normalize_against = [], stop_if_not_converge::Bool=false)
 
    converged = false

    ##### left-to-right 'warmup', put MPS in right orthogonal form
    # Index of W is: left'' - right'' - physical' - physical
    # Index notation: no prime = ket, one prime = bra, two primes = operator link
    Nsites = length(M)
    if length(Mb) != Nsites
        throw(ArgumentError("Length of M and Mb must match!"))
    end

    sites, links = find_sites_and_links(M)

    physdim = dim(sites[1])
    trivSites = Int(floor(log(physdim, chi_max))) # The first and last trivSites need not be sweeped.

    # Each element in normalize_against should be a tuple (Mi, Mib, amp)
    # Corresponding to adding a term amp * Mi*Mib to the Hamiltonian
    # For this we record LNA, RNA, LNAb, RNAb
    # LNA[i] corresponds to the product of Mib with the current M at site i
    # Simialr for the other three
    NumNA = length(normalize_against)
    LNA = Vector{ITensor}[]
    RNA = Vector{ITensor}[]
    LNAb = Vector{ITensor}[]
    RNAb = Vector{ITensor}[]
    Namp = ComplexF64[]
    MN = MPS[]
    MNb = MPS[]

    unit_itensor = ITensor(ComplexF64(1))

    for (i,item) in enumerate(normalize_against)
        push!(LNA, fill(unit_itensor, Nsites))
        push!(RNA, fill(unit_itensor, Nsites))
        push!(LNAb, fill(unit_itensor, Nsites))
        push!(RNAb, fill(unit_itensor, Nsites))
        push!(MN, item[1])
        push!(MNb, item[2])
        push!(Namp, item[3])
    end
    
    # The L[i] operator is the MPO contracted with the MPS and its dagger for sites <= i-1
    # R[i] is contracted for sites >= i+1
    L = fill(unit_itensor, Nsites)
    R = fill(unit_itensor, Nsites)

    pos = trivSites+1

    for p = Nsites:-1:pos+1 # Do right normalization, from site Nsites to trivSites+2

        # Shape of M is: left bond - physical bond - right bond
        if size(M[p]) != size(Mb[p])
            throw(ArgumentError("Shapes of M[p] and Mb[p] must match!"))
        end
        
        # Set the p-th matrix to right normal form, and multiply the transform matrix to p-1
        M[p], Mb[p], I, Ib, links[p-1] = decomp(M[p], Mb[p], links[p-1]; chi_max=chi_max, timing=debug, method=method, return_newlink=true) # linkind=0 means last

        M[p-1] = M[p-1]*I
        Mb[p-1] = Mb[p-1]*Ib

        # Construct R[p-1]. The indices of R is: left'' - left' - left
        R[p-1] = Mb[p]*R[p]*W[p]*M[p]

        if debug
            fprintln("M[$p] = ", inds(M[p]))
            fprintln("Mb[$p] = ", inds(Mb[p]))
            fprintln("I[$p] = ", inds(I))
            fprintln("Ib[$p] = ", inds(Ib))
            fprintln("M[$(p-1)] = ",inds(M[p-1]))
            fprintln("Mb[$(p-1)] = ", inds(Mb[p-1]))
            fprintln("R[$(p-1)] = ", inds(R[p-1]))
        end

        for i = 1:NumNA

            if debug
                fprintln("MN[$i][$p] = ", inds(MN[i][p]))
                fprintln("MNb[$i][$p] = ", inds(MNb[i][p]))
                fprintln("RNA[$i][$p] = ", inds(RNA[i][p]))
                fprintln("RNAb[$i][$p] = ", inds(RNAb[i][p]))
            end

            RNA[i][p-1] = RNA[i][p]*MNb[i][p]*M[p]*delta(sites[p],sites[p]')
            RNAb[i][p-1] = RNAb[i][p]*MN[i][p]*Mb[p]*delta(sites[p],sites[p]')
        end

    end

    for p = 1:pos-1 # Do left normalization, from site 1 to trivSites

        if size(M[p]) != size(Mb[p])
            throw(ArgumentError("Shapes of M[p] and Mb[p] must match!"))
        end
        
        # Set the p-th matrix to left normal form, and multiply the transform matrix to p+1
        M[p], Mb[p], I, Ib, links[p] = decomp(M[p], Mb[p], links[p]; chi_max=chi_max, timing=debug, method=method, return_newlink=true) # linkind=0 means last

        M[p+1] = M[p+1]*I
        Mb[p+1] = Mb[p+1]*Ib

        # Construct L[p+1]. The indices of R is: left'' - left' - left
        L[p+1] = Mb[p]*L[p]*W[p]*M[p]

        if debug
            fprintln("links = ", links)
            fprintln("M[$p] = ", inds(M[p]))
            fprintln("Mb[$p] = ", inds(Mb[p]))
            fprintln("I[$p] = ", inds(I))
            fprintln("Ib[$p] = ", inds(Ib))
            fprintln("M[$(p+1)] = ",inds(M[p+1]))
            fprintln("Mb[$(p+1)] = ", inds(Mb[p+1]))
            fprintln("L[$(p+1)] = ", inds(L[p+1]))
        end

        for i = 1:NumNA
            LNA[i][p+1] = LNA[i][p]*MNb[i][p]*M[p]*delta(sites[p],sites[p]')
            LNAb[i][p+1] = LNAb[i][p]*MN[i][p]*Mb[p]*delta(sites[p],sites[p]')
        end

    end

    # Normalize M[pos] and Mb[pos] so that the trial wave functions are bi-normalized
    ratio = 1/sqrt((Mb[pos]*(M[pos]'))[])
    M[pos] *= ratio
    Mb[pos] *= ratio

    # At this point we have turned M[2:end] to right normal form, and constructed R[2:end]
    # We start the sweep at site 0
    # The effective Hamiltonian at site [i] is the contraction of L[i], R[i], and W[i]
    
    Ekeep = ComplexF64[]
    Hdifs = Float64[]

    k = 1

    while k < numsweeps+1
        
        ##### final sweep is only for orthogonalization (disable updates)
        if k == numsweeps+1
            updateon = false
            dispon = 0
        end
        
        ###### Optimization sweep: left-to-right
        for p = pos:Nsites-pos

            # Optimize at this step
            if updateon
                E, M[p], Mb[p] = eigLR(L[p], R[p], W[p], M[p], Mb[p],
                    sigma = sigma, use_sparse = true, tol = tol, timing = debug,
                    normalize_against = [(LNA[i][p],RNA[i][p],MNb[i][p],LNAb[i][p],RNAb[i][p],MN[i][p],Namp[i]) for i = 1:NumNA])    
                push!(Ekeep, E)
            end

            # Move the pointer one site to the right, and left-normalize the matrices at the currenter pointer
            M[p], Mb[p], I, Ib, links[p] = decomp(M[p], Mb[p], links[p]; chi_max=chi_max, timing=debug, method=method, return_newlink=true)

            M[p+1] = I*M[p+1]
            Mb[p+1] = Ib*Mb[p+1]

            # Construct L[p+1]
            L[p+1] = Mb[p]*L[p]*W[p]*M[p]

            for i = 1:NumNA
                LNA[i][p+1] = LNA[i][p]*MNb[i][p]*M[p]*delta(sites[p],sites[p]')
                LNAb[i][p+1] = LNAb[i][p]*MN[i][p]*Mb[p]*delta(sites[p],sites[p]')
            end

            if debug
                fprintln("links = ", links)
                fprintln("M[$p] = ", inds(M[p]))
                fprintln("Mb[$p] = ", inds(Mb[p]))
                fprintln("I[$p] = ", inds(I))
                fprintln("Ib[$p] = ", inds(Ib))
                fprintln("M[$(p+1)] = ",inds(M[p+1]))
                fprintln("Mb[$(p+1)] = ", inds(Mb[p+1]))
                fprintln("L[$(p+1)] = ", inds(L[p+1]))
            end
        
            ##### display energy
            if dispon == 2
                fprintln("Sweep: $k of $numsweeps, Loc: $p, chi: $chi_max, Energy: $(fmtcpx(Ekeep[end]))")
            end
            
        end
        
        ###### Optimization sweep: right-to-left
        for p = Nsites-pos+1:-1:pos+1

            # Optimize at this step
            if updateon
                E, M[p], Mb[p] = eigLR(L[p], R[p], W[p], M[p], Mb[p],
                    sigma = sigma, use_sparse = true, tol = tol, timing = debug,
                    normalize_against = [(LNA[i][p],RNA[i][p],MNb[i][p],LNAb[i][p],RNAb[i][p],MN[i][p],Namp[i]) for i = 1:NumNA])    
                push!(Ekeep, E)
            end

            # Move the pointer one site to the left, and right-normalize the matrices at the currenter pointer
            M[p], Mb[p], I, Ib, links[p-1] = decomp(M[p], Mb[p], links[p-1]; chi_max=chi_max, timing=debug, method=method, return_newlink=true)
            M[p-1] = M[p-1]*I
            Mb[p-1] = Mb[p-1]*Ib

            # Construct R[p-1]. The indices of R is: left'' - left - left'
            R[p-1] = Mb[p]*R[p]*W[p]*M[p]

            for i = 1:NumNA
                RNA[i][p-1] = RNA[i][p]*MNb[i][p]*M[p]*delta(sites[p],sites[p]')
                RNAb[i][p-1] = RNAb[i][p]*Mb[p]*MN[i][p]*delta(sites[p],sites[p]')
            end

            if debug
                fprintln("M[$p] = ", inds(M[p]))
                fprintln("Mb[$p] = ", inds(Mb[p]))
                fprintln("I[$p] = ", inds(I))
                fprintln("Ib[$p] = ", inds(Ib))
                fprintln("M[$(p-1)] = ",inds(M[p-1]))
                fprintln("Mb[$(p-1)] = ", inds(Mb[p-1]))
                fprintln("R[$(p-1)] = ", inds(R[p-1]))
            end
        
            ##### display energy
            if dispon == 2
                fprintln("Sweep: $k of $numsweeps, Loc: $p, chi: $chi_max, Energy: $(fmtcpx(Ekeep[end]))")
            end

        end
        
        # Calculate <H^2>-<H>^2
        RR = ITensor(1. +0im)
        _,Wlinks = find_sites_and_links(W)
        newlinks = Vector{Index}(undef, Nsites-1)
        for p = Nsites:-1:1
            physind = sites[p]
            newphys = Index(dim(physind), tags(physind))
            Wlow = W[p]*delta(physind',newphys) # Lower W has indices (physind, newphys, links), contracts with M
            Wupp = W[p]*delta(physind,newphys) # Upper W has indices (newphys, physind', links), contracts with Mb
            # Renew the left link
            if p > 1
                leftlink = Wlinks[p-1]
                newlinks[p-1] = Index(dim(leftlink), join(tags(leftlink), ",")*",upp")
                Wupp *= delta(leftlink, newlinks[p-1])
            end
            # Renew the right link
            if p < Nsites
                Wupp *= delta(Wlinks[p], newlinks[p])
            end
            if debug
                fprintln("At step $p, inds(Wlow) = $(inds(Wlow))")
                fprintln("At step $p, inds(Wupp) = $(inds(Wupp))")
                fprintln("At step $p, inds(M) = $(inds(M[p]))")
                fprintln("At step $p, inds(Mb) = $(inds(Mb[p]))")
            end

            RR = (Mb[p] * RR * M[p]) * (Wupp * Wlow)
            if debug
                fprintln("At step $p, inds(RR) = $(inds(RR))")
            end
        end

        Hdif = abs(RR[] - Ekeep[end]^2)
        push!(Hdifs, Hdif)

        if dispon >= 1
            fprintln("Sweep: $k of $numsweeps, Energy: $(fmtcpx(Ekeep[end])), Hdif: $(fmtf(Hdif)), Bonddim: $chi_max, tol: $tol")
        end

        cut = max(tol, eps(Float64)) * 10
        # Early termination if converged
        if abs(std(Ekeep[end-(2*Nsites-4*pos+1):end])) < cut && Hdif < cut
            fprintln("Converged")
            converged = true
            k = numsweeps+1
        end

        k += 1

    end

    # Clean up memory
    foreach(finalize, [LNA, RNA, LNAb, RNAb, Namp, MN, MNb, L, R])
    GC.gc()

    if !converged && stop_if_not_converge
        throw(ErrorException("Failed to converge after $numsweeps sweeps."))
    end
            
    return Ekeep, Hdifs, M, Mb

end