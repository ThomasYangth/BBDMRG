# -*- coding: utf-8 -*-
# doDMRG_bb.py

using Dates
using TensorOperations
using Statistics # For std
using JLD2 # For saving and processing array

include("Config.jl")
include("BiorthoLib.jl")

using Printf

function fmtf(x)
    return @sprintf("%.3f", x)
end

function fmtcpx(z)
    return @sprintf("%.3f %+.3fim", real(z), imag(z))
end

"""
    doDMRG_excited(M, Mb, W, chi_max;
        k=1, expected_gap = 1, tol = 1e-15, numsweeps = 10,
        dispon = 2, debug = false, method = "biortho",
        cut = 1e-8, stop_if_not_converge = true, savename = nothing)

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
    Es = Array{Complex, 1}()

    chi_start = 30 # 
    vt_amp = 15

    if savename is nothing
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
            k_loop_1:
            for i = 1:k
                thisM = []
                for j = 1:L
                    if !haskey(file, "M$(i)R$(j)")
                        break k_loop_1
                    end
                    push!(thisM, file["M$(i)R$(j)"])
                end
                push!(Ms, MPS(thisM))
            end

            k_loop_2:
            for i = 1:k
                thisM = []
                for j = 1:L
                    if !haskey(file, "M$(i)L$(j)")
                        break k_loop_2
                    end
                    push!(thisM, file["M$(i)L$(j)"])
                end
                push!(Mbs, MPS(thisM))
            end

        end
    end

    function dosave()
        jldopen(filename, (!override && isfile(filename)) ? "r+" : "w+") do file

            file["Es"] = Es

            for i, M in enumerate(Ms)
                for key, mat in M.asdict("M$(i)R")
                    if override && !haskey(file, key)
                        file[key] = mat
                    end
                end
            end

            for i, M in enumerate(Mbs)
                for key, mat in M.asdict("M$(i)L")
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

            sigma = Complex(0)
            #sigma = length(Es) > 0 ? Es[end] : 0

            if method == BB

                Ekeep, Hdifs, Y, Yb, _, _ = doDMRG_IncChi(M, Mb, W, chi_max;,
                    normalize_against = [(Ms[i],Mbs[i],-expected_gap*(thisk-i)) for i = 1:thisk-1],
                    sigma=sigma, vt_amp=vt_amp, tol_end=tol, chi_start=chi_start,
                    numsweeps=numsweeps, dispon=dispon, debug=debug, method=method)

                if Hdifs[end] < 1e-3
                    fprintln("Found eigenvalue $thisk = $(fmtcpx(Ekeep[-1]))")
                else
                    fprintln("ERROR: Failed to converge for eigenvalue $thisk: <Delta H^2> = $(fmtf(Hdifs[end]))")
                    if stop_if_not_converge
                        throw(RuntimeError())
                    end
                end

                push!(Es, Ekeep[-1])
                push!(Ms, Y)
                push!(Mbs, Yb)

            elseif method == LR

                Ekeep, Hdifs, Y, _,_,_ = doDMRG_IncChi(M, Mb, W, chi_max;
                    normalize_against = [(Ms[i],Mbs[i],-expected_gap*(thisk-i)) for i = 1:thisk-1],
                    sigma=sigma, vt_amp=vt_amp, tol_end=tol, chi_start=chi_start,
                    numsweeps=numsweeps, dispon=dispon, debug=debug, method=method)

                if Hdifs[end] < 1e-3
                    fprintln("Found eigenvalue $thisk = $(fmtcpx(Ekeep[-1]))")
                else
                    fprintln("ERROR: Failed to converge for eigenvalue $thisk: <Delta H^2> = $(fmtf(Hdifs[end]))")
                    if stop_if_not_converge
                        throw(RuntimeError())
                    end
                end

                push!(Es, Ekeep[-1])
                push!(Ms, Y)

            else
                throw(RuntimeError("Unrecognized method: $method"))
            end

            dosave()

        end

        if thisk != length(Ms)
            throw(RuntimeError("Unexpected: thisk = $(thisk), length(Ms) = $(length(Ms))."))
        end

        if thisk == length(Mbs) + 1 && method == LR

            # Right-normalize the M solution
            _,_,_,_, Z, Zb = doDMRG_bb(Ms[thisk], conj.(Ms[thisk]), W, chi_max; numsweeps=0, updateon=false)

            Ekeep, Hdifs, Y, _,_,_ = doDMRG_IncChi(Z, Zb, W, chi_max;,
                    normalize_against = [(Ms[i],Mbs[i],-expected_gap*(thisk-i)) for i = 1:thisk-1],
                    sigma = Es[end], vt_amp=vt_amp, tol_end=tol, chi_start=chi_start,
                    numsweeps=numsweeps, dispon=dispon, debug=debug, method=method)

            if Hdifs[end] < 1e-3
                fprintln("Found eigenvalue $thisk = $(fmtcpx(Ekeep[end]))")
            else
                fprintln("ERROR: Failed to converge for eigenvalue $thisk: <Delta H^2> = $(fmtf(Hdifs[end]))")
                if stop_if_not_converge
                    throw(RuntimeError())
                end
            end

            push!(Mbs, Y)

        end

        if thisk != length(Mbs)
            throw(RuntimeError("Unexpected: thisk = $(thisk), length(Mbs) = $(length(Mbs))."))
        end

        dosave()

    end

    return Ms, Mbs, Es

end
    

function doDMRG_IncChi(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
    chi_inc::Int = 10, chi_start::Int = 20, init_sweeps::Int = 5, inc_sweeps::Int = 2,
    tol_start::Float64 = 1e-3, tol_end::Float64 = 1e-6, vt_amp::Int = 3, vt_sweeps::Int = 3,
    numsweeps::Int = 10, dispon::Int = 2, debug = false, method::DM_Method = LR,
    sigma::Complex = 0.+0im, normalize_against = [])

    _,_,M,Mb,_,_ = doDMRG_bb(M, Mb, W, chi_start;
        tol=tol_start, numsweeps=init_sweeps, dispon=dispon, updateon=true,
        debug=debug, method=method, normalize_against=normalize_against, sigma=sigma)
    
    chi = chi_start + chi_inc
    while chi < chi_max
        _,_,M,Mb,_,_ = doDMRG_bb(M, Mb, W, chi;
            tol=tol_start, numsweeps=inc_sweeps, dispon=dispon, updateon=true,
            debug=debug, method=method, normalize_against=normalize_against, sigma=sigma)
        chi += chi_inc
    end

    chi = chi_max
    tol = tol_start
    while tol > tol_end
        _,_,M,Mb,_,_ = doDMRG_bb(M, Mb, W, chi;
            tol=tol, numsweeps=vt_sweeps, dispon=dispon, updateon=true,
            debug=debug, method=method, normalize_against=normalize_against, sigma=sigma)
        tol *= 10^(-vt_amp)
    end

    return doDMRG_bb(M, Mb, W, chi_max; tol=tol_end, numsweeps=numsweeps,
        dispon=dispon, updateon=true, debug=debug, method=method, normalize_against=normalize_against, sigma=sigma)

end

"""
------------------------
By Tian-Hua Yang, based on www.tensors.net, (v1.1) by Glen Evenbly.
------------------------

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
function doDMRG_bb(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
    numsweeps::Int = 10, sigma::Complex = 0.+0im, dispon = 2, updateon = true, debug = false,
    method::DM_Method = LR, tol::Float=0, normalize_against = []):
 
    ##### left-to-right 'warmup', put MPS in right orthogonal form
    # Index of W is: left'' - right'' - physical' - physical
    # Index notation: no prime = ket, one prime = bra, two primes = operator link
    Nsites = length(M)
    if length(Mb) != Nsites
        throw(ArgumentError("Length of M and Mb must match!"))
    end

    # Each element in normalize_against should be a tuple (Mi, Mib, amp)
    # Corresponding to adding a term amp * Mi*Mib to the Hamiltonian
    # For this we record LNA, RNA, LNAb, RNAb
    # LNA[i] corresponds to the product of Mib with the current M at site i
    # Simialr for the other three
    NumNA = length(normalize_against)
    LNA = []
    RNA = []
    LNAb = []
    RNAb = []
    Namp = []
    MN = []
    MNb = []
    for i,item in enumerate(normalize_against)
        LNA.append(fill(Complex.(ones(1,1)), Nsites))
        RNA.append(fill(Complex.(ones(1,1)), Nsites))
        LNAb.append(fill(Complex.(ones(1,1)), Nsites))
        RNAb.append(fill(Complex.(ones(1,1)), Nsites))
        push!(MN, item[1])
        push!(MNb, item[2])
        push!(Namp, item[3])
    end
    
    # The L[i] operator is the MPO contracted with the MPS and its dagger for sites <= i-1
    # R[i] is contracted for sites >= i+1
    L = fill(Complex.(ones(1,1,1)), Nsites)
    R = fill(Complex.(ones(1,1,1)), Nsites)
    Y = fill(Complex.(ones(1,1,1)), Nsites)
    Yb = fill(Complex.(ones(1,1,1)), Nsites)
    Z = fill(Complex.(ones(1,1,1)), Nsites)
    Zb = fill(Complex.(ones(1,1,1)), Nsites)

    for p = Nsites:-1:2 # Do right normalization, from site Nsites to 2

        # Shape of M is: left bond - physical bond - right bond
        if size(M[p]) != size(Mb[p])
            throw(RuntimeError("Shapes of M[p] and Mb[p] must match!"))
        end
        
        # Set the p-th matrix to right normal form, and multiply the transform matrix to p-1
        Z[p], Zb[p], I, Ib = right_decomp(M[p], Mb[p]; chi_max=chi_max, timing=debug, method=method)
        M[p-1] = ncon([M[p-1],I], [[-1,-2,1],[1,-3]])
        Mb[p-1] = ncon([Mb[p-1],Ib], [[-1,-2,1],[1,-3]])

        # Construct R[p-1]. The indices of R is: left'' - left' - left
        R[p-1] = ncon([R[p], W[p], Zb[p], Z[p]],[[3,1,5],[-1,3,2,4],[-2,2,1],[-3,4,5]])

        for i = 1:NumNA
            RNA[i][p-1] = ncon([RNA[i][p], MNb[i][p], Z[p]], [[1,2],[-1,3,1],[-2,3,2]])
            RNAb[i][p-1] = ncon([RNAb[i][p], Zb[p], MN[i][p]], [[1,2],[-1,3,1],[-2,3,2]])
        end

    end

    # Normalize M[1] and Mb[1] so that the trial wave functions are bi-normalized
    ratio = 1/sqrt(ncon([M[0],Mb[0]],[[1,2,3],[1,2,3]]))
    M[1] *= ratio
    Mb[1] *= ratio

    # At this point we have turned M[2:end] to right normal form, and constructed R[2:end]
    # We start the sweep at site 0
    # The effective Hamiltonian at site [i] is the contraction of L[i], R[i], and W[i]
    
    Ekeep = Complex[]
    Hdifs = Float64[]

    k = 1

    while k <= numsweeps+1
        
        ##### final sweep is only for orthogonalization (disable updates)
        if k == numsweeps+1
            updateon = false
            dispon = 0
        end
        
        ###### Optimization sweep: left-to-right
        for p = 1:Nsites-1

            # Optimize at this step
            if updateon
                E, M[p], Mb[p] = eigLR(L[p], R[p], W[p], M[p], Mb[p],
                    sigma = sigma, use_sparse = true, tol = tol, timing = debug,
                    normalize_against = [(LNA[i][p],RNA[i][p],MNb[i][p],LNAb[i][p],RNAb[i][p],MN[i][p],Namp[i]) for i = 1:NumNA])    
                push!(Ekeep, E)
            end

            # Move the pointer one site to the right, and left-normalize the matrices at the currenter pointer
            Y[p], Yb[p], I, Ib = left_decomp(M[p], Mb[p]; chi_max=chi_max, timing=debug, method=method)

            M[p+1] = ncon([I,Z[p+1]], [[-1,1],[1,-2,-3]])
            Mb[p+1] = ncon([Ib,Zb[p+1]], [[-1,1],[1,-2,-3]])

            # Construct L[p+1]
            L[p+1] = ncon([L[p], W[p], Yb[p], Y[p]], [[3,1,5],[3,-1,2,4],[1,2,-2],[5,4,-3]])

            for i = 1:NumNA
                LNA[i][p+1] = ncon([LNA[i][p], MNb[i][p], Y[p]], [[1,2],[1,3,-1],[2,3,-2]])
                LNAb[i][p+1] = ncon([LNAb[i][p], Yb[p], MN[i][p]], [[1,2],[1,3,-1],[2,3,-2]])
            end
        
            ##### display energy
            if dispon == 2
                fprintln("Sweep: $k of $numsweeps, Loc: $p,Energy: $(@sprintf("%.3f",Ekeep[end]))")
            end
            
        end

        # Set Y[end]
        Y[end] = M[end]
        Yb[end] = Mb[end]
        
        ###### Optimization sweep: right-to-left
        for p = Nsites:-1:2

            # Optimize at this step
            if updateon
                E, M[p], Mb[p] = eigLR(L[p], R[p], W[p], M[p], Mb[p],
                    sigma = sigma, use_sparse = true, tol = tol, timing = debug,
                    normalize_against = [(LNA[i][p],RNA[i][p],MNb[i][p],LNAb[i][p],RNAb[i][p],MN[i][p],Namp[i]) for i = 1:NumNA])    
                push!(Ekeep, E)
            end

            # Move the pointer one site to the left, and right-normalize the matrices at the currenter pointer
            Z[p], Zb[p], I, Ib = right_decomp(M[p], Mb[p]; chi_max=chi_max, timing=debug, method=method)
            M[p-1] = ncon([Y[p-1],I], [[-1,-2,1],[1,-3]])
            Mb[p-1] = ncon([Yb[p-1],Ib], [[-1,-2,1],[1,-3]])

            # Construct R[p-1]. The indices of R is: left'' - left - left'
            R[p-1] = ncon([R[p], W[p], Zb[p], Z[p]],[[3,1,5],[-1,3,2,4],[-2,2,1],[-3,4,5]])

            for i = 1:NumNA
                RNA[i][p-1] = ncon([RNA[i][p], MNb[i][p], Z[p]], [[1,2],[-1,3,1],[-2,3,2]])
                RNAb[i][p-1] = ncon([RNAb[i][p], Zb[p], MN[i][p]], [[1,2],[-1,3,1],[-2,3,2]])
            end
        
            ##### display energy
            if dispon == 2
                fprintln("Sweep: $k of $numsweeps, Loc: $p, Energy: $(@sprintf("%.3f",Ekeep[end]))")
            end

        end

        # Set Z[1]
        Z[1] = M[1]
        Zb[1] = Mb[1]
        
        # Calculate <H^2>-<H>^2
        RR = Complex.ones((1,1,1,1))
        for p = Nsites:-1:1
            RR = ncon([Zb[p],RR,W[p],W[p],Z[p]], [[-3,2,1],[5,3,1,6],[-1,5,2,4],[-2,3,4,7],[-4,7,6]])
        end

        Hdif = abs(RR[1]-Ekeep[end]^2)
        push!(Hdifs, Hdif)

        if dispon >= 1
            fprintln("Sweep: $k of $numsweeps, Energy: $(fmtcpx(Ekeep[end])), Hdif: $(fmtf(Hdif)), Bonddim: $chi_max, tol: $tol")
        end

        cut = max(tol, eps(Float64)) * 10
        # Early termination if converged
        if abs(std(Ekeep[end-2*Nsites:end])) < cut && Hdif < cut:
            fprintln("Converged")
            k = numsweeps+1
        end

        k += 1

    end

    # Clean up memory
    foreach(finalize, [LNA, RNA, LNAb, RNAb, Namp, MN, MNb, L, R, RR])
    GC.gc()
            
    return Ekeep, Hdifs, Y, Yb, Z, Zb

end