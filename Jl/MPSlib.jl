using LinearAlgebra: I

include("Config.jl")
include("OPlib.jl")

function OpSumMPS(op::Operator, sites::Sites)

    # Creates a MPS with the following bond dimension:
    # Axis 0 - indicates initial state
    # Axis 1 - indicates final state
    # All operators are added in a transition process 0 -> some axes -> 1
    # If the operator is one-site or two site, "some axes" is one axes
    # Otherwise, the dimensionality of "some axes" equals to (size of operator - 1)

    L = length(sites)

    thisax = 3 # Current axes
    dims = [] # dims[i]:dims[i+1] are the axes for operator i
    spans = [] # spans[i] records the span of operator i
    types = [] # types[i] records the Pauli string of operator i
    vals = [] # vals[i] records the coefficient of operator i
    signs = [] # signs[i] records the sign of operator i

    single_site = [0,0,0,0] # Record the amplitude of single-site operators

    for term in op
        span = len(term)
        if span > 1
            push!(dims, thisax)
            push!(spans, span)
            push!(types, term.inds)
            push!(vals, abs(term.coef)^(1/span))
            push!(signs, sign(term.coef))
            thisax += (span - 1)
        else
            single_site[term.inds[1]] += val
        end
    end

    # The bulk MPS tensor
    M0 = ComplexF64.(zeros(thisax, 4, thisax))
    M0[1, 1, 1] = 1 # Initial state to itself
    M0[2, 1, 2] = 1 # Final state to itself

    for i = 1:4
        M0[1, i, 2] = single_site[i] # Initial state can directly hop to final state, yielding a single-site operator
    end

    for (i,dim) in enumerate(dims)
        span = spans[i]
        val = vals[i]
        type_indices = types[i]
        M0[1, type_indices[1], dim] = val * signs[i] # Initial state to transition axes
        M0[dim + span - 2, type_indices[end], 1] = val # Transition axes to final state
        for j in 1:span-2
            M0[dim + j - 1, type_indices[j], dim + j] = val # Within transition axes
        end
    end

    if L == 1
        return MPS([ITensor(M0[1,:,2], sites[1])])
    else
        links = [Index(thisax, "$i-link-$(i+1)" for i=1:L-1)]
        ML = ITensor(M0[1,:,:], sites[1], links[1])
        MR = ITensor(M0[:,:,2], links[end], sites[end])
        return MPS(vcat([ML], [ITensor(M0, links[i-1], sites[i], links[i]) for i = 2:L-1], [MR]))
    end

end

"""
    sizeMPO(sites::Array{Index})

Returns a MPO that measures the size of an operator.
"""

function sizeMPO(sites::Sites)

    M0 = ComplexF64.(zeros(4, 4, 2, 2))
    M0[:,:, 1,1] = Matrix(I, 4, 4) # Initial state to itself
    M0[:,:, 2,2] = Matrix(I, 4, 4) # Final state to itself
    M0[:,:, 1,2] = Diagonal([0,1,1,1]) # Operator size

    L = length(sites)

    if L == 1
        return MPO([ITensor(M0[:,:,1,2], sites[1]', sites[1])])
    else
        links = [Index(thisax, "$i-link-$(i+1)" for i=1:L-1)]
        ML = ITensor(M0[:,:,1,:], sites[1]', sites[1], links[1])
        MR = ITensor(M0[:,:,:,2], sites[end]', sites[end], links[end])
        return MPO(vcat([ML], [ITensor(M0, sites[i]', sites[i], links[i-1], links[i]) for i = 2:L-1], [MR]))
    end

end

# Realized in a similar way as OpSumMPS
function LindbladMPO(H, Lis, sites; dagger = false)

    thisax = Ref(3) # Current axes
    dims = Ref([]) # dims[i]:dims[i+1] are the axes for operator i
    mats = Ref([])

    single_site = Ref(ComplexF64.(zeros(4,4))) # Record the amplitude of single-site operators

    function add_mats(mat)
        span = length(mat)
        if span > 1
            push!(dims[], thisax[])
            push!(mats[], mat)
            thisax[] += (span-1)
        elseif span == 1
            single_site[] += mat[1]
        end
    end

    # Hamiltonian
    for term in H
        if dagger
            add_mats(getMats(term; type="L", add_coef=1im))
            add_mats(getMats(term; type="R", add_coef=-1im))
        else
            add_mats(getMats(term; type="L", add_coef=-1im))
            add_mats(getMats(term; type="R", add_coef=1im))
        end
    end
        
    function mul_mats(mat1, mat2)
        return [mat1[i]*mat2[i] for i in eachindex(mat1)]
    end

    # Jump operators
    for Li in Lis
        for tL0 in Li
            for tR in Li
                tL = conj(tL0)
                if dagger
                    add_mats(mul_mats(getMats(tL; type="L"), getMats(tR; type="R")))
                else
                    add_mats(mul_mats(getMats(tL; type="R"), getMats(tR; type="L")))
                end
                add_mats(mul_mats(getMats(tL; type="L", add_coef=-1/2), getMats(tR; type="L")))
                add_mats(mul_mats(getMats(tR; type="R"), getMats(tL; type="R", add_coef=-1/2)))
            end
        end
    end

    # The bulk MPO tensor
    M0 = ComplexF64.(zeros(4, 4, thisax[], thisax[]))
    M0[:,:, 1,1] = Matrix(I, 4, 4) # Initial state to itself
    M0[:,:, 2,2] = Matrix(I, 4, 4) # Final state to itself
    M0[:,:, 1,2] = single_site[] # Initial state can directly hop to final state, yielding a single-site operator
    for (i,dim) in enumerate(dims[])
        mat = mats[][i]
        span = length(mat)
        M0[:,:, 1,dim] = mat[1] # Initial state to transition axes
        M0[:,:, dim+span-2,2] = mat[end] # Transition axes to final state
        for j = 1:span-2
            M0[:,:, dim+j-1,dim+j] = mat[j] # Within transition axes   
        end
    end
    
    if L == 1
        return MPO([ITensor(M0[:,:,1,2], sites[1]', sites[1])])
    else
        links = [Index(thisax[], "$i-link-$(i+1)") for i=1:L-1]
        ML = ITensor(M0[:,:,1,:], sites[1]', sites[1], links[1])
        MR = ITensor(M0[:,:,:,2], sites[end]', sites[end], links[end])
        return MPO(vcat([ML], [ITensor(M0, sites[i]', sites[i], links[i-1], links[i]) for i = 2:L-1], [MR]))
    end

end

function printMPSO(W)
    H = reduce(*, W)
    Winds = inds(H)
    fprintln("Indices: $inds")
    fprintln(Array(H, Winds...))
end

function MPS_to_Vector(W)
    H = reduce(*, W)
    Winds = inds(H)
    comb = combiner(Winds...)
    mati = inds(comb)[1]
    return Array(H*comb, mati)
end

function MPO_to_Matrix(W)
    H = reduce(*, W)
    Winds = [i for i in inds(H) if plev(i)==0]
    comb = combiner(Winds...)
    mati = inds(comb)[1]
    return Array(H*comb*comb', mati', mati)
end

function printInds(W)
    for (p,A) in enumerate(W)
        fprintln("Pos $p, indices $(inds(A))")
    end
end