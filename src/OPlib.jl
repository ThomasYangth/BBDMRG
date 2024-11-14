export Term, Operator, getOp, LindbladMPO, LindbladMPO_W

const Term = NamedTuple{(:coef, :inds), Tuple{ComplexF64, Vector{Int}}}
const Operator = Vector{Term}

function Base.conj(t::Term)
    return Term((coef=conj(t.coef), inds=t.inds))
end

function Base.length(t::Term)
    return length(t.inds)
end

IND_MAP = Dict(1=>"I", 2=>"X", 3=>"Y", 4=>"Z")
REV_IND_MAP = Dict('I'=>1, 'X'=>2, 'Y'=>3, 'Z'=>4)
PAULIS = Dict(
    "I"=>ComplexF64.([1 0; 0 1]),
    "X"=>ComplexF64.([0 1; 1 0]),
    "Y"=>ComplexF64.([0 -1im; 1im 0]),
    "Z"=>ComplexF64.([1 0; 0 -1]),
    "XL"=>ComplexF64.([0 1 0 0; 1 0 0 0; 0 0 0 -1im; 0 0 1im 0]),
    "XR"=>ComplexF64.([0 1 0 0; 1 0 0 0; 0 0 0 1im; 0 0 -1im 0]),
    "YL"=>ComplexF64.([0 0 1 0; 0 0 0 1im; 1 0 0 0; 0 -1im 0 0]),
    "YR"=>ComplexF64.([0 0 1 0; 0 0 0 -1im; 1 0 0 0; 0 1im 0 0]),
    "ZL"=>ComplexF64.([0 0 0 1; 0 0 -1im 0; 0 1im 0 0; 1 0 0 0]),
    "ZR"=>ComplexF64.([0 0 0 1; 0 0 1im 0; 0 -1im 0 0; 1 0 0 0]),
    "IL"=>ComplexF64.([1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]),
    "IR"=>ComplexF64.([1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])
)

include("Config.jl")

function Base.show(io::IO, t::Term)
    # Customize the display format
    print(io, "("*fmtcpx(t.coef)*")")
    for i in t.inds
        print(io, IND_MAP[i])
    end
end

function getMats(t::Term; type::String = "", add_coef::Number = 1)
    if !(type in ["","L","R"])
        throw(ArgumentError("type must be either ''(Direct), 'L'eft, or 'R'ight!"))
    end
    if length(t) == 0
        return []
    end
    add_coef = ComplexF64(add_coef)
    coef = t.coef * add_coef
    sgn = coef / abs(coef)
    val = abs(coef)^(1/length(t))
    lst = [val*PAULIS[IND_MAP[ind]*type] for ind in t.inds]
    lst[1] *= sgn
    return lst
end

function getOp(terms::Dict{String, <:Number})
    op = Term[]
    for (type,val) in terms
        push!(op, Term((coef=ComplexF64(val), inds=[REV_IND_MAP[uppercase(c)] for c in type])))
    end
    return op
end