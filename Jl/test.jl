include("NHDMRG.jl")
using .NHDMRG

v = [1,2,3,4]
w = [4,3,2,1]

M1 = w*transpose(v)

Y, Yb = decomp_biortho_on_rho(M1)
println(size(Y))
println(size(Yb))
Y = vec(Y)
Yb = vec(Yb)
println(Y./Y[1])
println(Yb./Yb[1])