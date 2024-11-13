using ITensors

#println("After using ITensors, methods of conj:")
#println(methods(conj))

include("Config.jl")
include("MPSlib.jl")
include("doDMRG_bb.jl")

#println("After using Config, methods of conj:")
#println(methods(conj))

function ranmat(shape...)
    return randn(shape...) .+ 1im.*randn(shape...)
end

function compare_DMRG_to_ED(mpo::MPO, sites::Sites; chi_max::Int=50, M::MPS=nothing, Mb::MPS=nothing, method::DM_Method=LR)

    L = length(sites)

    if isnothing(M)
        M = randomMPS(Complex, sites; linkdims=min(L, chi_max))
    end
    if isnothing(Mb)
        Mb = conj(M)'
    end

    chi = 10
    while chi < chi_max
        _,_,M,Mb,_,_ = doDMRG_bb(M, Mb, mpo, chi; numsweeps=1, method=method)
        chi = min(chi+5, chi_max)
    end

    # DMRG
    Ek, _, Y, _, _, _ = doDMRG_bb(M, Mb, mpo, chi_max; numsweeps=5, method=method)
    vD = MPS_to_Vector(Y)

    # ED
    op = MPO_to_Matrix(mpo)
    w,v = eigen(op)
    mw = sortperm(abs.(w))[1]

    angle = abs(dot(vD, v[:,mw])) / (norm(vD) * norm(v[:,mw]))

    fprintln("DMRG Energy $(Ek[end]) v.s. ED Energy $(w[mw])")
    fprintln("State overlap: $angle")

end

# def compare_DMRG_to_ED_X (mpo, k, chi_max=50, tol=1e-9, numsweeps=10, which = "SM", M=None, Mb=None, method="biortho", stop_if_not_converge = True, savename = None):

#     if savename is None:
#         savename = datetime.now().strftime("%m%d%Y_%H%M%S")

#     sz = 4
#     L = len(mpo)

#     if M is None:
#         M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
#     if Mb is None:
#         Mb = [m.conj() for m in M]

#     Ms, Mbs, Es = doDMRG_excited(M, Mb, mpo, chi_max, k, expected_gap=0.5, tol=tol, numsweeps=numsweeps, method=method, which=which, stop_if_not_converge=stop_if_not_converge, log_write=log_write)

#     # ED
#     op = MPO_to_Matrix(mpo)
#     w,v = LA.eig(op)

#     if which == "LR":
#         mws = np.argsort(-np.real(w))[:k]
#     else:
#         mws = np.argsort(np.abs(w))[:k]

#     for i in range(k):

#         mw = mws[i]
#         vD = MPS(Ms[i]).contract().flatten()

#         angle = np.abs(np.conj(vD) @ v[:,mw]) / (LA.norm(vD) * LA.norm(v[:,mw]))

#         log_write(f"DMRG Energy {Es[i]} v.s. ED Energy {w[mw]}")
#         log_write(f"State overlap: {angle}")

#     datas = {"E_dmrg":Es, "E_ed":w[mws], "v_ed": v[:,mws]}
#     for i in range(k):
#         for j in range(L):
#             datas[f"M{i}{j}"] = Ms[i][j]
#             datas[f"Mb{i}{j}"] = Mbs[i][j]
#     np.savez(savename, **datas)

function do_DMRGX_and_save(mpo::MPO, sites::Sites, k::Int, savename::String; chi_max::Int=50, tol::Float64=0., numsweeps::Int=10, M::MPS=undef, Mb::MPS=undef, method::DM_Method=LR)

    if isnothing(M)
        M = randomMPS(Complex, sites; linkdims=min(L, chi_max))
    end
    if isnothing(Mb)
        Mb = conj(M)'
    end

    doDMRG_excited(M, Mb, mpo, chi_max;
        k=k, expected_gap=1., tol=tol,
        numsweeps=numsweeps, dispon=2, debug=true, method=method,
        cut=1e-8, stop_if_not_converge=true, savename=savename, override=false)

end
# def do_DMRGX_and_save (mpo, k, chi_max=50, tol=1e-15, numsweeps=10, which = "SM", M=None, Mb=None, method="biortho", stop_if_not_converge = True, savename = None):
    
#     if savename is None:
#         savename = datetime.now().strftime("%m%d%Y_%H%M%S")

#     sz = 4
#     L = len(mpo)

#     if M is None:
#         M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
#     if Mb is None:
#         Mb = [m.conj() for m in M]

#     _, _, _ = doDMRG_excited(M, Mb, mpo, chi_max, k, expected_gap=0.5, tol=tol, numsweeps=numsweeps, method=method, which=which, stop_if_not_converge=stop_if_not_converge, log_write=log_write, savename=savename)

#     """
#     datas = {"E_dmrg":Es}
#     for i in range(k):
#         for j in range(L):
#             datas[f"M{i}{j}"] = Ms[i][j]
#             datas[f"Mb{i}{j}"] = Mbs[i][j]
#     np.savez(savename, **datas)
#     """

# def time_DMRG (mpo, which="LR", M=None, Mb=None):

#     sz = 4
#     L = len(mpo)

#     if M is None:
#         M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
#     if Mb is None:
#         Mb = [m.conj() for m in M]

#     t1 = time()
#     Ek, _, Y, _, _, _ = doDMRG_bb(M, Mb, mpo, 50, which=which, numsweeps=5, method="lrrho")
#     log_write(f"DMRG total cost {time()-t1}s")

# def plot_DMRG_spectrum (mpo, k, chi_max=50, numsweeps=10, which = "SM", M=None, Mb=None, method="biortho", stop_if_not_converge = True, savename = None):

#     sz = 4
#     L = len(mpo)

#     if M is None:
#         M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
#     if Mb is None:
#         Mb = [m.conj() for m in M]

#     # DMRG
#     Es,_,_ = doDMRG_excited()
#     vD = MPS(Y).contract().flatten()

#     # ED
#     op = MPO_to_Matrix(mpo)
#     w,v = LA.eig(op)

#     if which == "SR":
#         mw = np.argsort(np.real(w))[0]
#     elif which == "SM":
#         mw = np.argsort(np.abs(w))[0]
#     elif which == "LR":
#         mw = np.argsort(np.real(w))[-1]
#     elif isinstance(which, complex):
#         mw = np.argsort(np.abs(w-which))[0]
#     else:
#         raise Exception("Invalid instance which = {}".format(which))

#     angle = np.abs(np.conj(vD) @ v[:,mw]) / (LA.norm(vD) * LA.norm(v[:,mw]))

#     log_write(f"DMRG Energy {Ek[-1]} v.s. ED Energy {w[mw]}")
#     log_write(f"State overlap: {angle}")



# mpo = LindbladMPO(10, Operator({"ZZ":1, "X":0.5}), [Operator({"X":0.1,"Y":0.1j}), Operator({"Z":0.1})], dagger=True)
# #time_DMRG(mpo, which=-2+0j)
# #compare_DMRG_to_ED(mpo, -0.1+0j, chi_max=40, method="lrrho")
# compare_DMRG_to_ED_X(mpo, 5, chi_max=10, numsweeps=5, which="SM", method="lrrho", stop_if_not_converge=False)
# #do_DMRGX_and_save(mpo, 3, chi_max=50, tol=1e-6, savename="04292024", numsweeps=10, which="SM", method="lrrho", stop_if_not_converge=False)
# exit()


sz = 2
L = 5

"""
M = np.random.randn(sz,sz,sz) + 1j*np.random.randn(sz,sz,sz)
Mb = np.random.randn(sz,sz,sz) + 1j*np.random.randn(sz,sz,sz)

Y, Yb, I, Ib = right_decomp(M, Mb, chi_max=10)

log_write(np.shape(M), np.shape(Mb))
log_write(np.shape(Y), np.shape(Yb), np.shape(I), np.shape(Ib))

log_write(np.linalg.norm(ncon([Y,Yb],[[-1,1,2],[-2,1,2]])-np.eye(np.shape(Y)[0])))
log_write(np.linalg.norm(ncon([Y,I],[[1,-2,-3],[-1,1]])-M))
log_write(np.linalg.norm(ncon([Yb,Ib],[[1,-2,-3],[-1,1]])-Mb))
"""

sites = [Index(4, "Site$i") for i = 1:L]

M = randomMPS(ComplexF64, sites; linkdims=2)

### Non-Hermitian
#Mb = cat([ranmat(1,sz,sz)], [ranmat(sz,sz,sz) for _ = 2:L-1], [ranmat(sz,sz,1)])
Mb = conj(M)'
#W = randomMPO(sites)
diss = sqrt(0.1)
W = LindbladMPO(getOp(Dict("ZZ"=>1, "Z"=>0.7, "X"=>1.5)), [getOp(Dict("X"=>diss)), getOp(Dict("Y"=>diss)), getOp(Dict("Z"=>diss))], sites; dagger=false)

do_DMRGX_and_save(W, sites, 5, "Test1111Datas"; M=M, Mb=Mb, method=BB)

#compare_DMRG_to_ED(W, sites; M=M, Mb=Mb)

# exit()

# println("M:")
# println(M)
# println()

# println("Mb:")
# println(Mb)
# println()

# println("W:")
# println(W)
# println()


# ### Hermitian
# #Mb = [m.conj() for m in M]
# #W = [ranmatH(1,sz,sz,sz)] + [ranmatH(sz,sz,sz,sz) for _ in range(L-2)] + [ranmatH(sz,1,sz,sz)]

# E, Hdifs, _,_,_,_ = doDMRG_bb(M, Mb, W, 50; debug=true, numsweeps=1, method=LR)

"""
from matplotlib import pyplot as plt
plt.plot(np.arange(np.size(E))[50:], np.real(E)[50:])
plt.plot(np.arange(np.size(E))[50:], np.imag(E)[50:])
plt.show()
plt.close()

plt.plot(np.arange(np.size(Hdifs)), np.abs(Hdifs))
plt.show()
plt.close()
"""
