from BiorthoLib import left_decomp, right_decomp
from doDMRG_bb import *
from MPSlib import *
from OPlib import *
from time import time

def ranmat (*shape):
    return np.random.randn(*shape) + 1j*np.random.randn(*shape)

def ranmatH (*shape):
    rm = ranmat(*shape)
    return (rm + np.swapaxes(rm.conj(),-1,-2))/2

def compare_DMRG_to_ED (mpo, which="LR", M=None, Mb=None):

    sz = 4
    L = len(mpo)

    if M is None:
        M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
    if Mb is None:
        Mb = [m.conj() for m in M]

    # DMRG
    Ek, _, Y, _, _, _ = doDMRG_bb(M, Mb, mpo, 50, which=which, numsweeps=5, method="lrrho")
    vD = MPS(Y).contract().flatten()

    # ED
    op = MPO_to_Matrix(mpo)
    w,v = LA.eig(op)

    if which == "SR":
        mw = np.argsort(np.real(w))[0]
    elif which == "SM":
        mw = np.argsort(np.abs(w))[0]
    elif which == "LR":
        mw = np.argsort(np.real(w))[-1]
    elif isinstance(which, complex):
        mw = np.argsort(np.abs(w-which))[0]
    else:
        raise Exception("Invalid instance which = {}".format(which))

    angle = np.abs(np.conj(vD) @ v[:,mw]) / (LA.norm(vD) * LA.norm(v[:,mw]))

    print(f"DMRG Energy {Ek[-1]} v.s. ED Energy {w[mw]}")
    print(f"State overlap: {angle}")

def time_DMRG (mpo, which="LR", M=None, Mb=None):

    sz = 4
    L = len(mpo)

    if M is None:
        M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
    if Mb is None:
        Mb = [m.conj() for m in M]

    t1 = time()
    Ek, _, Y, _, _, _ = doDMRG_bb(M, Mb, mpo, 50, which=which, numsweeps=5, method="lrrho")
    print(f"DMRG total cost {time()-t1}s")

def plot_DMRG_spectrum (mpo, M=None, Mb=None):

    sz = 4
    L = len(mpo)

    if M is None:
        M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
    if Mb is None:
        Mb = [m.conj() for m in M]

    # DMRG
    Ek, _, Y, _, _, _ = doDMRG_bb(M, Mb, mpo, 50, which=which, numsweeps=5, method="lrrho")
    vD = MPS(Y).contract().flatten()

    # ED
    op = MPO_to_Matrix(mpo)
    w,v = LA.eig(op)

    if which == "SR":
        mw = np.argsort(np.real(w))[0]
    elif which == "SM":
        mw = np.argsort(np.abs(w))[0]
    elif which == "LR":
        mw = np.argsort(np.real(w))[-1]
    elif isinstance(which, complex):
        mw = np.argsort(np.abs(w-which))[0]
    else:
        raise Exception("Invalid instance which = {}".format(which))

    angle = np.abs(np.conj(vD) @ v[:,mw]) / (LA.norm(vD) * LA.norm(v[:,mw]))

    print(f"DMRG Energy {Ek[-1]} v.s. ED Energy {w[mw]}")
    print(f"State overlap: {angle}")

mpo = LindbladMPO(6, Operator({"ZZ":1, "X":0.5}), [Operator({"X":0.1,"Y":0.1j}), Operator({"Z":0.1})], dagger=False)
time_DMRG(mpo, which=-2+0j)
#compare_DMRG_to_ED(mpo, -2+0j)
exit()


sz = 2
L = 8

"""
M = np.random.randn(sz,sz,sz) + 1j*np.random.randn(sz,sz,sz)
Mb = np.random.randn(sz,sz,sz) + 1j*np.random.randn(sz,sz,sz)

Y, Yb, I, Ib = right_decomp(M, Mb, chi_max=10)

print(np.shape(M), np.shape(Mb))
print(np.shape(Y), np.shape(Yb), np.shape(I), np.shape(Ib))

print(np.linalg.norm(ncon([Y,Yb],[[-1,1,2],[-2,1,2]])-np.eye(np.shape(Y)[0])))
print(np.linalg.norm(ncon([Y,I],[[1,-2,-3],[-1,1]])-M))
print(np.linalg.norm(ncon([Yb,Ib],[[1,-2,-3],[-1,1]])-Mb))
"""

M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]

### Non-Hermitian
Mb = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
W = [ranmat(1,sz,sz,sz)] + [ranmat(sz,sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,1,sz,sz)]

### Hermitian
#Mb = [m.conj() for m in M]
#W = [ranmatH(1,sz,sz,sz)] + [ranmatH(sz,sz,sz,sz) for _ in range(L-2)] + [ranmatH(sz,1,sz,sz)]

E, Hdifs, _,_,_,_ = doDMRG_bb(M, Mb, W, 50, debug=False, numsweeps=10, method="biortho")

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