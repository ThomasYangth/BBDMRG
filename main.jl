from BiorthoLib import left_decomp, right_decomp
from doDMRG_bb import *
from MPSlib import *
from OPlib import *
from time import time
from datetime import datetime


def log_write (content, end = "\n"):
    print(content, end=end, flush=True)
    if end == "\n":
        print(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+"::", end = " ", flush = True)

def ranmat (*shape):
    return np.random.randn(*shape) + 1j*np.random.randn(*shape)

def ranmatH (*shape):
    rm = ranmat(*shape)
    return (rm + np.swapaxes(rm.conj(),-1,-2))/2

def compare_DMRG_to_ED (mpo, which="LR", chi_max=50, M=None, Mb=None, method="biortho"):

    sz = 4
    L = len(mpo)

    if M is None:
        M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
    if Mb is None:
        Mb = [m.conj() for m in M]

    chi = 10
    while chi < chi_max:
        _,_,M,Mb,_,_ = doDMRG_bb(M, Mb, mpo, chi, which=which, numsweeps=1, method=method)
        chi = min(chi+5, chi_max)

    # DMRG
    Ek, _, Y, _, _, _ = doDMRG_bb(M, Mb, mpo, chi_max, which=which, numsweeps=5, method=method)
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

    log_write(f"DMRG Energy {Ek[-1]} v.s. ED Energy {w[mw]}")
    log_write(f"State overlap: {angle}")

def compare_DMRG_to_ED_X (mpo, k, chi_max=50, tol=1e-9, numsweeps=10, which = "SM", M=None, Mb=None, method="biortho", stop_if_not_converge = True, savename = None):

    if savename is None:
        savename = datetime.now().strftime("%m%d%Y_%H%M%S")

    sz = 4
    L = len(mpo)

    if M is None:
        M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
    if Mb is None:
        Mb = [m.conj() for m in M]

    Ms, Mbs, Es = doDMRG_excited(M, Mb, mpo, chi_max, k, expected_gap=0.5, tol=tol, numsweeps=numsweeps, method=method, which=which, stop_if_not_converge=stop_if_not_converge, log_write=log_write)

    # ED
    op = MPO_to_Matrix(mpo)
    w,v = LA.eig(op)

    if which == "LR":
        mws = np.argsort(-np.real(w))[:k]
    else:
        mws = np.argsort(np.abs(w))[:k]

    for i in range(k):

        mw = mws[i]
        vD = MPS(Ms[i]).contract().flatten()

        angle = np.abs(np.conj(vD) @ v[:,mw]) / (LA.norm(vD) * LA.norm(v[:,mw]))

        log_write(f"DMRG Energy {Es[i]} v.s. ED Energy {w[mw]}")
        log_write(f"State overlap: {angle}")

    datas = {"E_dmrg":Es, "E_ed":w[mws], "v_ed": v[:,mws]}
    for i in range(k):
        for j in range(L):
            datas[f"M{i}{j}"] = Ms[i][j]
            datas[f"Mb{i}{j}"] = Mbs[i][j]
    np.savez(savename, **datas)

def do_DMRGX_and_save (mpo, k, chi_max=50, tol=1e-15, numsweeps=10, which = "SM", M=None, Mb=None, method="biortho", stop_if_not_converge = True, savename = None):
    
    if savename is None:
        savename = datetime.now().strftime("%m%d%Y_%H%M%S")

    sz = 4
    L = len(mpo)

    if M is None:
        M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
    if Mb is None:
        Mb = [m.conj() for m in M]

    _, _, _ = doDMRG_excited(M, Mb, mpo, chi_max, k, expected_gap=0.5, tol=tol, numsweeps=numsweeps, method=method, which=which, stop_if_not_converge=stop_if_not_converge, log_write=log_write, savename=savename)

    """
    datas = {"E_dmrg":Es}
    for i in range(k):
        for j in range(L):
            datas[f"M{i}{j}"] = Ms[i][j]
            datas[f"Mb{i}{j}"] = Mbs[i][j]
    np.savez(savename, **datas)
    """

def time_DMRG (mpo, which="LR", M=None, Mb=None):

    sz = 4
    L = len(mpo)

    if M is None:
        M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
    if Mb is None:
        Mb = [m.conj() for m in M]

    t1 = time()
    Ek, _, Y, _, _, _ = doDMRG_bb(M, Mb, mpo, 50, which=which, numsweeps=5, method="lrrho")
    log_write(f"DMRG total cost {time()-t1}s")

def plot_DMRG_spectrum (mpo, k, chi_max=50, numsweeps=10, which = "SM", M=None, Mb=None, method="biortho", stop_if_not_converge = True, savename = None):

    sz = 4
    L = len(mpo)

    if M is None:
        M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
    if Mb is None:
        Mb = [m.conj() for m in M]

    # DMRG
    Es,_,_ = doDMRG_excited()
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

    log_write(f"DMRG Energy {Ek[-1]} v.s. ED Energy {w[mw]}")
    log_write(f"State overlap: {angle}")

if __name__ == "__main__":

    mpo = LindbladMPO(10, Operator({"ZZ":1, "X":0.5}), [Operator({"X":0.1,"Y":0.1j}), Operator({"Z":0.1})], dagger=True)
    #time_DMRG(mpo, which=-2+0j)
    #compare_DMRG_to_ED(mpo, -0.1+0j, chi_max=40, method="lrrho")
    compare_DMRG_to_ED_X(mpo, 5, chi_max=10, numsweeps=5, which="SM", method="lrrho", stop_if_not_converge=False)
    #do_DMRGX_and_save(mpo, 3, chi_max=50, tol=1e-6, savename="04292024", numsweeps=10, which="SM", method="lrrho", stop_if_not_converge=False)
    exit()


sz = 2
L = 8

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

M = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]

### Non-Hermitian
Mb = [ranmat(1,sz,sz)] + [ranmat(sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,sz,1)]
W = [ranmat(1,sz,sz,sz)] + [ranmat(sz,sz,sz,sz) for _ in range(L-2)] + [ranmat(sz,1,sz,sz)]

### Hermitian
#Mb = [m.conj() for m in M]
#W = [ranmatH(1,sz,sz,sz)] + [ranmatH(sz,sz,sz,sz) for _ in range(L-2)] + [ranmatH(sz,1,sz,sz)]

E, Hdifs, _,_,_,_ = doDMRG_bb(M, Mb, W, 50, debug=False, numsweeps=3, method="biortho")

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
