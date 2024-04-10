import numpy as np
from BiorthoLib import left_decomp, right_decomp
from doDMRG_bb import doDMRG_bb
from ncon import ncon

def ranmat (*shape):
    return np.random.randn(*shape) + 1j*np.random.randn(*shape)

def ranmatH (*shape):
    rm = ranmat(*shape)
    return (rm + np.swapaxes(rm.conj(),-1,-2))/2

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