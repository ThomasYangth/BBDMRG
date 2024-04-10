# -*- coding: utf-8 -*-
"""
mainDMRG_MPO.py
---------------------------------------------------------------------
Script file for initializing the Hamiltonian of a 1D spin chain as an MPO \
before passing to the DMRG routine.

    by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 19/1/2019
"""

#### Preamble
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from ncon import ncon
from doDMRG_MPO import doDMRG_MPO 

##### Example 1: XX model #############
#######################################

##### Set bond dimensions and simulation options
chi = 16
Nsites = 50

OPTS_numsweeps = 4 # number of DMRG sweeps
OPTS_dispon = 2 # level of output display
OPTS_updateon = True # level of output display
OPTS_maxit = 2 # iterations of Lanczos method
OPTS_krydim = 4 # dimension of Krylov subspace

#### Define Hamiltonian MPO (quantum XX model)
chid = 2
sP = np.sqrt(2)*np.array([[0, 0],[1, 0]])
sM = np.sqrt(2)*np.array([[0, 1],[0, 0]])
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0,-1]])
sI = np.array([[1, 0], [0, 1]])
M = np.zeros([4,4,chid,chid])
M[0,0,:,:] = sI; M[3,3,:,:] = sI
M[0,1,:,:] = sM; M[1,3,:,:] = sP
M[0,2,:,:] = sP; M[2,3,:,:] = sM
ML = np.array([1,0,0,0]).reshape(4,1,1) #left MPO boundary
MR = np.array([0,0,0,1]).reshape(4,1,1) #right MPO boundary

#### Initialize MPS tensors
A = [0 for x in range(Nsites)]
A[0] = np.random.rand(1,chid,min(chi,chid))
for k in range(1,Nsites):
    A[k] = np.random.rand(A[k-1].shape[2],chid,min(min(chi,A[k-1].shape[2]*chid),chid**(Nsites-k-1)))

#### Do DMRG sweeps (2-site approach)
En1, A, sWeight, B = doDMRG_MPO(A,ML,M,MR,chi, numsweeps = OPTS_numsweeps, dispon = OPTS_dispon, 
                                updateon = OPTS_updateon, maxit = OPTS_maxit, krydim = OPTS_krydim)

#### Increase bond dim and reconverge
chi = 32
En2, A, sWeight, B = doDMRG_MPO(A,ML,M,MR,chi, numsweeps = OPTS_numsweeps, dispon = OPTS_dispon, 
                                updateon = OPTS_updateon, maxit = OPTS_maxit, krydim = OPTS_krydim)

#### Compare with exact results (computed from free fermions)
H = np.diag(np.ones(Nsites-1),k=1) + np.diag(np.ones(Nsites-1),k=-1)
D = LA.eigvalsh(H)
EnExact = 2*sum(D[D < 0])

##### Plot results
plt.figure(1)
plt.yscale('log')
plt.plot(range(len(En1)), En1 - EnExact, 'b', label="chi = 16")
plt.plot(range(len(En2)), En2 - EnExact, 'r', label="chi = 32")
plt.legend()
plt.title('DMRG for XX model')
plt.xlabel('Update Step')
plt.ylabel('Ground Energy Error')
plt.show()

#### Compute 2-site reduced density matrices, local energy profile
rhotwo = [0 for x in range(Nsites-1)]
hamloc = (np.real(np.kron(sX,sX) + np.kron(sY,sY))).reshape(2,2,2,2)
Enloc = np.zeros(Nsites-1)
for k in range(Nsites-1):
    rhotwo[k] = ncon([A[k],np.conj(A[k]),A[k+1],np.conj(A[k+1]),sWeight[k+2],
                     sWeight[k+2]],[[1,-3,2],[1,-1,3],[2,-4,4],[3,-2,5],[4,6],[5,6]])
    Enloc[k] = ncon([hamloc,rhotwo[k]],[[1,2,3,4],[1,2,3,4]])