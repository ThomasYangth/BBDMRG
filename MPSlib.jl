from Config import *
from OPlib import Operator

class MPS:

    def __init__ (self, Mlist = None, M0 = None, ML = None, MR = None, L = 0):

        if Mlist is None and M0 is None:
            raise Exception("At least one of Mlist and M0 must not be none!")
        elif M0 is not None:
            if Mlist is not None:
                raise Exception("Initialization of MPS got conflicting arguments 'Mlist' and 'M0'. Please use only one of them.")
            if L == 0:
                raise Exception("Please provide a positive length L to initialize with M0!")
            if L == 1:
                self.Mlist = [np.copy(M0)]
            else:
                if ML is None or MR is None:
                    raise Exception("Provide ML and MR for initialization!")
                self.Mlist = [np.copy(ML)] + [np.copy(M0) for _ in range(L-2)] + [np.copy(MR)]
        else:
            self.Mlist = Mlist
        
        if L > 2:
            if len(np.shape(self.Mlist[0])) == 2:
                self.Mlist[0] = self.Mlist[0][np.newaxis,:,:]

            if len(np.shape(self.Mlist[-1])) == 2:
                self.Mlist[-1] = self.Mlist[-1][:,:,np.newaxis]
            

    def copy (self):
        return MPS([np.copy(M) for M in self.Mlist])

    def __len__ (self):
        return len(self.Mlist)
    
    def __imul__ (self, ratio):
        if len(self) == 0:
            return
        sgn = ratio / np.abs(ratio)
        val = np.abs(ratio) ** (1/len(self))
        self.Mlist[0] *= val*sgn
        for i in range(1, len(self)):
            self.Mlist[i] *= val
    
    def __mul__ (self, ratio):
        mps1 = self.copy()
        mps1 *= ratio
        return mps1
    
    def conj (self):

        return MPS([M.conj() for M in self.Mlist])
    
    def __matmul__ (self, other):
        if len(self) != len(other):
            raise Exception("Only MPS of the same length can be multiplied!")
        if len(self) == 0:
            return 0
        elif len(self) == 1:
            return np.trace(self[0].conj().T @ other[0])
        
        L = ncon([self[0].conj(), other[0]],[[1,-1],[1,-2]])
        for i in range(1, len(self)-1):
            L = ncon([L, self[i].conj(), other[i]], [[1,2],[1,3,-1],[2,3,-2]])
        L = ncon([L,self[-1].conj(),other[-1].conj()],[[1,2],[1,3],[2,3]])
        return L.flatten()[0]
    
    def __iadd__ (self, other):

        if len(self) != len(other):
            raise Exception("Only MPS of the same length can be added!")
        if len(self) == 0:
            return
        elif len(self) == 1:
            newM = np.zeros((1,4,1), dtype=complex)
            newM[0,:,0] = self[0][0,:,0] + other[0][0,:,0]
        else:

            for i in range(len(self)):
 
                MA = self[i]
                MB = other[i]
                dAL,_,dAR = np.shape(MA)
                dBL,_,dBR = np.shape(MB)
                nM = np.zeros((dAL+dBL,4,dAR+dBR), dtype=complex)
                nM[:dAL,:,:dAR] = MA
                nM[dAL:,:,dAR:] = MB

                if i == 0:
                    nM = np.tensordot(np.ones((1,2)), nM, axes=1)
                elif i == len(self)-1:
                    nM = np.tensordot(nM, np.ones((2,1)), axes=1)

                self[i] = nM

    def __add__ (self, other):
        mps1 = self.copy()
        mps1 += other
        return mps1
    
    def norm (self):
        return np.sqrt(np.real(self @ self))
    
    def normalize (self):
        if len(self) == 0:
            return
        coef = self.norm() ** (1/len(self))
        for i in range(len(self)):
            self[i] = self[i]*coef
        return self
    
    def contract (self):

        L = len(self)
        return ncon([self[i] for i in range(L)], [[i+1, -(i+1), (i+1)%L+1] for i in range(L)])

    def __getitem__ (self, key):
        return self.Mlist[key]

    def __setitem__ (self, key, value):
        self.Mlist[key] = value

    def __iter__ (self):
        for M in self.Mlist:
            yield M

    def __str__(self):
        s = ""
        for i in range(len(self)):
            s += f"Tensor {i+1}\n"
            s += str(self.Mlist[i]) + "\n"
        return s
    
    def __repr__(self):
        return self.__str__()

    def asdict (self, name = "M"):
        dct = {}
        for i,M in enumerate(self.Mlist):
            dct[f"{name}{i}"] = M
        return dct
    
class MPO:

    def __init__ (self, Wlist = None, W0 = None, WL = None, WR = None, L = 0):

        if Wlist is None and W0 is None:
            raise Exception("At least one of Wlist and W0 must not be none!")
        elif W0 is not None:
            if Wlist is not None:
                raise Exception("Initialization of WPS got conflicting arguments 'Wlist' and 'W0'. Please use only one of them.")
            if L == 0:
                raise Exception("Please provide a positive length L to initialize with W0!")
            if L == 1:
                self.Wlist = [np.copy(W0)]
            else:
                if WL is None or WR is None:
                    raise Exception("Provide WL and WR for initialization!")
                self.Wlist = [np.copy(WL)] + [np.copy(W0) for _ in range(L-2)] + [np.copy(WR)]
        else:
            self.Wlist = Wlist
        
        if L > 2:
            if len(np.shape(self.Wlist[0])) == 3:
                self.Wlist[0] = self.Wlist[0][np.newaxis,:,:,:]

            if len(np.shape(self.Wlist[-1])) == 3:
                self.Wlist[-1] = self.Wlist[-1][:,np.newaxis,:,:]

    def copy (self):
        return MPO([np.copy(W) for W in self.Wlist])

    def transpose (self):
        mpo1 = self.copy()
        for i in range(len(mpo1)):
            mpo1.Wlist[i] = mpo1.Wlist[i].transpose([0,1,3,2])
        return mpo1

    def __len__ (self):
        return len(self.Wlist)
    
    def __imul__ (self, ratio):
        if len(self) == 0:
            return
        sgn = ratio / np.abs(ratio)
        val = np.abs(ratio) ** (1/len(self))
        self.Wlist[0] *= val*sgn
        for i in range(1, len(self)):
            self.Wlist[i] *= val
    
    def __mul__ (self, ratio):
        mpo1 = self.copy()
        mpo1 *= ratio
        return mpo1
    
    def __matmul__ (self, other):
        
        if len(self) != len(other):
            raise Exception("Only MPS/MPO of the same length can be multiplied!")
        
        if len(self) == 0:
            return 0

        raise Exception("MPO.__matmul__ is not realized yet!")
    
    def __iadd__ (self, other):

        if len(self) != len(other):
            raise Exception("Only MPS of the same length can be added!")
        if len(self) == 0:
            return
        elif len(self) == 1:
            newW = np.zeros((1,1,4,4), dtype=complex)
            newW[0,0,:,:] = self[0][0,0,:,:] + other[0][0,0,:,:]
        else:

            for i in range(len(self)):
 
                WA = self[i]
                WB = other[i]
                dAL,dAR,_,_ = np.shape(WA)
                dBL,dBR,_,_ = np.shape(WB)
                nW = np.zeros((dAL+dBL,dAR+dBR,4,4), dtype=complex)
                nW[:dAL,:dAR,:,:] = WA
                nW[dAL:,dAR:,:,:] = WB

                if i == 0:
                    nW = ncon([np.ones((1,2)),nW], [[-1,1],[1,-2,-3,-4]])
                elif i == len(self)-1:
                    nW = ncon([nW,np.ones((2,1))], [[-1,1,-3,-4],[1,-2]])

                self[i] = nW

    def __add__ (self, other):
        mpo1 = self.copy()
        mpo1 += other
        return mpo1
    
    def contract (self):

        L = len(self)
        return ncon([self[i] for i in range(L)], [[i+1, (i+1)%L+1, -(i+1), -(i+L+1)] for i in range(L)])

    def __getitem__ (self, key):
        return self.Wlist[key]

    def __setitem__ (self, key, value):
        self.Wlist[key] = value

    def __iter__ (self):
        for W in self.Wlist:
            yield W

    def __str__(self):
        s = ""
        for i in range(len(self)):
            s += f"Tensor {i+1}\n"
            s += str(self.Wlist[i]) + "\n"
        return s
    
    def __repr__(self):
        return self.__str__()

    def asdict (self, name = "W"):
        dct = {}
        for i,W in enumerate(self.Wlist):
            dct[f"{name}{i}"] = W
        return dct
    
def expectation (psiL, psiR, *ops):
    L = len(psiL)
    if len(psiR) != L or np.any([len(op) != L for op in ops]):
        raise Exception("expectation() requires the length of all objects to be the same!")
    opnum = len(ops)
    T = np.ones([1]*(2+opnum))
    for p in range(1,L):
        T = ncon([T, psiL[p].conj(), psiR[p]] + [op[p] for op in ops], [])
    return T.flatten()[0]


def OpSumMPS(L, op):

    # Creates a MPS with the following bond dimension:
    # Axis 0 - indicates initial state
    # Axis 1 - indicates final state
    # All operators are added in a transition process 0 -> some axes -> 1
    # If the operator is one-site or two site, "some axes" is one axes
    # Otherwise, the dimensionality of "some axes" equals to (size of operator - 1)

    thisax = 2 # Current axes
    dims = [] # dims[i]:dims[i+1] are the axes for operator i
    spans = [] # spans[i] records the span of operator i
    types = [] # types[i] records the Pauli string of operator i
    vals = [] # vals[i] records the coefficient of operator i
    signs = [] # signs[i] records the sign of operator i

    single_site = np.zeros(4) # Record the amplitude of single-site operators

    for term in op:
        span = len(term)
        if span > 1:
            dims.append(thisax)
            spans.append(span)
            types.append(term.inds)
            vals.append(abs(term.coef) ** (1/span))
            signs.append(np.sign(term.coef))
            thisax += (span - 1)
        else:
            single_site[term.coef[0]] += val

    # The bulk MPS tensor
    M0 = np.zeros((thisax, 4, thisax), dtype=complex)
    M0[0, 0, 0] = 1 # Initial state to itself
    M0[1, 0, 1] = 1 # Final state to itself
    for i in range(4):
        M0[0, i, 1] = single_site[i] # Initial state can directly hop to final state, yielding a single-site operator
    for i, dim in enumerate(dims):
        span = spans[i]
        val = vals[i]
        type_indices = types[i]
        M0[0, type_indices[0], dim] = val * signs[i] # Initial state to transition axes
        M0[dim + span - 2, type_indices[-1], 1] = val # Transition axes to final state
        for j in range(1, span - 1):
            M0[dim + j - 1, type_indices[j], dim + j] = val # Within transition axes

    if L == 1:
        return [M0[np.array([0])[:,np.newaxis], :, np.array([1])[np.newaxis,:]]]
    else:
        # The edge MPS
        ML = M0[[0], :, :]
        MR = M0[:, :, [1]]
        return MPS([ML] + [M0]*(L-2) + [MR])

def sizeMPO (L):

    M0 = np.zeros((2, 2, 4, 4), dtype=complex)
    M0[0,0, :,:] = np.eye(4) # Initial state to itself
    M0[1,1, :,:] = np.eye(4) # Final state to itself
    M0[0,1, :,:] = np.diag([0,1,1,1]) # Operator size

    if L == 1:
        return MPO([M0[np.array([0])[:,np.newaxis],np.array([1])[np.newaxis,:],:,:]])
    else:
        # The edge MPO
        ML = M0[[0],:,:,:]
        MR = M0[:,[1],:,:]
        return MPO([ML] + [M0]*(L-2) + [MR])

# Realized in a similar way as OpSumMPS
def LindbladMPO (L, H, Lis, dagger = False):

    thisax = 2 # Current axes
    dims = [] # dims[i]:dims[i+1] are the axes for operator i
    mats = []

    single_site = np.zeros((4,4), dtype=complex) # Record the amplitude of single-site operators

    def add_mats (mat):
        nonlocal single_site, thisax, dims, mats
        span = len(mat)
        if span > 1:
            dims.append(thisax)
            mats.append(mat)
            thisax += (span-1)
        elif span == 1:
            single_site += mat[0]

    # Hamiltonian
    for term in H:
        if dagger:
            add_mats(term.getMats(type="L", add_coef=1j))
            add_mats(term.getMats(type="R", add_coef=-1j))
        else:
            add_mats(term.getMats(type="L", add_coef=-1j))
            add_mats(term.getMats(type="R", add_coef=1j))
        
    def mul_mats (mat1, mat2):
        return [mat1[i]@mat2[i] for i in range(len(mat1))]

    # Jump operators
    for Li in Lis:
        for tL0 in Li:
            for tR in Li:
                tL = tL0.conj()
                if dagger:
                    add_mats(mul_mats(tL.getMats(type="L"), tR.getMats(type="R")))
                else:
                    add_mats(mul_mats(tL.getMats(type="R"), tR.getMats(type="L")))
                add_mats(mul_mats(tL.getMats(type="L", add_coef=-1/2), tR.getMats(type="L")))
                add_mats(mul_mats(tR.getMats(type="R"), tL.getMats(type="R", add_coef=-1/2)))

    # The bulk MPO tensor
    M0 = np.zeros((thisax, thisax, 4, 4), dtype=complex)
    M0[0,0, :,:] = np.eye(4) # Initial state to itself
    M0[1,1, :,:] = np.eye(4) # Final state to itself
    M0[0,1, :,:] = single_site # Initial state can directly hop to final state, yielding a single-site operator
    for i, dim in enumerate(dims):
        mat = mats[i]
        span = len(mat)
        M0[0,dim, :,:] = mat[0] # Initial state to transition axes
        M0[dim+span-2,1, :,:] = mat[-1] # Transition axes to final state
        for j in range(1, span-1):
            M0[dim+j-1,dim+j, :,:] = mat[j] # Within transition axes   
    
    if L == 1:
        return MPO([M0[np.array([0])[:,np.newaxis],np.array([1])[np.newaxis,:],:,:]])
    else:
        # The edge MPO
        ML = M0[[0],:,:,:]
        MR = M0[:,[1],:,:]
        return MPO([ML] + [M0]*(L-2) + [MR])

def printMPO (W):
    L = len(W)
    op = W.contract()
    nonzeros = np.vstack(np.nonzero(op))
    for j in range(np.shape(nonzeros)[1]):
        ind = nonzeros[:,j].flatten()
        val = op[*ind]
        print("".join([IND_MAP[i] for i in ind[L:]])+"->"+"".join([IND_MAP[i] for i in ind[:L]])+f" = {val}")

def MPO_to_Matrix (W):
    op = W.contract()
    op = np.reshape(op, (np.prod(np.shape(op)[:len(W)]), np.prod(np.shape(op)[len(W):])))
    return op