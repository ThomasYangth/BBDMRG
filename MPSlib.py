from Config import *

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
                self.L = 1
                self.Mlist = [np.copy(M0)]
            else:
                if ML is None or MR is None:
                    raise Exception("Provide ML and MR for initialization!")
                self.L = L
                self.Mlist = [np.copy(ML)] + [np.copy(M0) for _ in range(L-2)] + [np.copy(MR)]
        else:
            self.Mlist = Mlist
            self.L = len(Mlist)

    def __len__ (self):
        return self.L

    def __getitem__ (self, key):
        return self.Mlist[key]

    def __setitem__ (self, key, value):
        self.Mlist[key] = value

    def __str__(self):
        s = ""
        for i in range(len(self)):
            s += f"Tensor {i+1}\n"
            s += str(self.Mlist[i]) + "\n"
        return s
    
    def __repr__(self):
        return self.__str__()

def OpSumMPS(L, ops):

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

    for type, val in ops:
        span = len(type)
        type = type.upper()
        if span > 1:
            dims.append(thisax)
            spans.append(span)
            types.append(type)
            vals.append(abs(val) ** (1/span))
            signs.append(np.sign(val))
            thisax += (span - 1)
        else:
            # Example using a dictionary for 'pauli' conversion
            pauli = {'X': 0, 'Y': 1, 'Z': 2, 'I': 3}  # Modify as necessary
            single_site[pauli[type]] += val

    # The bulk MPS tensor
    M0 = np.zeros((thisax, 4, thisax), dtype=complex)
    M0[0, 0, 0] = 1 # Initial state to itself
    M0[1, 0, 1] = 1 # Final state to itself
    for i in range(4):
        M0[0, i, 1] = single_site[i] # Initial state can directly hop to final state, yielding a single-site operator
    for i, dim in enumerate(dims):
        span = spans[i]
        val = vals[i]
        type_indices = [pauli[c] for c in types[i]]
        M0[0, type_indices[0], dim] = val * signs[i] # Initial state to transition axes
        M0[dim + span - 2, type_indices[-1], 1] = val # Transition axes to final state
        for j in range(1, span - 1):
            M0[dim + j - 1, type_indices[j], dim + j] = val # Within transition axes

    # The edge MPS
    ML = M0[0, :, :]
    MR = M0[:, :, 1]

    return MPS(M0=M0, ML=ML, MR=MR, L=L)

def normalize_MPS (M):

    if len(M) == 1:
        return MPS([M[0] / np.linalg.norm(M[0].flatten())])

    L = ncon([M[0], M[0].conj()],[[1,-1],[1,-2]])
    for i in range(1, len(M)-1):
        L = ncon([L, M[i], M[i].con()], [[1,2],[1,3,-1],[2,3,-2]])
    L = ncon([L,M[-1],M[-1].conj()],[[1,2],[1,3],[2,3]])
    norm = (L.flatten()[0])**(1/L)
    return MPS([Mi/norm for Mi in M])