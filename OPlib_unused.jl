# Initializations for multiplication functions
function single_times (id1, id2)
    if id1 == 1
        return 1, id2 # Id times anything gives itself
    elseif id2 == 1
        return 1, id1 # Id times anything gives itself
    elseif id1 == id2
        return 1, 0 # Each of Sx, Sy, Sz squares to Id
    else
        return ((id2-id1+3)%3==1 ? 1im:-1im), 9-id1-id2
    end
end

# Multiply two terms
function term_times (inds1, inds2)
    coeff = 1
    if isinstance(inds1, Term):
        coeff *= inds1.coef
        inds1 = inds1.inds
    if isinstance(inds2, Term):
        coeff *= inds2.coef
        inds2 = inds2.inds
    l1 = len(inds1)
    l2 = len(inds2)
    if l1 != l2:
        raise Exception("Only terms with the same length can be multiplied!")
    ninds = np.zeros((l1,), dtype=int)
    for i in range(l1):
        c, r = single_times(inds1[i], inds2[i])
        coeff *= c
        ninds[i] = r
    return Term(coeff, ninds)

# The term class
class Term:

    def __init__ (self, coef, inds):
        self.coef = coef
        if isinstance(inds, str):
            self.inds = [IND_MAP[c] for c in inds]
        else:
            self.inds = inds

    def copy (self):
        return Term(self.coef, self.inds[:])

    def __imul__ (self, obj):
        print("imul called")
        if isinstance(obj, Term):
            t = self.__mul__(obj)
            self.coef = t.coef
            self.inds = t.inds
        else:
            self.coef *= obj
    
    def __mul__ (self, obj):
        if isinstance(obj, Term):
            return term_times(self, obj)
        else:
            return Term(self.coef*obj, self.inds)
    
    def __rmul__ (self, ratio):
        return self.__mul__(ratio)
    
    def conj (self):
        t1 = self.copy()
        t1.coef = np.conj(t1.coef)
        return t1

    def __repr__ (self):
        if self.coef == 0:
            return "0"
        elif len(self.inds) == 0:
            return "({:.2f})I".format(self.coef)
        else:
            return "({:.2f}){}".format(self.coef, "".join([IND_MAP[id] for id in self.inds]))

    def __str__ (self):
        return self.__repr__()
    
    def __iter__ (self):
        yield self.inds
        yield self.coef

    def __len__ (self):
        if self.coef == 0:
            return 0
        else:
            return len(self.inds)
    
    def getMats (self, type = "", add_coef = 1):
        if type not in ["","L","R"]:
            raise Exception("type must be either ''(Direct), 'L'eft, or 'R'ight!")
        if len(self) == 0:
            return []
        coef = self.coef * add_coef
        sgn = coef / np.abs(coef)
        val = np.abs(coef) ** (1/len(self))
        lst = [val*PAULIS[IND_MAP[ind]+type] for ind in self.inds]
        lst[0] *= sgn
        return lst
    
# Functions for multiplying operators on a lattice

# Expand a shorter index list to a longer one by adding zeros to it
def expand (arr, flen, pos = 0):
    l = len(arr)
    if pos < 0:
        return np.concatenate((np.zeros((flen-l,), dtype=int), arr))
    elif pos == 0:
        return np.concatenate((arr, np.zeros((flen-l,), dtype=int)))
    else:
        return np.concatenate((np.zeros((pos,), dtype=int), arr, np.zeros((flen-l-pos,), dtype=int)))

# Multiply two terms, taking results of a given length
def lat_times (inds1, inds2, flen):
    reslst = Operator()
    l1 = len(inds1)
    l2 = len(inds2)
    if flen > max(l1,l2):
        reslst.append(term_times(expand(inds1, flen), expand(inds2, flen, -1)))
        reslst.append(term_times(expand(inds1, flen, -1), expand(inds2, flen)))
    elif flen == l1:
        for i in range(flen-l2+1):
            reslst.append(term_times(inds1, expand(inds2, flen, i)))
    elif flen == l2:
        for i in range(flen-l1+1):
            reslst.append(term_times(expand(inds1, flen, i), inds2))
    else:
        raise Exception("flen must be no less than max(len(inds1),len(inds2))!")
    return reslst

# Multiply two terms and take all the results
def lat_times_all (inds1, inds2):
    reslst = Operator()
    l1 = len(inds1)
    l2 = len(inds2)
    for i in range(max(l1,l2), l1+l2):
        reslst.join(lat_times(inds1,inds2,i))
    return reslst

# The Operator Class
class Operator:

    def __init__ (self, terms = []):
        self.terms = []
        if isinstance(terms, dict):
            terms = [(value,key) for (key,value) in terms.items()]
        for term in terms:
            if isinstance(term, Term):
                self.terms.append(term)
            else:
                self.terms.append(Term(*term))

    def copy (self):
        return Operator([term.copy() for term in self.terms])

    def append (self, term):
        if term.coef == 0:
            return
        term.inds = np.trim_zeros(term.inds)
        for t in self.terms:
            if np.array_equal(term.inds, t.inds):
                t.coef += term.coef
                if t.coef == 0:
                    self.terms.remove(t)
                return
        self.terms.append(term)

    def join (self, tlist):
        for term in tlist.terms:
            self.append(term)

    def trace (self):
        tr = 0
        for term in self.terms:
            if len([i for i in term.inds if i != 0]) == 0:
                tr += term.coef
        return tr

    def dag (self):
        op = self.copy()
        for term in op.terms:
            term.coef = np.conj(term.coef)
        return op

    def inner (self, op):
        return np.real((self.dag() * op).trace())
        
    def norm (self):
        return np.sqrt(self.inner(self))

    def truncate (self, crit_size):
        op = Operator()
        for term in self.terms:
            if len([i for i in term.inds if i != 0]) <= crit_size:
                op.append(term)
        return op

    def __iadd__ (self, obj):

        if isinstance(obj, Term):
            return self.append(obj)
        elif isinstance(obj, Operator):
            return self.join(obj)
        else:
            raise Exception(f"Operator can only be added with Term or Operator, not {type(obj)}")

    def __add__ (self, obj):

        tl = self.copy()
        tl.__iadd__(obj)
        return tl

    def __radd__ (self, obj):
        return self.__add__(obj)
        
    def __rplus__ (self, obj):
        return self.__plus__(obj)

    def __sub__ (self, obj):
        return self.__add__(obj*(-1))

    def __rsub__ (self, obj):
        return (self*(-1)).__add__(obj)
    
    def __mul__ (self, obj):
        
        if isinstance(obj, (Number, np.number)):
            op1 = self.copy()
            for term in op1.terms:
                term.coef *= obj
            return op1
        else:
            opp = obj
            if isinstance(obj, Term):
                opp = Operator([obj])
            else:
                if not isinstance(obj, Operator):
                    raise Exception(f"An Operator can only be multiplied with a number, Term or Operator, not {type(obj)}")
            resop = Operator()
            for t1 in self.terms:
                for t2 in opp.terms:
                    resop.join(lat_times_all(t1.inds, t2.inds)*(t1.coef*t2.coef))
            return resop
            
    def __rmul__ (self, obj):
        if isinstance(obj, (Number, np.number)):
            op1 = self.copy()
            for term in op1.terms:
                term.coef *= obj
            return op1
        else:
            opp = obj
            if isinstance(obj, Term):
                opp = Operator([obj])
            else:
                if not isinstance(obj, Operator):
                    raise Exception("An Operator can only be multiplied with a number, Term or Operator!")
            resop = Operator()
            for t1 in self.terms:
                for t2 in opp.terms:
                    resop.join(lat_times_all(t2.inds, t1.inds)*(t1.coef*t2.coef))
            return resop
        
    def __imul__ (self, obj):
        op = self.__mul__(obj)
        self.terms = op.terms

    # Commutator
    def __matmul__ (self, obj):
        return self.__mul__(obj) - self.__rmul__(obj)
    
    def __iter__ (self):
        for term in self.terms:
            yield term

    def maxlen (self):
        maxlen = 0
        for t in self.terms:
            if len(t.inds) > maxlen:
                maxlen = len(t.inds)
        return maxlen

    def __repr__ (self):
        if len(self.terms) == 0:
            return "0"
        rep = ""
        for term in self.terms:
            rep += term.__repr__() + " + "
        return rep[:-3]

    def __str__ (self):
        return self.__repr__()