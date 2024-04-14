
USE_GPU = True

if USE_GPU:
    import cupy as cp
    from nconCUPY import ncon as nconCUPY
    def ncon (M, I):
        return nconCUPY([cp.array(arr) for arr in M], I).get()
        
else:
    from ncon import ncon as ncon

import numpy as np
from numpy import linalg as LA

IND_MAP = {'I':0, 'X':1, 'Y':2, 'Z':3, 0:"I", 1:"X", 2:"Y", 3:"Z"}
PAULIS = {
    'I': np.eye(2),
    'X': np.array([[0,1],[1,0]]),
    'Y': np.array([[0,-1j],[1j,0]]),
    'Z': np.array([[1,0],[0,-1]]),
    "XL": np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
    "XR": np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1j], [0, 0, -1j, 0]]),
    "YL": np.array([[0, 0, 1, 0], [0, 0, 0, 1j], [1, 0, 0, 0], [0, -1j, 0, 0]]),
    "YR": np.array([[0, 0, 1, 0], [0, 0, 0, -1j], [1, 0, 0, 0], [0, 1j, 0, 0]]),
    "ZL": np.array([[0, 0, 0, 1], [0, 0, -1j, 0], [0, 1j, 0, 0], [1, 0, 0, 0]]),
    "ZR": np.array([[0, 0, 0, 1], [0, 0, 1j, 0], [0, -1j, 0, 0], [1, 0, 0, 0]]),
    "IL": np.eye(4),
    "IR": np.eye(4)
}