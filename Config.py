
USE_GPU = False

if USE_GPU:
    import cupy as np
    from cupy import linalg as LA
    from nconCUPY import ncon
else:
    import numpy as np
    from numpy import linalg as LA
    from ncon import ncon