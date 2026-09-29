import numpy as np

def requireNumpyArray(np_array):
    return np.require(np_array, requirements=['F_CONTIGUOUS', 'ALIGNED'])
