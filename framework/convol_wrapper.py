import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import os

_test1 = ctypes.CDLL(os.getcwd() + '/lib_mine.so')

_test1.convol.argtypes = (ndpointer(dtype=np.float32,ndim=3,flags='C_CONTIGUOUS'), ndpointer(dtype=np.float32,ndim=2,flags='C_CONTIGUOUS'),ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS'), ndpointer(dtype=np.float32,ndim=3,flags='C_CONTIGUOUS'),ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS'))

_test1.convol.restype = None

def convol_C(out,weight,kernel_shape,x,x_shape,outplanes,h_out,w_out,stride,kernel_size,bias):
	global _test1
	#print(f"convol_wrapper = {x_shape}")

	_test1.convol(out,weight,kernel_shape,x,x_shape,ctypes.c_int(outplanes),ctypes.c_int(h_out),ctypes.c_int(w_out),ctypes.c_int(stride),ctypes.c_int(kernel_size),bias)

	return out


