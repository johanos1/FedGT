import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

lib = ctypes.cdll.LoadLibrary("./src/C_code/BCJR_4_python.so")
fun = lib.BCJR
fun.restype = None
p_ui8_c = ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")
p_d_c = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
fun.argtypes = [
    p_ui8_c,
    p_d_c,
    p_ui8_c,
    p_d_c,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    p_d_c,
    p_ui8_c,
]
r = 3
N = 6

# Inputs
H = np.array(
    [[1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1]], dtype=np.uint8
)
LLRi = 1.19 * np.ones((1, N), dtype=np.double)
test_vec = np.array([[0, 0, 1]], dtype=np.uint8)
ChannelMatrix = np.array([[0.85, 0.15], [0.1, 0.9]], dtype=np.double)
threshold_dec = 0.598
LLRO = np.empty((1, N), dtype=np.double)
DEC = np.empty((1, N), dtype=np.uint8)
# Call of the function
fun(H, LLRi, test_vec, ChannelMatrix, threshold_dec, N, r, LLRO, DEC)
# print(LLRO)
# print(DEC)
