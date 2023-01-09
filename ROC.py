import numpy as np
import random
from dotmap import DotMap

import logging
import os
import time
from math import log

import ctypes

# Calling C functions with numpy inputs
from numpy.ctypeslib import ndpointer


def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":

    # get arguments
    total_MC_it = 100
    threshold_vec = np.arange(1, 2.1, 0.1).tolist()
    client_number = 15

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

    # Group testing parameters
    if client_number == 15:
        parity_check_matrix = np.array(
            [
                [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
            ],
            dtype=np.uint8,
        )
    number_tests = parity_check_matrix.shape[0]
    sidja = int(time.time())
    P_FA = np.zeros(len(threshold_vec))
    P_MD = np.zeros(len(threshold_vec))
    mali_number = 1
    LLRO = np.empty((1, client_number), dtype=np.double)
    prevalence = mali_number / client_number
    LLRi = log((1 - prevalence) / prevalence) * np.ones(
        (1, client_number), dtype=np.double
    )
    noiseless = True
    prop = 0.01
    if noiseless == True:
        prop = 1e-6
    ChannelMatrix = np.array([[1 - prop, prop], [prop, 1 - prop]], dtype=np.double)
    DEC = np.empty((1, client_number), dtype=np.uint8)
    for indeks_group, threshold_dec in enumerate(threshold_vec):
        set_random_seed(sidja)
        FA = 0
        MD = 0
        print("Starting with threshold_dec : {}".format(threshold_dec))
        for monte_carlo_iterr in range(total_MC_it):
            malicious_clients = np.random.permutation(client_number)
            malicious_clients = malicious_clients[:mali_number].tolist()
            defective = np.zeros((1, client_number), dtype=np.uint8)
            defective[:, malicious_clients] = 1

            syndrome = np.matmul(defective, parity_check_matrix.transpose())
            if noiseless == True:
                tests = np.array(syndrome > 0, dtype=np.uint8)
            fun(
                parity_check_matrix,
                LLRi,
                tests,
                ChannelMatrix,
                threshold_dec,
                client_number,
                number_tests,
                LLRO,
                DEC,
            )
            MD = MD + np.sum(DEC[defective == 1] == 0)
            FA = FA + np.sum(DEC[defective == 0] == 1)
        P_MD[indeks_group] = MD / (mali_number * total_MC_it)
        P_FA[indeks_group] = FA / ((client_number - mali_number) * total_MC_it)
    cwd = os.getcwd()
    name = "ROC_" + time.strftime("%Y%m%d_%H%M%S") + "_it_" + str(total_MC_it)
    directory = os.path.join(cwd, "csv_files", name)
    if not os.path.exists(directory):
        os.mkdir(directory)
    np.savetxt(directory + "/P_FA.csv", P_FA, delimiter=",")
    np.savetxt(directory + "/P_MD.csv", P_MD, delimiter=",")
    np.savetxt(directory + "/lambdas.csv", np.array(threshold_vec), delimiter=",")
    np.savetxt(directory + "/MCits.csv", np.array([total_MC_it]), delimiter=",")
    np.savetxt(directory + "/prevalence.csv", np.array([prevalence]), delimiter=",")
    np.savetxt(directory + "/ChannelMatrix.csv", ChannelMatrix, delimiter=",")
    np.savetxt(directory + "/seed.csv", np.array([sidja]), delimiter=",")
