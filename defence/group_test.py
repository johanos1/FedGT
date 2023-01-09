import numpy as np
import logging
import ctypes
# Calling C functions with numpy inputs
from numpy.ctypeslib import ndpointer



class Group_Test:
    def __init__(self, n_clients, prevalence, threshold_dec, min_acc, threshold_from_max_acc, P_MD_test, P_FA_test):
        self.n_clients = n_clients
        self.parity_check_matrix = self._get_test_matrix()
        self.n_tests = self.parity_check_matrix.shape[0]
        self.threshold_dec = threshold_dec
        
        # Set up the decoding algorithm based on C-code
        lib = ctypes.cdll.LoadLibrary("./src/C_code/BCJR_4_python.so")
        self.fun = lib.BCJR
        self.fun.restype = None
        p_ui8_c = ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")
        p_d_c = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
        self.fun.argtypes = [
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
        
        self.min_acc = min_acc
        self.threshold_from_max_acc = threshold_from_max_acc

        self.LLRO = np.empty((1, self.n_clients), dtype=np.double)
        self.DEC = np.empty((1, self.n_clients), dtype=np.uint8)
        
        if prevalence == 0:
            prevalence = 0.1 # this is based on a mismatched idea
            
        self.LLRi = np.log((1 - prevalence) / prevalence) * np.ones(
            (1, self.n_clients), dtype=np.double
        )
        
        self.ChannelMatrix = np.array(
            [[1-P_FA_test, P_FA_test], [P_MD_test, 1-P_MD_test]], dtype=np.double
        )
    
    def _get_test_matrix(self):
        # fmt: off
        if self.n_clients == 15:
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
        elif self.n_clients == 31:
            parity_check_matrix = np.array(
                [
                    [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                    [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
                    [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,],
                    [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,],
                    [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,],
                    [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0,],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,],
                ],
                dtype=np.uint8,
            )
        # fmt: on
        return parity_check_matrix 
        
    def perform_group_test(self, group_acc):
     
        max_acc = group_acc.max()
        if max_acc < self.min_acc:
            tests = np.ones((1, self.n_tests), dtype=np.uint8)
        else:
            tests = np.zeros((1, self.n_tests), dtype=np.uint8)
            tests[:, group_acc < self.threshold_from_max_acc  * max_acc] = 1
        self.fun(
            self.parity_check_matrix,
            self.LLRi,
            tests,
            self.ChannelMatrix,
            self.threshold_dec,
            self.n_clients,
            self.n_tests,
            self.LLRO,
            self.DEC,
        )
        return self.DEC

    def noiseless_group_test(self, syndrome):
             
        tests = np.zeros((1, self.n_tests), dtype=np.uint8)
        tests[0, syndrome[0,:] > 0] = 1
        self.fun(
            self.parity_check_matrix,
            self.LLRi,
            tests,
            self.ChannelMatrix,
            self.threshold_dec,
            self.n_clients,
            self.n_tests,
            self.LLRO,
            self.DEC,
        )
        return self.DEC
      
    def get_group_accuracies(self, client_models, server):
        group_acc = np.zeros(self.n_tests)
        for i in range(self.n_tests):
        # np.where gives a tuple where first entry is the list we want
            client_idxs = np.where(
                self.parity_check_matrix[i, :] == 1
            )[0].tolist()
            group = []
            for idx in client_idxs:
                group.append(client_models[idx])

            # aggregation returns a list so pick the (only) item
            model = server.aggregate_models(
                group, update_server=False
            )[0]
            # note, aside from accuracy, we have access to precision, recall, and f1 score for each class
            (
                group_acc[i],
                cf_matrix, 
                class_precision,
                class_recall,
                class_f1,
            ) = server.evaluate(
                test_data=False, eval_model=model
            )
        return group_acc, class_precision, class_recall, class_f1
        
    
    