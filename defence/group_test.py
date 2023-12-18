import numpy as np
import logging
import ctypes

# Calling C functions with numpy inputs
from numpy.ctypeslib import ndpointer


class Group_Test:
    def __init__(
        self,
        n_clients,
        n_tests,
        n_classes,
        prevalence,
        threshold_dec,
        min_acc,
        threshold_from_max_acc,
        P_MD_test,
        P_FA_test,
    ):
        self.n_clients = n_clients
        self.n_tests = n_tests
        self.n_classes = n_classes
        self.n_pca_components = 4
        self.parity_check_matrix = self._get_test_matrix()
        assert self.n_tests == self.parity_check_matrix.shape[0], "Wrong no of rows in H!"
        assert self.n_clients == self.parity_check_matrix.shape[1], "Wrong no of cols in H!"
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
            prevalence = 0.1  # this is based on a mismatched idea

        self.LLRi = np.log((1 - prevalence) / prevalence) * np.ones((1, self.n_clients), dtype=np.double)

        self.ChannelMatrix = np.array([[1 - P_FA_test, P_FA_test], [P_MD_test, 1 - P_MD_test]], dtype=np.double)

    def _get_test_matrix(self):
        # fmt: off
        if self.n_clients == 15:
            if self.n_tests == 8:
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
            if self.n_tests == 4:
                parity_check_matrix = np.array(
                    [
                        [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                        [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                    ],
                    dtype = np.uint8,
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
        # if max_acc < self.min_acc:
        #     tests = np.ones((1, self.n_tests), dtype=np.uint8)
        # else:
        tests = np.zeros((1, self.n_tests), dtype=np.uint8)
        tests[:, group_acc < self.threshold_from_max_acc * max_acc] = 1
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
        tests[0, syndrome[0, :] > 0] = 1
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
        class_precision = np.zeros((self.n_tests, self.n_classes))
        class_recall = np.zeros((self.n_tests, self.n_classes))
        class_f1 = np.zeros((self.n_tests, self.n_classes))
        
        model_list = [[[] for j in range(self.n_classes)] for i in range(self.n_tests)]
        for i in range(self.n_tests):
            # np.where gives a tuple where first entry is the list we want
            client_idxs = np.where(self.parity_check_matrix[i, :] == 1)[0].tolist()
            group = []
            for idx in client_idxs:
                group.append(client_models[idx])

            # aggregation returns a list so pick the (only) item
            model = server.aggregate_models(group, update_server=False)
            # note, aside from accuracy, we have access to precision, recall, and f1 score for each class
            (
                group_acc[i],
                cf_matrix,
                class_precision[i, :],
                class_recall[i, :],
                class_f1[i, :],
            ) = server.evaluate(test_data=False, eval_model=model)
            
            
            for name, param in model.items():
                if server.model_name == "efficientnet":
                    if name == "base_model.classifier.weight" or name == "base_model.classifier.bias":
                        weights = param.data.cpu().numpy()
                        for j in range(self.n_classes):
                            if weights.ndim == 1:
                                weights=np.expand_dims(weights, axis=1)
                            model_list[i][j].extend(weights[j,:]) # for each label, we use both bias and weights
                elif server.model_name == "logistic_regression":
                    weights = param.data.cpu().numpy()
                    for j in range(self.n_classes):
                        if weights.ndim == 1:
                                weights=np.expand_dims(weights, axis=1)
                        model_list[i][j].extend(weights[j,:]) # for each label, we use both bias and weights
        
        model_list_2d = [np.array(sublist) for sublist in model_list] # turn lists into 2D arrays
        model_list_3d = np.stack(model_list_2d, axis=0) # stack along the first axis to create 3D matrix with dim: tests x labels x weights
        pca_features = self.get_pca(model_list_3d)
        
        group_acc_2d = np.expand_dims(np.array(group_acc), axis=1)
        group_recall_2d = np.array(class_recall)
        group_precision_2d = np.array(class_precision)
        
        gt_features = np.concatenate((pca_features, group_acc_2d, group_recall_2d, group_precision_2d), axis=1)
        
        self.cluster_test(gt_features)
                
        return group_acc, class_precision, class_recall, class_f1

    def get_pca(self, model_list_3d, verbose = True):
        # make a pca decomposition of the weights
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        #normalize client outputs to zero mean and std 1
        std_scaler = StandardScaler()
        
        pca_features = np.zeros((self.n_tests,self.n_classes,self.n_pca_components))
        explained_variance = np.zeros((self.n_pca_components, self.n_classes))
        for i in range(self.n_classes):
            group_models = model_list_3d[:,i,:]
            scaled = std_scaler.fit_transform(group_models)
            pca = PCA(n_components=self.n_pca_components)
            pca_components = pca.fit_transform(scaled)
            pca_features[:,i,:] = pca_components
            explained_variance[:,i] = pca.explained_variance_ratio_
        
        if verbose:
            np.set_printoptions(precision=1)
            
            pca_features_1d = pca_features[:,:,0]
            
            print(f"first PCA component (group x label): \n {pca_features_1d}")
            print(f"Explained variance (pca component x label): \n {explained_variance}")
        
        pca_features.reshape(pca_features.shape[0], pca_features.shape[1]*pca_features.shape[2])
        return pca_features
    
    def cluster_test(self, X):
        from sklearn.cluster import DBSCAN
        
        db = DBSCAN(eps=0.3, min_samples=10).fit(X)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)