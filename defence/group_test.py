import numpy as np
import logging
import ctypes
import torch.nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Calling C functions with numpy inputs
from numpy.ctypeslib import ndpointer


class Group_Test:
    def __init__(
        self,
        n_clients,
        n_tests,
        prevalence,
        threshold_dec,
        min_acc,
        threshold_from_max_acc,
        P_MD_test,
        P_FA_test,
    ):
        self.n_clients = n_clients
        self.n_tests = n_tests
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
        if self.n_clients == self.n_tests:
            parity_check_matrix = np.identity(self.n_clients, dtype=np.uint8)
            return parity_check_matrix
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
    
    def _dunn_index(self, data, labels, centroids): 
        num_samples = len(data)
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        
        if num_clusters == 1:
            return 0  # Dunn index undefined for a single cluster
        
        dist_centr = np.inf*np.ones((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(i+1, num_clusters):
                dist_centr[i, j] = dist_centr[j, i] = np.linalg.norm(centroids[i] - centroids[j])
        enumerator = dist_centr.min()
        distances = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                distances[i, j] = distances[j, i] = np.linalg.norm(data[i] - data[j]) 

        a = np.zeros(num_samples)  # Average distance within clusters
        b = np.zeros(num_samples)  # Minimum average distance to other clusters
        denumerator = 0 
        for i in range(num_samples):
            cluster_label = labels[i]
            same_cluster_indices = np.where(labels == cluster_label)[0]
            num_same_cluster = len(same_cluster_indices)
            
            # Compute a[i]
            if num_same_cluster > 1:
                a[i] = np.max(distances[i, same_cluster_indices])
                if max(denumerator, a[i]) == a[i]:
                    denumerator = a[i]
        dull_index = enumerator/denumerator
        return dull_index

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
        return self.DEC, self.LLRO
    
    def perform_group_test(self, group_acc, group_PCA, ss_thres, DELTA):

        tests = np.zeros((1, self.n_tests), dtype=np.uint8)
        tests[0, :] = self.perform_clustering_and_testing(group_acc, group_PCA, ss_thres)
        prev_est = (self.n_tests - np.sum(tests[0, :] == 0)) * 0.05 
        if prev_est == 0:
            prev_est = 0.1
        LLRinput = np.log((1 - prev_est) / prev_est) * np.ones((1, self.n_clients), dtype=np.double)
        td = DELTA + np.log((1 - prev_est) / prev_est)
        self.fun(
            self.parity_check_matrix,
            LLRinput, #self.LLRi,
            tests,
            self.ChannelMatrix,
            td, #self.threshold_dec,
            self.n_clients,
            self.n_tests,
            self.LLRO,
            self.DEC,
        )
        return self.DEC, self.LLRO, tests, LLRinput, td
    
    def perform_gt(self, test_values):

        tests = np.zeros((1, self.n_tests), dtype=np.uint8)
        tests[0, :] = test_values
        #look_up_nm = [0, 1, 2, 3, 3, 4, 5, 5, 5]
        look_up_nm = [5, 5, 5, 4, 3, 3, 2, 1, 0]
        nm_est = look_up_nm[np.sum(tests[0, :] == 0)]
        if nm_est == 0:
            nm_est = 0.1 * self.n_clients
        prev_est = nm_est / self.n_clients
        LLRinput = np.log((1 - prev_est) / prev_est) * np.ones((1, self.n_clients), dtype=np.double)
        td = 0.0
        self.fun(
            self.parity_check_matrix,
            LLRinput, #self.LLRi,
            tests,
            self.ChannelMatrix,
            0.0, #td, #self.threshold_dec,
            self.n_clients,
            self.n_tests,
            self.LLRO,
            self.DEC,
        )
        idx_sort = np.argsort(self.LLRO)
        if nm_est != 0.1 * self.n_clients:
            self.DEC[0, idx_sort[:, :nm_est]] = 1
            identical_soft = np.where(self.LLRO[0,:] == self.LLRO[0, idx_sort][0, nm_est-1])[0]
            if len(identical_soft) > 1:
                self.DEC[0, identical_soft] = 1
            td = self.LLRO[0, idx_sort][0, nm_est] 
        return self.DEC, self.LLRO, LLRinput, td
    
    def perform_clustering_and_testing(self, group_acc, group_PCA, ss_thres):

        assert group_acc.size == group_PCA.size, "Wrong size of the group acc and group PCA!"
        assert group_acc.size == self.n_tests, "Wrong number of groups!"
        X = np.column_stack((group_acc, group_PCA))
        poss_clusters = np.arange(1, np.sum(self.parity_check_matrix, axis = 1).max() + 2,dtype = int).tolist()
        l_cls = len(poss_clusters)
        s_scores = -1 * np.ones(l_cls)
        d_scores = -1 * np.ones(l_cls)
        all_labels = np.empty((l_cls, self.n_tests))
        for idx_clst, clst in enumerate(poss_clusters):
            kmeans = KMeans(n_clusters=clst, random_state=0)
            kmeans.fit(X)
            labels_data = kmeans.labels_
            all_labels[idx_clst, :] = labels_data
            centroids = kmeans.cluster_centers_
            if clst == 1:
                s_scores[idx_clst] = 0.0
            else:
                s_scores[idx_clst] = silhouette_score(X, labels_data)
            d_scores[idx_clst] = self._dunn_index(X, labels_data, centroids)
        if s_scores.max() < ss_thres:
            tests = np.zeros(self.n_tests, dtype=np.uint8)
            return tests, s_scores, d_scores
        best_cluster = np.argmax(d_scores)
        tot_clust = poss_clusters[best_cluster]
        indices_dict = {}
        for value in range(tot_clust):
            indices_dict[value] = np.where(all_labels[best_cluster, :] == value)[0]
        temp_array = -1 * np.ones((tot_clust, 3))
        for key, items in indices_dict.items():
            temp_array[key, 0] = len(items)
            temp_array[key, 1] = np.mean(group_acc[items]) if group_acc[items].size != 0 else -1 # Check this plz
            temp_array[key, 2] = np.mean(group_PCA[items]) if group_PCA[items].size != 0 else -1 # Check this as well
        temp_array = temp_array.transpose()
        tests = np.ones(self.n_tests, dtype=np.uint8)
        if np.sum(temp_array[1, :] == np.max(temp_array[1, :])) == 1:
            idx_max = np.argmax(temp_array[1, :])
            tests[indices_dict[idx_max]] = 0
            return tests, s_scores, d_scores
        else:
            sorted_PCA = np.argsort(temp_array[2,:])
            sorted_test_ready = temp_array[:, sorted_PCA]
            sorted_idxs = np.arange(tot_clust)[sorted_PCA]
            if sorted_test_ready[1, 0] > sorted_test_ready[1, -1]:
                tests[indices_dict[sorted_idxs[0]]] = 0
            elif sorted_test_ready[1, 0] < sorted_test_ready[1, -1]:
                tests[indices_dict[sorted_idxs[-1]]] = 0
            elif sorted_test_ready[1, 0] == sorted_test_ready[1,-1]:
                if sorted_test_ready[0, 0] > sorted_test_ready[0, -1]:
                    tests[indices_dict[sorted_idxs[0]]] = 0
                elif sorted_test_ready[0, 0] < sorted_test_ready[0, -1]:
                    tests[indices_dict[sorted_idxs[-1]]] = 0
                elif sorted_test_ready[0, 0] == sorted_test_ready[0, -1]:
                    tests = np.zeros(self.n_tests, dtype=np.uint8)
            return tests, s_scores, d_scores

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
        return self.DEC, self.LLRO

    def get_group_accuracies(self, client_models, server, num_classes=10):
        group_acc = np.zeros(self.n_tests)
        class_precision = np.zeros((self.n_tests, num_classes))
        class_recall = np.zeros((self.n_tests, num_classes))
        class_f1 = np.zeros((self.n_tests, num_classes))
        loss_per_label = np.zeros((self.n_tests, num_classes))
        for i in range(self.n_tests):
            # np.where gives a tuple where first entry is the list we want
            client_idxs = np.where(self.parity_check_matrix[i, :] == 1)[0].tolist()
            group = []
            for idx in client_idxs:
                group.append(client_models[idx])

            # aggregation returns a list so pick the (only) item
            model = server.aggregate_models(group, update_server=False)[0]
            # note, aside from accuracy, we have access to precision, recall, and f1 score for each class
            (
                group_acc[i],
                cf_matrix,
                class_precision[i, :],
                class_recall[i, :],
                class_f1[i, :],
                loss_per_label[i, :],
            ) = server.evaluate(test_data=False, eval_model=model)
            ### Check this for non-ISIC datasets ### 
            true_positives = np.diag(cf_matrix)
            total_actuals = np.sum(cf_matrix, axis=1)
            recalls = true_positives / total_actuals
            group_acc[i] = np.mean(recalls)
        return group_acc, class_precision, class_recall, class_f1, loss_per_label

    def get_pca_components(self, client_models, server, no_comp = 4, num_classes = 10):
        # DEBUG
        sim_obj = torch.nn.CosineSimilarity(dim=1)
        pca = PCA(n_components=no_comp, svd_solver='full')

        server.model.to("cpu")
        #server_model = torch.flatten(server.model.fc.weight)
        if server.model._get_name() == 'LogisticRegression':
            server_model = server.model.linear.weight
            server_model_flattened = torch.flatten(server.model.linear.weight)
            str_name = "linear.weight"
        elif server.model._get_name() == 'ImageNet':
            server_model = server.model.fc.weight
            server_model_flattened = torch.flatten(server.model.fc.weight)
            str_name = "fc.weight"
        elif server.model._get_name() == 'Baseline':
            server_model = server.model.base_model.classifier.weight
            server_model_flattened = torch.flatten(server.model.base_model.classifier.weight)
            str_name = "base_model.classifier.weight"
        
        tmp_matrix = []
        sim_flattened_list = []
        agg_models = []
        agg_models_per_label = [[] for _ in range(num_classes)]
        for i in range(self.n_tests):
            client_idxs = np.where(self.parity_check_matrix[i, :] == 1)[0].tolist()
            group = []
            for idx in client_idxs:
                group.append(client_models[idx])

            # aggregation returns a list so pick the (only) item
            group_model = server.aggregate_models(group, update_server=False)[0]
            
            #sim_cos.append(sim_obj(server_model, torch.flatten(group_model["fc.weight"])-server_model).detach().numpy())
            tmp_matrix.append(sim_obj(server_model, group_model[str_name]-server_model).detach().numpy())
            sim_flattened_list.append(torch.nn.functional.cosine_similarity(server_model_flattened, torch.flatten(group_model[str_name]-server_model), dim=0).detach().numpy())
                        
            # Convert list of tensors to list of numpy arrays
            #agg_models.append(torch.flatten(group_model["fc.weight"]).detach().numpy())
            agg_models.append(torch.flatten(group_model[str_name]).detach().numpy())
            for jjj in range(num_classes):
                agg_models_per_label[jjj].append(group_model[str_name][jjj, :].detach().numpy())
        
        matrix = np.array(agg_models)
        ### Normalization ### 
        #matrix = (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
        pca_components = pca.fit_transform(matrix)
        #pca_components = pca.components_.transpose()
        pca_components_variance = pca.explained_variance_
        pca_components_variance_ratio = pca.explained_variance_ratio_

        matrix_per_label = np.array(agg_models_per_label)
        pca_per_label = np.empty([num_classes, self.n_tests, no_comp])
        variance_per_label = np.empty([num_classes, no_comp])
        variance_ratio_per_label = np.empty([num_classes, no_comp])
        for i in range(num_classes):
            #pca.fit(matrix_per_label[i,:,:].transpose())
            #matrix_per_label[i,:,:] = (matrix_per_label[i,:,:] - np.mean(matrix_per_label[i,:,:], axis=0)) / np.std(matrix_per_label[i,:,:], axis=0)
            pca_per_label[i, :, :] = pca.fit_transform(matrix_per_label[i,:,:])#pca.components_.transpose()
            variance_per_label[i, :] = pca.explained_variance_
            variance_ratio_per_label[i, :] = pca.explained_variance_ratio_
        
        sim_cos = np.array(tmp_matrix)
        sim_flattened = np.array(sim_flattened_list)
        
        return sim_cos, sim_flattened, pca_components, pca_components_variance, pca_components_variance_ratio, pca_per_label, variance_per_label, variance_ratio_per_label
