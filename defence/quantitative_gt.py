import numpy as np
import logging
import ctypes
# Calling C functions with numpy inputs
from numpy.ctypeslib import ndpointer
import torch.nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def is_monotonic_array(array):
    
    return np.all(np.diff(array) > 0) or np.all(np.diff(array) < 0)


class Quantitative_Group_Test:
    def __init__(
        self,
        n_clients,
        n_tests,
        prevalence,
        is_irregular = True,
        BP_iteration = 100
    ):
        self.n_clients = n_clients
        self.n_tests = n_tests
        self.irregular = is_irregular
        if self.irregular == True and self.n_clients == 15 and self.n_tests == 8:
            self.dc = 4
            self.dv = 4
        self.BP_iteration = BP_iteration
        (self.parity_check_matrix, self.VN, self.CN, self.vn_deg, self.cn_deg) = self._get_test_matrix()
        assert self.n_tests == self.parity_check_matrix.shape[0], "Wrong no of rows in H!"
        assert self.n_clients == self.parity_check_matrix.shape[1], "Wrong no of cols in H!"

        # Set up the decoding algorithm based on C-code
        lib = ctypes.cdll.LoadLibrary("./src/C_code/QGT/FedQGT_decoder.so")
        p_ui8_c = ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")
        p_i16_c = ndpointer(ctypes.c_int16, flags="C_CONTIGUOUS")
        p_d_c = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
        self.fun = lib.BP_decoder
        self.fun.restype = None
        self.fun.argtypes = [
            ctypes.c_uint16,
            ctypes.c_uint16,
            ctypes.c_uint8,
            ctypes.c_uint8,
            p_i16_c,
            p_i16_c,
            p_ui8_c,
            p_d_c,
            ctypes.c_uint16,
            p_ui8_c,
            p_ui8_c,
            p_ui8_c,
        ]

        self.DEC = np.empty((1, self.n_clients), dtype=np.uint8)

        if prevalence == 0:
            prevalence = 0.1  # this is based on a mismatched idea

        self.LLRi = prevalence * np.ones(self.n_clients, dtype=np.double)

    def _get_test_matrix(self):
        if self.irregular == True:
            filename = f"ldpc_h_irreg_dv_{self.dv}_dc_{self.dc}_N{self.n_clients}_Nk{self.n_tests}.txt"
        else:
            filename = f"ldpc_h_reg_dv_{self.dv}_dc_{self.dc}_N{self.n_clients}_Nk{self.n_tests}.txt"
        VN = np.loadtxt("./src/matrices/VN_" + filename, dtype="int16", delimiter=" ")
        assert VN.shape[0] == self.n_clients, "Wrong number of rows with VN"
        assert VN.shape[1] == self.dv, "Wrong number of cols with VN"
        CN = np.loadtxt("./src/matrices/CN_" + filename, dtype="int16", delimiter=" ")
        assert CN.shape[0] == self.n_tests, "Wrong number of rows with CN"
        assert CN.shape[1] == self.dc, "Wrong number of cols with CN"

        if self.irregular == True:
            vn_deg = np.loadtxt("./src/matrices/vn_deg_" + filename, dtype="uint8", delimiter=" ")
            assert vn_deg.shape[0] == self.n_clients, "Wrong number of rows with vn_deg"
            cn_deg = np.loadtxt("./src/matrices/cn_deg_" + filename, dtype="uint8", delimiter=" ")
            assert cn_deg.shape[0] == self.n_tests, "Wrong number of elements with cn_deg"
        else:
            vn_deg = self.dv * np.ones(self.n_clients, dtype=np.uint8)
            cn_deg = self.dc * np.ones(self.n_tests, dtype=np.uint8)
            assert vn_deg.shape[0] == self.n_clients, "Wrong number of rows with vn_deg"
            assert cn_deg.shape[0] == self.n_tests, "Wrong number of elements with cn_deg"

        parity_check_matrix = np.zeros((self.n_tests, self.n_clients), dtype=np.uint8)
        for i in range(self.n_tests):
            parity_check_matrix[i, CN[i, : cn_deg[i]]] = 1
        return parity_check_matrix, VN, CN, vn_deg, cn_deg
    
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
    
    def noiseless_group_test(self, syndrome):
        
        self.fun(self.n_clients, self.n_tests, self.dv, self.dc, self.VN, self.CN, syndrome, self.LLRi, self.BP_iteration, self.cn_deg, self.vn_deg, self.DEC)
        return self.DEC
    
    def perform_gt(self, test_values):

        tests = np.zeros((1, self.n_tests), dtype=np.uint8)
        tests[0, :] = test_values
        self.fun(self.n_clients, self.n_tests, self.dv, self.dc, self.VN, self.CN, tests, self.LLRi, self.BP_iteration, self.cn_deg, self.vn_deg, self.DEC)
        return self.DEC

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
            kmeans = KMeans(n_clusters=clst, n_init=10, random_state=0)
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
            temp_array[key, 1] = np.mean(group_acc[items]) if group_acc[items].size != 0 else -1 # Mean accuracy
            temp_array[key, 2] = np.mean(group_PCA[items]) if group_PCA[items].size != 0 else -1 # Mean of first component
        temp_array = temp_array.transpose()
        tests = np.ones(self.n_tests, dtype=np.uint8)
        sorted_PCA = np.argsort(temp_array[2,:])
        sorted_test_ready = temp_array[:, sorted_PCA]
        sorted_idxs = np.arange(tot_clust)[sorted_PCA]
        if is_monotonic_array(sorted_test_ready[1, :]):
            ind_acc_sort = np.argsort(sorted_test_ready[1,:])[::-1] # either 0,1,...,tot_clust-1 or reversed
        else: #if not monotonic, just check the polar opposites 
            if sorted_test_ready[1, 0] > sorted_test_ready[1, -1]:
                ind_acc_sort = np.arange(tot_clust)
            elif sorted_test_ready[1, 0] < sorted_test_ready[1, -1]:
                ind_acc_sort = np.arange(tot_clust - 1, -1, -1)
            elif sorted_test_ready[1, 0] == sorted_test_ready[1,-1]:
                if sorted_test_ready[0, 0] > sorted_test_ready[0, -1]:
                    ind_acc_sort = np.arange(tot_clust)
                elif sorted_test_ready[0, 0] < sorted_test_ready[0, -1]:
                    ind_acc_sort = np.arange(tot_clust - 1, -1, -1)
                elif sorted_test_ready[0, 0] == sorted_test_ready[0, -1]:
                    tests = np.zeros(self.n_tests, dtype=np.uint8)
                    return tests, s_scores, d_scores
        for id_s, id in enumerate(sorted_idxs):
            tests[indices_dict[id]] = ind_acc_sort[id_s]
        return tests, s_scores, d_scores

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
            model = server.aggregate_models(group, update_server=False)
            
            # note, aside from accuracy, we have access to precision, recall, and f1 score for each class
            (
                group_acc[i],
                cf_matrix,
                class_precision[i, :],
                class_recall[i, :],
                class_f1[i, :],
                loss_per_label[i, :],
            ) = server.evaluate(test_data=False, eval_model=model)
            true_positives = np.diag(cf_matrix)
            total_actuals = np.sum(cf_matrix, axis=1)
            recalls = true_positives / total_actuals
            group_acc[i] = np.mean(recalls)
        return group_acc, class_precision, class_recall, class_f1, loss_per_label

    def get_pca_components(self, client_models, server, no_comp = 4, num_classes = 10):
        # 
        sim_obj = torch.nn.CosineSimilarity(dim=1)
        pca = PCA(n_components=no_comp, svd_solver='full')

        server.model.to("cpu")
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
            group_model = server.aggregate_models(group, update_server=False)
            tmp_matrix.append(sim_obj(server_model, group_model[str_name]-server_model).detach().numpy())
            sim_flattened_list.append(torch.nn.functional.cosine_similarity(server_model_flattened, torch.flatten(group_model[str_name]-server_model), dim=0).detach().numpy())

            # Convert list of tensors to list of numpy arrays
            agg_models.append(torch.flatten(group_model[str_name]).detach().numpy())
            for jjj in range(num_classes):
                agg_models_per_label[jjj].append(group_model[str_name][jjj, :].detach().numpy())
        
        matrix = np.array(agg_models)
        pca_components = pca.fit_transform(matrix)
        pca_components_variance = pca.explained_variance_
        pca_components_variance_ratio = pca.explained_variance_ratio_

        matrix_per_label = np.array(agg_models_per_label)
        pca_per_label = np.empty([num_classes, self.n_tests, no_comp])
        variance_per_label = np.empty([num_classes, no_comp])
        variance_ratio_per_label = np.empty([num_classes, no_comp])
        for i in range(num_classes):
            pca_per_label[i, :, :] = pca.fit_transform(matrix_per_label[i,:,:])
            variance_per_label[i, :] = pca.explained_variance_
            variance_ratio_per_label[i, :] = pca.explained_variance_ratio_
        
        sim_cos = np.array(tmp_matrix)
        sim_flattened = np.array(sim_flattened_list)
        
        return sim_cos, sim_flattened, pca_components, pca_components_variance, pca_components_variance_ratio, pca_per_label, variance_per_label, variance_ratio_per_label
