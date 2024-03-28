import numpy as np


class QI_Test:
    def __init__(
            self,
            n_clients,
            n_tests,
            n_classes,
            threshold,
            value,
    ):
        self.n_clients = n_clients
        self.n_tests = n_tests
        self.n_classes = n_classes
        self.threshold = threshold
        self.value = value
        self.parity_check_matrix = self._get_test_matrix()
        assert self.n_tests == self.parity_check_matrix.shape[0], "Wrong no of rows in H!"
        assert self.n_clients == self.parity_check_matrix.shape[1], "Wrong no of cols in H!"

    def perform_QI_test(self, group_acc):

        scores = np.zeros((1, self.n_clients), dtype=np.uint8)

        # ToDo write QI logic

        return scores

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
