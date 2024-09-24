import numpy as np


class GTG_Test:
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

    def perform_GTG_test(self, group_acc):
        # ToDo
        scores = np.zeros(self.n_clients)
        return scores

    def get_group_accuracy(self, client_models, server, group4test):
        group_acc = np.zeros(self.n_tests)
        class_precision = np.zeros((self.n_tests, self.n_classes))
        class_recall = np.zeros((self.n_tests, self.n_classes))
        class_f1 = np.zeros((self.n_tests, self.n_classes))

        model_list = [[[] for j in range(self.n_classes)] for i in range(self.n_tests)]
        for i in range(self.n_tests):
            client_idxs = group4test[0:i]
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
