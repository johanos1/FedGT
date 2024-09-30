import numpy as np
from itertools import combinations


class SHAP_Test:
    def __init__(
            self,
            n_clients,
            n_classes,
    ):
        self.n_clients = n_clients
        self.n_tests = 2**n_clients
        self.n_classes = n_classes

    def perform_SHAP_test(self, group_acc, r):
        groups4test = np.flip(np.array([[int(x) for x in format(i, f'0{self.n_clients}b')] for i in range(self.n_tests)]))
        scores = np.zeros(self.n_clients)

        for i in range(self.n_clients):
            tmp = 0
            for subset in groups4test:
                if subset[i] == 0:
                    continue
                # Create the subset without the participant 'i'
                subset_without_i = np.copy(subset)
                subset_without_i[i] = 0

                subset_index = np.where(np.all(groups4test == subset, axis=1) == True)
                subset_without_i_index = np.where(np.all(groups4test == subset_without_i, axis=1) == True)

                marginal_contribution = group_acc[r][subset_index[0][0]] - group_acc[r][subset_without_i_index[0][0]]
                subset_size = np.sum(subset) - 1
                weight = (np.math.factorial(subset_size) * np.math.factorial(self.n_clients - subset_size - 1)) / np.math.factorial(self.n_clients)
                tmp += weight * marginal_contribution

            scores[i] = tmp
        return scores

    def all_group_accuracies(self, client_models, server):
        group_acc = np.zeros(self.n_tests)
        class_precision = np.zeros((self.n_tests, self.n_classes))
        class_recall = np.zeros((self.n_tests, self.n_classes))
        class_f1 = np.zeros((self.n_tests, self.n_classes))
        groups4test = np.flip(np.array([[int(x) for x in format(i, f'0{self.n_clients}b')] for i in range(self.n_tests)]))

        model_list = [[[] for j in range(self.n_classes)] for i in range(self.n_tests)]
        for i in range(self.n_tests-1):
            # np.where gives a tuple where first entry is the list we want
            client_idxs = np.where(groups4test[i, :] == 1)[0].tolist()
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

        return group_acc, class_precision, class_recall, class_f1