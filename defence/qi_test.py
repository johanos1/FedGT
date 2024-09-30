import numpy as np


class QI_Test():
    def __init__(self, n_clients, n_tests, n_classes, threshold, value, groups4test):

        self.n_clients = n_clients
        self.n_tests = n_tests
        self.n_classes = n_classes
        self.threshold = threshold
        self.value = value
        self.groups4test = groups4test
        assert self.n_tests == self.groups4test.shape[0], "Wrong no of rows in H!"
        assert self.n_clients == self.groups4test.shape[1], "Wrong no of cols in H!"

    def scoring(self, accuracies, norm, scores, g1, g2):
        for u in range(self.n_clients):
            if self.groups4test[g1][u]:
                if self.value == 'count':
                    scores[u] -= 1 / norm[u]
                elif self.value == 'actual':
                    scores[u] += (accuracies[g1] - accuracies[g2]) / norm[u]
            if self.groups4test[g2][u]:
                if self.value == 'count':
                    scores[u] += 1 / norm[u]
                elif self.value == 'actual':
                    scores[u] -= (accuracies[g1] - accuracies[g2]) / norm[u]
        return scores

    def perform_QI_test(self, group_acc, r):
        normalize = np.sum(self.groups4test, 0)
        scores = np.zeros(self.n_clients)
        for j in range(group_acc.shape[1]):
            for k in range(j+1, group_acc.shape[1]):
                if group_acc[r][j] < group_acc[r][k] - self.threshold:
                    self.scoring(group_acc[r], normalize, scores, j, k)
                if group_acc[r][j] > group_acc[r][k] + self.threshold:
                    self.scoring(group_acc[r], normalize, scores, k, j)
        return scores
