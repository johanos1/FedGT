import numpy as np


class QI_Test():
    def __init__(self, n_clients, n_tests, n_classes, threshold, groups4test):

        self.n_clients = n_clients
        self.n_tests = n_tests
        self.n_classes = n_classes
        self.threshold = threshold
        self.groups4test = groups4test
        assert self.n_tests == self.groups4test.shape[0], "Wrong no of rows in H!"
        assert self.n_clients == self.groups4test.shape[1], "Wrong no of cols in H!"

    def scoring(self, norm, scores, badgroup, goodgroup=None):
        for u in range(self.n_clients):
            if self.groups4test[badgroup][u]:
                scores[u] -= 1 / norm[u]
            if goodgroup is not None:
                if self.groups4test[goodgroup][u]:
                    scores[u] += 1 / norm[u]
        return scores

    def perform_QI_test_inround(self, group_acc, r):
        normalize = np.sum(self.groups4test, 0)
        scores = np.zeros(self.n_clients)
        for j in range(group_acc.shape[1]):
            for k in range(j+1, group_acc.shape[1]):
                if group_acc[r][j] < group_acc[r][k] - self.threshold:
                    self.scoring(normalize, scores, j, k)
                if group_acc[r][j] > group_acc[r][k] + self.threshold:
                    self.scoring(normalize, scores, k, j)
        return scores

    def perform_QI_test_acrossround(self, group_imp, r):
        normalize = np.sum(self.groups4test, 0)
        scores = np.zeros(self.n_clients)
        for j in range(group_imp.shape[1]):
            for k in range(group_imp.shape[1]):
                if group_imp[r-1][j] < group_imp[r][k] - self.threshold:
                    self.scoring(normalize, scores, j, k)
            if group_imp[r][j] < 0 - self.threshold:  # not checking the first test-round
                self.scoring(normalize, scores, j)
        return scores
