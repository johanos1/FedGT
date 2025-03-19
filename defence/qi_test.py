import numpy as np


class QI_Test():
    def __init__(self, n_clients, n_tests, n_classes, threshold, groups4test, past_groups4test=None):

        self.n_clients = n_clients
        self.n_tests = n_tests
        self.n_classes = n_classes
        self.threshold = threshold
        self.groups4test_curr = groups4test
        if past_groups4test is not None:
            self.groups4test_past = past_groups4test
        else:
            self.groups4test_curr = groups4test
        assert self.n_tests == self.groups4test_curr.shape[0], "Wrong no of rows in H!"
        assert self.n_clients == self.groups4test_curr.shape[1], "Wrong no of cols in H!"

    def scoring_across(self, norm, scores, badgroup, goodgroup=None):
        for u in range(self.n_clients):
            if self.groups4test_past[badgroup][u]:
                scores[u] -= 1 / norm[u]
            if goodgroup is not None and self.groups4test_curr[goodgroup][u]:
                scores[u] += 1 / norm[u]
        return scores

    def scoring_inround(self, norm, scores, badgroup, goodgroup=None):
        for u in range(self.n_clients):
            if self.groups4test_curr[badgroup][u]:
                scores[u] -= 1 / norm[u]
            if goodgroup is not None and self.groups4test_curr[goodgroup][u]:
                scores[u] += 1 / norm[u]
        return scores

    def perform_QI_test_inround(self, group_imp, r):
        normalize = np.sum(self.groups4test_curr, 0)
        scores = np.zeros(self.n_clients)
        for j in range(group_imp.shape[1]):
            if group_imp[r][j] < 0 - self.threshold:
                self.scoring_inround(normalize, scores, j)
            for k in range(group_imp.shape[1]):
                if group_imp[r][j] < group_imp[r][k] - self.threshold:
                    self.scoring_inround(normalize, scores, j, k)
        return scores

    def perform_QI_test_acrossround(self, group_imp, r):
        normalize = np.sum(self.groups4test_curr, 0) + np.sum(self.groups4test_past, 0)
        scores = np.zeros(self.n_clients)
        for j in range(group_imp.shape[1]):
            if group_imp[r][j] < 0 - self.threshold:
                self.scoring_across(normalize, scores, j)
            for k in range(group_imp.shape[1]):
                if group_imp[r-1][j] < group_imp[r][k] - self.threshold:
                    self.scoring_across(normalize, scores, j, k)
        return scores
