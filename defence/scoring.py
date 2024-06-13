import numpy as np


class QI_Scoring:
    def __init__(
            self,
            parity_check_matrix,
            threshold,
            value,
    ):
        self.threshold = threshold
        self.value = value
        self.parity_check_matrix = parity_check_matrix
        self.n_clients = self.parity_check_matrix.shape[1]
        self.n_tests = self.parity_check_matrix.shape[0]
        self.scores = np.zeros(self.n_clients)

    def update_QI_scores(self, group_acc):
        
        for i in range(group_acc.shape[0] - 1): # For each group
            for j in range(group_acc.shape[1]): # For each client

                for k in range(group_acc.shape[1]): # For each client
                    if group_acc[i][j] < group_acc[i+1][k] - self.threshold:
                        # Bad
                        for u in range(self.n_clients):
                            if self.parity_check_matrix[j][u]:
                                if self.value == 'count':
                                    self.scores[u] -= 1
                                elif self.value == 'actual':
                                    self.scores[u] = self.scores[u] + (group_acc[i][j] - group_acc[i+1][k])
                        # Good
                        for u in range(self.n_clients):
                            if self.parity_check_matrix[k][u]:
                                if self.value == 'count':
                                    self.scores[u] += 1
                                elif self.value == 'actual':
                                    self.scores[u] = self.scores[u] - (group_acc[i][j] - group_acc[i+1][k])
                # Ugly
                if group_acc[i][j] < 0 - self.threshold:
                    for u in range(self.n_clients):
                        if self.parity_check_matrix[j][u]:
                            if self.value == 'count':
                                self.scores[u] -= 1
                            elif self.value == 'actual':
                                self.scores[u] = self.scores[u] + group_acc[i][j]
                if i == group_acc.shape[0] - 2:
                    if group_acc[i + 1][j] < 0 - self.threshold:
                        for u in range(self.n_clients):
                            if self.parity_check_matrix[j][u]:
                                if self.value == 'count':
                                    self.scores[u] -= 1
                                elif self.value == 'actual':
                                    self.scores[u] = self.scores[u] + group_acc[i+1][j]
        return self.scores