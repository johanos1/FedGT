import numpy as np

x = np.array([0.22, 0.49, 0.5, 0.51, 0.52, 0.52, 0.53, 0.57])
dx = np.diff(x)
for k in range(1, 5):
    tol = 1
    clusters = 0
    while clusters < k and tol > 0:
        clusters = np.sum(dx > tol == True)
        tol = -0.01
