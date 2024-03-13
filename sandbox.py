import numpy as np
# from computation_graph import *

T = np.array([[-0.07426286,  0.12243876],
              [-0.0481759 , -0.0481759 ],
              [ 0.12243876, -0.07426286]])

print(T.flatten('F'))

# D = np.sum(T, axis=1, keepdims=True) * np.ones_like(T)
# print(D)

# D = T - np.mean(T, axis=1, keepdims=True)
# print(D)

# y = np.array([[1],
#               [0],
#               [1]])
#
# M2 = np.zeros_like(T)
# row_idxs = np.arange(T.shape[0])
# y_flat = y.flatten()
# M2[row_idxs, y_flat] = T[row_idxs, y_flat]
# print(M2)