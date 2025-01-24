import numpy as np
from scipy import sparse

# # 定义一个矩阵
# A = np.array([[4, 1], [1, 2]])
#
# # 求矩阵的逆
# A_inv = np.linalg.inv(A)
#
# print("矩阵A：\n", A)
# print("矩阵A的逆：\n", A_inv)
#
# # 验证逆矩阵
# print("A * A_inv：\n", np.dot(A, A_inv))


n = 5
m = n*2
# A = sparse.csc_matrix((m, m))
# print(A.shape)
A = np.zeros((m, m))
# print(A)
e_size_B = 2
for i in range(0, n-1):
    A[i:i+2, i:i+2] += sparse.csr_matrix([[1, -1], [-1, 1]])
    # print(f"第{i+1}次迭代---------------------------------------------------")
    # print(A)
print('A -----------------------------------------------')
print(A)
A[n:, n:] = A[0:n, 0:n]
print('A -----------------------------------------------')
print(A)

B = np.zeros((m, m))
for i in range(0, n-2):
    B[i:i+3, i:i+3] += [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    # print(f"第{i + 1}次迭代---------------------------------------------------")
    # print(B)
print('B -----------------------------------------------')
print(B)
B[n:, n:] = B[0:n, 0:n]
print('B -----------------------------------------------')
print(B)

C = np.zeros((m, m))
for i in range(0, n):
    C[i:i+1, i:i+1] += [1]
print('C -----------------------------------------------')
print(C)
C[n:, n:] = C[0:n, 0:n]
print('C -----------------------------------------------')
print(C)

# print('P -----------------------------------------------')
# print(1.0*A + 0.2*B +0.2*C)

# sparse_P = sparse.csc_matrix(A+B+C)
# print(sparse_P)


# H = np.zeros((4, 6))
# for i in range(0, 4):
#     H[i, i:i+3] = [4, -8, 4]
# print('H -----------------------------------------------')
# print(H)
H = np.zeros((m-2, m))
for i in range(0, m-2):
    H[i, i:i+3] = [4, -8, 4]
print('H -----------------------------------------------')
print(H)
AA = np.eye(m, m)
AA[1:m-1, :] = H
print('AA -----------------------------------------------')
print(AA)
