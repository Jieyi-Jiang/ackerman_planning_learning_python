import numpy as np
from scipy import sparse
import osqp
import matplotlib.pyplot as plt

def rectangular_wave(period, magnitude, n):
    pts = np.zeros((2, period*n))
    pts[0, :] = np.linspace(0, period*n-1, period*n)
    for i in range(0, n):
        pts[1 , i*period : i*period + round(period/2) ] = np.linspace(magnitude, magnitude, round(period/2))
    pts = pts.transpose()
    return pts

def triangular_wave(period, magnitude, n):
    pts = np.zeros((2, period*n))
    pts[0, :] = np.linspace(0, period*n-1, period*n)
    for i in range(0, n):
        pts[1 , i*period : (i+1)*period ] = np.linspace(0, magnitude, period)
    pts = pts.transpose()
    return pts


# def make_matrix_P(n=100, w_A=1.0, w_B=1.0, w_C=1.0):
#     m = n*2
#     A = sparse.csc_matrix((m, m))
#     B = sparse.csc_matrix((m, m))
#     C = sparse.eye(m, m)
#     for i in range(0, n - 1):
#         A[i:i+2, i:i+2] += sparse.csr_matrix([[1, -1], [-1, 1]])
#     A[n:, n:] = A[0:n, 0:n]
#     for i in range(0, n - 2):
#         B[i:i + 3, i:i + 3] += sparse.csr_matrix([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
#     B[n:, n:] = B[0:n, 0:n]
#     return w_A*A + w_B*B + w_C*C

def make_matrix_P(n=100, w_A=1.0, w_B=1.0, w_C=1.0):
    m = n*2
    A = np.zeros((m, m))
    B = np.zeros((m, m))
    C = np.eye(m, m)
    for i in range(0, n - 1):
        A[i:i+2, i:i+2] += [[1, -1], [-1, 1]]
    A[n:, n:] = A[0:n, 0:n]
    for i in range(0, n - 2):
        B[i:i + 3, i:i + 3] += [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    B[n:, n:] = B[0:n, 0:n]
    _P = w_A*A + w_B*B + w_C*C
    _P = _P * 2.0
    # point_xy = np.zeros((m, 1))
    # point_xy[0:n, 0] = pts[:, 0]
    # point_xy[n:, 0] = pts[:, 1]
    # _P = _P * point_xy
    return sparse.csc_matrix(_P)

def make_matrix_Q(pts, n=100):
    m = n*2
    _Q = np.zeros((m, 1))
    for i in range(0, m):
        _Q[i, 0] = -2
    point_xy = np.zeros((m, 1))
    point_xy[0:n, 0] = pts[:, 0]
    point_xy[n:, 0] = pts[:, 1]
    _Q = _Q * point_xy
    return _Q

def make_matrix_A(pts, n=100):
    m = n*2
    # A = np.zeros((m, m))
    _A = np.eye(m, m)
    return sparse.csc_matrix(_A)

def make_matrix_l(pts, n=100, R=np.inf):
    m = n*2
    _l = np.zeros((m, 1))
    point_xy = np.zeros((m, 1))
    point_xy[0:n, 0] = pts[:, 0]
    point_xy[n:, 0] = pts[:, 1]
    # for i in range(0, m):
    #     _l[i, 0] = -np.inf
    _l = point_xy - R
    # 等式约束起点和终点
    _l[0, 0]    = pts[0, 0]
    _l[n-1, 0]  = pts[n-1, 0]
    _l[n, 0]    = pts[0, 1]
    _l[m-1, 0]  = pts[n-1, 1]
    return _l

def make_matrix_u(pts, n=100, R=np.inf):
    m = n*2
    _u = np.zeros((m, 1))
    point_xy = np.zeros((m, 1))
    point_xy[0:n, 0] = pts[:, 0]
    point_xy[n:, 0] = pts[:, 1]
    # for i in range(0, m):
    #     _u[i, 0] = +np.inf
    _u = point_xy + R
    # 等式约束起点和终点
    _u[0, 0]   = pts[0, 0]
    _u[n-1, 0] = pts[n - 1, 0]
    _u[n, 0]   = pts[0, 1]
    _u[m-1, 0] = pts[n - 1, 1]
    return _u

# points = triangular_wave(50, 20, 2)
points = rectangular_wave(50, 20, 2)
# print(points)

P = make_matrix_P(100, 10.0,1.0, 1.0)
Q = make_matrix_Q(points, 100)
A = make_matrix_A(points, 100)
# R = np.inf
R = 3
l = make_matrix_l(points, 100, R)
u = make_matrix_u(points, 100, R)
prob = osqp.OSQP()
prob.setup(P, Q, A, l, u, alpha=1.0)
res = prob.solve()
result = res.x
result_x = result[0:100]
result_y = result[100:]
print([[result_x], [result_y]])

print(type(res))
print(type(result))

print(points.shape)
print(type(points))
# print(points)
plt.figure()
plt.plot(points[:, 0], points[:, 1], 'b--')
plt.plot(result_x, result_y, 'r-')
plt.show()


