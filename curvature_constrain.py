import numpy as np
from numpy.ma.core import power
from numpy.matlib import zeros
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
        pts[1 , i*period : i*period + round(period/2) ] = np.linspace(0, magnitude, round(period/2))
        pts[1, i*period + round(period/2): (i + 1) * period] = np.linspace(magnitude, 0, period - round(period/2))
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
    _A = np.zeros((m + (m-4), m))
    A_pos = np.eye(m, m)
    H = np.zeros((m - 4, m))
    for i in range(0, n - 2):
        H[i, i:i + 3] = [4, -8, 4]
    H[n-2:, n:] = H[0:n-2, 0:n]
    _A[0:m, :] = A_pos
    _A[m:, :] = H
    return sparse.csc_matrix(_A)

def make_matrix_l(pts, n=100, R=np.inf, R_min = 0.0):
    m = n*2
    _l = np.zeros(( m + (m-4), 1))
    point_xy = np.zeros((m, 1))
    point_xy[0:n, 0] = pts[:, 0]
    point_xy[n:, 0] = pts[:, 1]
    # for i in range(0, m):
    #     _l[i, 0] = -np.inf
    # 距离约束
    l_pos = np.zeros((m, 1))
    l_pos = point_xy - R
    # 等式约束起点和终点
    l_pos[0, 0]    = pts[0, 0]
    l_pos[n-1, 0]  = pts[n-1, 0]
    l_pos[n, 0]    = pts[0, 1]
    l_pos[m-1, 0]  = pts[n-1, 1]
    _l[0:m, 0] = l_pos[:, 0]

    # 曲率约束
    # l_cur = zeros((m - 4, 1))
    delta_d = np.zeros((m - 4, 1))
    # for i in range(1, n - 2):
    #     # delta_d[i, 0] += np.sqrt( np.power(pts[i][0] - pts[i-1][0], 2) + np.power(pts[i][1]- pts[i-1][1], 2) )
    #     # delta_d[i, 0] += np.sqrt( np.power(pts[i+1][0] - pts[i][0], 2) + np.power(pts[i+1][1] - pts[i][1], 2) )
    #     delta_d[i, 0] += np.power(pts[i][0] - pts[i - 1][0], 2) + np.power(pts[i][1] - pts[i - 1][1], 2)
    #     delta_d[i, 0] += np.power(pts[i + 1][0] - pts[i][0], 2) + np.power(pts[i + 1][1] - pts[i][1], 2)
    #     delta_d[i, 0] /= 2
    # delta_d = np.power((delta_d / R_min), 2)
    for i in range(1, n - 2):
        delta_d[i, 0] = np.power(R_min, 2)
        # delta_d[i, 0] = -np.inf
    delta_d[n-2:] = delta_d[0:n-2]

    F_pref = np.zeros((m - 4, 1))
    for i in range(1, n - 2):
        F_pref[i, 0] += np.power(2 * pts[i][0] - pts[i - 1][0] - pts[i + 1][0], 2)
        F_pref[i, 0] += np.power(2 * pts[i][1] - pts[i - 1][1] - pts[i + 1][1], 2)
    F_pref[n-2:] = F_pref[:n-2]

    H = np.zeros((m - 4, m))
    for i in range(0, n - 2):
        H[i, i:i + 3] = [4, -8, 4]
    H[n-2:, n:] = H[0:n-2, 0:n]
    F_diff_dot_Pos = np.dot(H, point_xy)
    l_cur = delta_d - F_pref + F_diff_dot_Pos
    # for i in range(0, m - 4):
    #     l_cur[i, 0] = -np.inf
    _l[m:] = l_cur

    return _l

def make_matrix_u(pts, n=100, R=np.inf, R_min=4.0):
    m = n*2
    _u = np.zeros((m + (m-4), 1))
    point_xy = np.zeros((m, 1))
    point_xy[0:n, 0] = pts[:, 0]
    point_xy[n:, 0] = pts[:, 1]
    # for i in range(0, m):
    #     _u[i, 0] = +np.inf

    # 位置约束
    u_pos = np.zeros((m, 1))
    u_pos = point_xy + R

    # 等式约束起点和终点
    u_pos[0, 0]   = pts[0, 0]
    u_pos[n-1, 0] = pts[n - 1, 0]
    u_pos[n, 0]   = pts[0, 1]
    u_pos[m-1, 0] = pts[n - 1, 1]
    _u[0:m, 0] = u_pos[:, 0]

    # # 曲率约束
    # delta_d = np.zeros((m - 4, 1))
    # # for i in range(1, n - 2):
    # #     # delta_d[i, 0] += np.sqrt( np.power(pts[i][0] - pts[i-1][0], 2) + np.power(pts[i][1]- pts[i-1][1], 2) )
    # #     # delta_d[i, 0] += np.sqrt( np.power(pts[i+1][0] - pts[i][0], 2) + np.power(pts[i+1][1] - pts[i][1], 2) )
    # #     delta_d[i, 0] += np.power(pts[i][0] - pts[i - 1][0], 2) + np.power(pts[i][1] - pts[i - 1][1], 2)
    # #     delta_d[i, 0] += np.power(pts[i + 1][0] - pts[i][0], 2) + np.power(pts[i + 1][1] - pts[i][1], 2)
    # #     delta_d[i, 0] /= 2
    # # delta_d = np.power((delta_d / R_min), 2)
    # for i in range(1, n - 2):
    #     delta_d[i, 0] = np.power(R_min, 2)
    # delta_d[n-2:] = delta_d[0:n-2]
    #
    # F_pref = np.zeros((m - 4, 1))
    # for i in range(1, n - 2):
    #     F_pref[i, 0] += np.power(2 * pts[i][0] - pts[i - 1][0] - pts[i + 1][0], 2)
    #     F_pref[i, 0] += np.power(2 * pts[i][1] - pts[i - 1][1] - pts[i + 1][1], 2)
    # F_pref[n-2:] = F_pref[:n-2]
    #
    # H = np.zeros((m - 4, m))
    # for i in range(0, n - 2):
    #     H[i, i:i + 3] = [4, -8, 4]
    # H[n-2:, n:] = H[0:n-2, 0:n]
    # F_diff_dot_Pos = np.dot(H, point_xy)
    # u_cur = delta_d - F_pref + F_diff_dot_Pos

    # print('u_cur -----------------------------------------------')
    # print(u_cur)
    # print(u_cur.shape)
    u_cur = zeros((m - 4, 1))
    for i in range(0, m - 4):
        u_cur[i, 0] = +np.inf
    _u[m:, :] = u_cur
    return _u

# points = triangular_wave(50, 20, 2)
points = rectangular_wave(50, 20, 2)
# print(points)

P = make_matrix_P(100, 5.0,1.0, 1.0)
Q = make_matrix_Q(points, 100)
A = make_matrix_A(points, 100)
print(A.shape)
# R = np.inf
R = 10000
l = make_matrix_l(points, 100, R, 0.0)
print(l.shape)
u = make_matrix_u(points, 100, R, 10000000000)
# print(u.shape)
# print(l[4], u[4])
# print('l -----------------------------------------------------------------------------------------')
# print(l)
# print('u -----------------------------------------------------------------------------------------')
# print(u)
prob = osqp.OSQP()
prob.setup(P, Q, A, l, u, alpha=1.0)
res = prob.solve()
result = res.x
result_x = result[0:100]
result_y = result[100:]
# print([[result_x], [result_y]])

# print(type(res))
# print(type(result))
#
# print(points.shape)
# print(type(points))
# print(points)
plt.figure()
plt.plot(points[:, 0], points[:, 1], 'b--')
plt.plot(result_x, result_y, 'r-')
plt.show()


