import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy import sparse
import osqp
import time

# 八邻域搜索
#  ------------------------------------------>
#  |  x(i-1, j-1)   x(x, j-1)   x(i+1, j-1)  |      0   1   2
#  |  x(i-1, j)     x(i, j)     x(i+1, j)    |      7   X   3
#  |  x(i-1, j+1)   x(i, j+1)   x(i+1, j+1)  |      6   5   4
#  <------------------------------------------

# 四邻域搜索
#  ------------------------------------------>
#  |                x(x, j-1)                |          0
#  |  x(i-1, j)     x(i, j)     x(i+1, j)    |      3   X   1
#  |                x(i, j+1)                |          2
#  <------------------------------------------


# optimal the algorithm based on grid map
class Node:
    def __init__(self, x, y, cost, parent=None):
        self.position = [x, y]
        self.parent = parent
        self.cost = cost
        self.g = 0.0  # 从起点到当前节点的实际代价
        self.h = 0.0  # 从当前节点到终点的启发式代价
        self.f = 0.0  # 总代价 f = g + h

    def __eq__(self, other):
        return self.f == other.f

    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        return f"Node({self.position}, cost={self.cost}, g={self.g}, h={self.h}, f={self.f})"

    def same_point(self, other):
        if self.position[0] == other.position[0] and self.position[1] == other.position[1]:
            return True
        else:
            return False



def distance(a, b, method='euclidean'):
    if method == 'euclidean':
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    elif method == 'manhattan':
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    elif method == 'diagonal':
        x_dis = abs(a[0] - b[0])
        y_dis = abs(a[1] - b[1])
        return abs(x_dis - y_dis) + np.sqrt(2)*min([x_dis, y_dis])


def a_in_list(a:Node, node_list : list[Node]):
    for i in range(0, len(node_list)):
        node = node_list[i]
        if a.same_point(node):
            return i, node
    return -1, None

def a_star_search(start, goal, grid, threshold=np.inf, w_g = 1.0, w_h = 1.0, dis_method='euclidean', search_method='four'):
    open_list = []
    closed_list = []
    grid_shape = grid.shape
    open_area = np.zeros(grid_shape, dtype=np.uint8)
    closed_area = np.zeros(grid_shape, dtype=np.uint8)
    start_node = Node(start[0], start[1], grid[start[0]][start[1]])
    start_node.h = distance(start_node.position, start_node.position, dis_method)
    start_node.f = start_node.h
    goal_node = Node(goal[0], goal[1], grid[goal[0]][goal[1]])
    if search_method == 'eight':
        search_list_dir = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    elif search_method == 'four':
        search_list_dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    else:
        search_list_dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    print('start:', start_node)
    print('goal:', goal_node)
    print(f"w_g: {w_g}, w_h:{w_h}, dis_method: {dis_method}, search_method: {search_method}")
    heapq.heappush(open_list, start_node)
    open_area[start[0],  start[1]] = 1
    cnt = 0
    while open_list:
        # print(cnt)
        cnt += 1
        current_node = heapq.heappop(open_list)
        # print(f"{cnt} select: ", current_node)
        if current_node.same_point(goal_node):
            total_cost = current_node.g
            print("total cost:", total_cost)
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            # print('open list =====================================================\n')
            # for node in open_list:
            #     print(node)
            # print('closed list ===================================================\n', closed_list)
            # for node in closed_list:
            #     print(node)
            return open_area, closed_area, path[::-1]

        closed_list.append(current_node)
        closed_area[current_node.position[0], current_node.position[1]] = 1
        # 八邻域搜索
        #  ------------------------------------------>
        #  |  x(i-1, j-1)   x(x, j-1)   x(i+1, j-1)  |      0   1   2
        #  |  x(i-1, j)     x(i, j)     x(i+1, j)    |      7   X   3
        #  |  x(i-1, j+1)   x(i, j+1)   x(i+1, j+1)  |      6   5   4
        #  <------------------------------------------

        # 四邻域搜索
        #  ------------------------------------------>
        #  |                x(x, j-1)                |          0
        #  |  x(i-1, j)     x(i, j)     x(i+1, j)    |      3   X   1
        #  |                x(i, j+1)                |          2
        #  <------------------------------------------

        # for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        for di, dj in search_list_dir:
            neighbor_position = [current_node.position[0] + di, current_node.position[1] + dj]
            # 超出地图范围则忽略
            if not (0 <= neighbor_position[0] < grid_shape[0] and 0 <= neighbor_position[1] < grid_shape[1]):
                continue
            cost = grid[neighbor_position[0], neighbor_position[1]]
            neighbor_node = Node(neighbor_position[0], neighbor_position[1], cost, current_node)
            neighbor_node.g = current_node.g + neighbor_node.cost
            neighbor_node.h = distance(neighbor_node.position, goal_node.position, dis_method)
            # neighbor_node.h = distance(neighbor_node.position, goal_node.position, 'diagonal')
            neighbor_node.f = w_g * neighbor_node.g + w_h * neighbor_node.h

            # 超过阈值视为障碍物，忽略
            if cost >= threshold:
                continue

            # 如果在open表中，则处理
            n_index, old_node = a_in_list(neighbor_node, open_list)
            if n_index == -1:
                pass
            elif neighbor_node.g < old_node.g:
                open_list[n_index].g = neighbor_node.g
                open_list[n_index].f = neighbor_node.f
                open_list[n_index].parent = neighbor_node.parent
                continue
            else:
                continue

            # 如果在closed表中，则处理
            n_index, old_node = a_in_list(neighbor_node, closed_list)
            if n_index == -1:
                pass
            elif neighbor_node.g < old_node.g:
                closed_list[n_index].g = neighbor_node.g
                closed_list[n_index].f = neighbor_node.f
                closed_list[n_index].parent = neighbor_node.parent
                continue
                # del closed_list[n_index]
            else:
                continue


            # if not any(node.g <= neighbor_node.g for node in open_list):
            #     heapq.heappush(open_list, neighbor_node)
            heapq.heappush(open_list, neighbor_node)
            open_area[neighbor_node.position[0], neighbor_node.position[1]] = 1
            # print(neighbor_node)
            # print(open_list)
    return None

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


# # 示例
grid_1 = np.array([
    [0, 0, 0, 0, 1],  # 0
    [9, 0, 9, 0, 1],  # 1
    [0, 0, 0, 0, 1],  # 2
    [0, 9, 4, 1, 0],  # 3
    [9, 0, 0, 0, 0]   # 4
])


start_1 = [0, 0]
goal_1 = [3, 3]

map_1 = cv2.imread('./map3.jpg')
# map_1 = cv2.imread('./white_image.jpg')
map_1 = cv2.cvtColor(map_1, cv2.COLOR_BGR2GRAY)
map_1 = 255 - map_1
kernel = np.ones((2,2), dtype=np.uint8)
map_1_dilate = cv2.dilate(map_1, kernel, iterations=1)
map_1_list = np.array(map_1.tolist())
map_1_dilate_list = np.array(map_1_dilate.tolist())
grid_2_dilate = map_1_dilate_list + 1
grid_2 = map_1_list + 1  # 必须要 +1，防止存在花销为 0 的情况
start_2 = [5, 5]
goal_2 = [95, 60]
# goal_2 = [35, 75]

map_3 = cv2.imread('./labyrinth_3_small.jpg')
# map_1 = cv2.imread('./white_image.jpg')
map_3 = cv2.cvtColor(map_3, cv2.COLOR_BGR2GRAY)
print(type(map_3))
_, map_3 = cv2.threshold(map_3, 40, 255, cv2.THRESH_BINARY)
map_3 = 255 - map_3
kernel = np.ones((3,3), dtype=np.uint8)
map_3_dilate = cv2.dilate(map_3, kernel, iterations=1)
map_3_list = np.array(map_3.tolist())
map_3_dilate_list = np.array(map_3_dilate.tolist())
grid_3 = map_3_list + 1  # 必须要 +1，防止存在花销为 0 的情况
grid_3_dilate = map_3_dilate_list + 1
start_3 = [2, 2]
goal_3 = [45, 45]
# goal_2 = [35, 75]

# grid_dilate = grid_3_dilate
# grid = grid_3
# # open_area = np.zeros(grid.shape)
# start = start_3
# goal = goal_3

grid_dilate = grid_2_dilate
grid = grid_2
# open_area = np.zeros(grid.shape)
start = start_2
goal = goal_2


print(grid.shape)
start_time = time.time()
open_area, closed_area, path = a_star_search(start, goal, grid_dilate, 40, 0.6, 1.0, 'euclidean', 'eight')
open_area *= 200
# grid += open_area
# path = a_star_search(start, goal, grid, 30, 1.0, 1.0, 'diagonal', 'eight')
# path = a_star_search(start, goal, grid, 30, 1.0, 1.0, 'manhattan', 'eight')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"A*算法执行时间: {elapsed_time} 秒")
if not path is None:
    for i in path:
        temp = i[0]
        i[0] = i[1]
        i[1] = temp
    points = np.array(path)
    p_shape = points.shape
    v_num = p_shape[0]
    print(p_shape)
    # print(points)

    P = make_matrix_P(v_num, 1.0,1.0, 1.0)
    Q = make_matrix_Q(points, v_num)
    A = make_matrix_A(points, v_num)
    R = np.inf
    R = 2
    l = make_matrix_l(points, v_num, R)
    u = make_matrix_u(points, v_num, R)
    prob = osqp.OSQP()
    prob.setup(P, Q, A, l, u, alpha=1.0)
    res = prob.solve()
    result = res.x
    result_x = result[0:v_num]
    result_y = result[v_num:]
    # print([[result_x], [result_y]])
    # print("Path:", path)


    # 设置图片分辨率
    # plt.figure(dpi=800)
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    print(fig)
    print(ax)
    # 绘制栅格地图
    # ax.imshow(grid, cmap='Greys', origin='upper')
    # cmap = plt.get_cmap('viridis')
    cmap = plt.get_cmap('YlGnBu')
    # cmap = plt.get_cmap('Grays')
    # cmap = plt.get_cmap('summer')
    ax.imshow(grid, cmap=cmap, interpolation='nearest')
    # ax.imshow(grid, cmap=cmap, interpolation='none')
    cmap = plt.get_cmap('summer')
    masked1 = np.ma.masked_where(open_area == 0, open_area)
    masked2 = np.ma.masked_where(closed_area == 0, closed_area)
    # print(masked1)
    # ax.imshow(open_area, cmap=cmap, interpolation='nearest', alpha=0.3)
    ax.imshow(masked1, cmap=cmap, interpolation='nearest', alpha=0.5)
    # ax.imshow(open_area, cmap=cmap, interpolation='nearest', alpha=1.0)
    # ax.invert_yaxis()  # 反转Y坐标轴
    # 绘制路径
    path_x, path_y = zip(*path)
    # ax.plot(path_x, path_y, color='blue', marker='o', linestyle='-', linewidth=3, markersize=1)
    ax.plot(path_x, path_y, color='blue', linestyle='--', linewidth=1)
    ax.plot(result_x, result_y, color='red', linestyle='-', linewidth=2)
    ax.plot(start[1], start[0], color='blue', marker='o', markersize=10)
    ax.plot(goal[1], goal[0], color='green', marker='*', markersize=20)
    # # 设置轴的范围
    # ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    # ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    #
    # # 设置轴的刻度
    # ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    # ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    # ax.set_xticks(np.arange(0, grid.shape[1], 1), minor=False)
    # ax.set_yticks(np.arange(0, grid.shape[0], 1), minor=False)

    # 绘制网格线
    # ax.grid(which='minor', color='black', linestyle='-', linewidth=0)

    # 去掉背景边框
    # plt.subplots_adjust(left=0.1, right=0.1, top=0.1, bottom=0.1)
    # fig.patch.set_facecolor('none')
    # ax.patch.set_facecolor('none')
    # 去掉轴的标签
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

    # 显示图形
    plt.show()

else:
    print("No path found!")