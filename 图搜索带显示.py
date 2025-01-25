import heapq
import numpy as np
import matplotlib.pyplot as plt

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
    start_node = Node(start[0], start[1], grid[start[0]][start[1]])
    start_node.h = distance(start_node.position, start_node.position, dis_method)
    start_node.f = start_node.h
    goal_node = Node(goal[0], goal[1], grid[goal[0]][goal[1]])
    print('start:', start_node)
    print('goal:', goal_node)

    heapq.heappush(open_list, start_node)

    cnt = 0
    while open_list:
        # print(cnt)
        cnt += 1
        current_node = heapq.heappop(open_list)
        # print(f"{cnt} select: ", current_node)
        if current_node.same_point(goal_node):
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
            return path[::-1]

        closed_list.append(current_node)

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

        if search_method == 'eight':
            search_list_dir = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        elif search_method == 'four':
            search_list_dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        else:
            search_list_dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
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

            n_index, old_node = a_in_list(neighbor_node, open_list)
            if n_index == -1:
                pass
            # elif old_node.g > neighbor_node.g:
            #     continue
            # else:
            #     open_list.remove(old_node)
            elif neighbor_node.g < old_node.g:
                del open_list[n_index]
                # open_list.remove(old_node)
            else:
                continue
            # elif not old_node.g > neighbor_node.g:
            #     open_list.remove(old_node)
            # else:
            #     continue

            n_index, old_node = a_in_list(neighbor_node, closed_list)
            if n_index == -1:
                pass
            # elif old_node.g > neighbor_node.g:
            #     continue
            # else:
            #     closed_list.remove(old_node)
            elif neighbor_node.g < old_node.g:
                del open_list[n_index]
            else:
                continue

            # if not any(node.g <= neighbor_node.g for node in open_list):
            #     heapq.heappush(open_list, neighbor_node)
            heapq.heappush(open_list, neighbor_node)
            # print(neighbor_node)
            # print(open_list)
    return None


# 示例
grid = np.array([
    [0, 0, 0, 0, 1],  # 0
    [9, 0, 9, 0, 1],  # 1
    [0, 0, 0, 0, 1],  # 2
    [0, 2, 4, 1, 0],  # 3
    [10, 0, 10, 0, 0]   # 4
])
grid += 1

# grid = np.zeros((6, 6))
start = (0, 0)
goal = (4, 1)

path = a_star_search(start, goal, grid, 100)
if not path is None:
    for i in path:
        temp = i[0]
        i[0] = i[1]
        i[1] = temp

print("Path:", path)


# 创建图形和轴
fig, ax = plt.subplots()

# 绘制栅格地图
# ax.imshow(grid, cmap='Greys', origin='upper')
cmap = plt.get_cmap('viridis')
ax.imshow(grid, cmap=cmap, interpolation='nearest')
# ax.invert_yaxis()  # 反转Y坐标轴
# 绘制路径
path_x, path_y = zip(*path)
# ax.plot(path_x, path_y, color='blue', marker='o', linestyle='-', linewidth=2, markersize=8)
ax.plot(path_x, path_y, color='blue', linestyle='-', linewidth=2)
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
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

# 去掉轴的标签
ax.set_xticklabels([])
ax.set_yticklabels([])

# 显示图形
plt.show()