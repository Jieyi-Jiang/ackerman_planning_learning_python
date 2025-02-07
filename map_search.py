import heapq
import numpy as np

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
    for node in node_list:
        if a.same_point(node):
            return True
    return False

def a_star_search(start, goal, grid, threshold=np.inf):
    open_list = []
    closed_list = []
    grid_shape = grid.shape
    start_node = Node(start[0], start[1], grid[start[0]][start[1]])
    goal_node = Node(goal[0], goal[1], grid[goal[0]][goal[1]])
    print('start:', start_node)
    print('goal:', goal_node)

    heapq.heappush(open_list, start_node)

    cnt = 0
    while open_list:
        print(cnt)
        cnt += 1
        current_node = heapq.heappop(open_list)
        if current_node.same_point(goal_node):
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_list.append(current_node)

        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            neighbor_position = [current_node.position[0] + dx, current_node.position[1] + dy]
            if not (0 <= neighbor_position[0] < grid_shape[0] and 0 <= neighbor_position[1] < grid_shape[1]):
                continue
            if grid[neighbor_position[0]][neighbor_position[1]] >= threshold:
                continue
            neighbor_node = Node(neighbor_position[0], neighbor_position[1], grid[neighbor_position[0], neighbor_position[1]], current_node)

            if a_in_list(neighbor_node, closed_list):
                continue

            neighbor_node.g = current_node.g + current_node.cost
            neighbor_node.h = distance(neighbor_node.position, goal_node.position, 'manhattan')
            neighbor_node.f = 1.0 * neighbor_node.g  +  1.0 * neighbor_node.h

            if not any(node == neighbor_node and node.g < neighbor_node.g for node in open_list):
                heapq.heappush(open_list, neighbor_node)
            print(neighbor_node)
            # print(open_list)

    return None


# 示例
grid = np.array([
    [0, 0, 0, 0, 1],  # 0
    [9, 0, 9, 0, 1],  # 1
    [0, 0, 0, 0, 1],  # 2
    [0, 9, 1, 1, 0],  # 3
    [2, 0, 0, 0, 0]   # 4
])

# grid = np.zeros((6, 6))
start = (0, 0)
goal = (4, 1)

path = a_star_search(start, goal, grid, 3)
print("Path:", path)