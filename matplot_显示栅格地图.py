import matplotlib.pyplot as plt
import numpy as np

# 定义栅格地图
grid = np.array([
    [0, 0, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0]
])

# 定义路径
path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3)]

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制栅格地图
# ax.imshow(grid, cmap='Greys', origin='upper')
cmap = plt.get_cmap('viridis')
ax.imshow(grid, cmap=cmap, interpolation='nearest')
ax.invert_yaxis()  # 反转Y坐标轴

# 绘制路径
path_x, path_y = zip(*path)
ax.plot(path_x, path_y, color='blue', marker='o', linestyle='-', linewidth=2, markersize=8)

# 设置轴的范围
ax.set_xlim(-0.5, grid.shape[1] - 0.5)
ax.set_ylim(-0.5, grid.shape[0] - 0.5)

# 设置轴的刻度
ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
ax.set_xticks(np.arange(0, grid.shape[1], 1), minor=False)
ax.set_yticks(np.arange(0, grid.shape[0], 1), minor=False)

# 绘制网格线
ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

# 去掉轴的标签
ax.set_xticklabels([])
ax.set_yticklabels([])

# 显示图形
plt.show()




# import matplotlib.pyplot as plt
#
# x=[1,2,3,4,5,6]
# y=[10,20,30,40,50,60]
# plt.plot(x, y, color='red')
# plt.show()
#
# x = [1, 2, 3, 4, 5, 6]
# y = [10, 20, 30, 40, 50, 60]
# plt.plot(x, y, color='red')
#
# ax = plt.gca()  # 获取到当前坐标轴信息
# ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
# ax.invert_yaxis()  # 反转Y坐标轴
#
# plt.show()

