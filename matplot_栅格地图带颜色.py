import matplotlib.pyplot as plt
import numpy as np

# 定义栅格地图，其中不同的数字代表不同的权值
grid = np.array([
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8]
])

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制栅格地图，使用颜色映射来表示不同的权值
# 这里使用了'viridis'颜色映射，你可以根据需要选择其他颜色映射
cmap = plt.get_cmap('viridis')
ax.imshow(grid, cmap=cmap, interpolation='nearest')
ax.invert_yaxis()

# 添加颜色条
plt.colorbar(ax.images[0], ax=ax)

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


