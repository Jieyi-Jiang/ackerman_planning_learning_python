import matplotlib.pyplot as plt
import numpy as np

# 创建一个新的图形和坐标轴
fig, ax = plt.subplots()

# 初始化一个空列表来存储轨迹点
x_data, y_data = [], []

# 设置坐标轴范围
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')


# 实时更新函数
def update_plot(x, y):
    # 将新的轨迹点添加到列表中
    x_data.append(x)
    y_data.append(y)

    # 绘制新的轨迹点
    ax.plot(x_data, y_data, 'bo-')  # 'bo-' 表示蓝色圆点连线

    # 刷新图形
    plt.draw()
    plt.pause(1)  # 暂停0.01秒以更新图形


# 模拟实时数据生成
for i in range(100):
    # 生成随机的二维坐标点
    x = np.random.rand() * 10  # 生成0到10之间的随机数
    y = np.random.rand() * 10  # 生成0到10之间的随机数

    # 更新图形
    update_plot(x, y)

# 显示图形
plt.show()
