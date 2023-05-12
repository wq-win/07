import matplotlib.pyplot as plt
import numpy as np

X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
# meshgrid 生成网格，此处生成两个 shape = (20,20) 的 ndarray, 详见参考资料2,3
U, V = np.meshgrid(X, Y)

# X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
# U, V = np.cos(X), np.sin(Y)


fig, ax = plt.subplots()
# 绘制箭头
# q = ax.quiver(X, Y, U, V, )
ax.quiver(X, Y, U, V, pivot='mid',color='r')
# 该函数绘制一个箭头标签在 (X, Y) 处， 长度为U, 详见参考资料4
# ax.quiverkey(q, X=0.3, Y=1.1, U=10,
#              label='Quiver key, length = 10',labelpos='E')
# plt.axis('equal')
# plt.xticks(range(-5, 6))
# plt.yticks(range(-5, 6))
plt.grid()
plt.show()
