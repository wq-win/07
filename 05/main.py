import matplotlib.pyplot as plt
import numpy as np


def rk4(h, y, inputs, f):
    '''
    用于数值积分的rk4函数。
    args:
        h - 步长
        y - 当前状态量
        inputs - 外界对系统的输入
        f - 常微分或偏微分方程
    return:
        y_new - 新的状态量,即经过h时间之后的状态量
    '''
    k1 = f(y, inputs)
    k2 = f(y + h / 2 * k1, inputs)
    k3 = f(y + h / 2 * k2, inputs)
    k4 = f(y + h * k3, inputs)

    y_new = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_new


# 参数
I = 0
c = 0.01
gL = 19
EL = -67
gNa = 74
v_half = 1.5
k = 16
ENa = 60

# 画图
v_start = -60
v_end = 41
dt = 0.01
v = np.arange(v_start, v_end, dt) # v = np.linspace(v_start,v_end,100)
m_inf = 1 / (1 + np.exp(v_half - v) / k)
print(m_inf)

plt.plot(v, m_inf, label='m_inf')
plt.xlabel('V')
plt.ylabel('m(v)')
plt.legend()
plt.show()
