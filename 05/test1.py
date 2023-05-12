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


class Instantaneous_Na:
    def __init__(self, I):
        self.I = I
        self.c = 10
        self.gL = 19
        self.EL = -67
        self.gNa = 74
        self.v_half = 1.5
        self.k = 16
        self.ENa = 60

    def derivative(self, v, inputs=0):
        m_inf = 1 / (1 + np.exp((self.v_half - v) / self.k))
        Dv = (self.I - self.gL * (v - self.EL) - self.gNa * m_inf * (v - self.ENa))/ self.c
        return Dv

    def step(self, v, dt, inputs=0):
        v_new = rk4(dt, v, inputs, self.derivative)
        return v_new


# 参数
dt = 0.001
t_start = 0
t_end = 5
times = np.arange(t_start, t_end, dt)

# 初始化对象
iNa = Instantaneous_Na(60)
v = np.arange(-60, 45, 5)
v_state = []
# 数值积分
for t in times:
    v_state.append(v)
    v = iNa.step(v, dt)

# print(v_state)
plt.figure()
plt.plot(times,v_state)
plt.xlabel('time (ms)')
plt.ylabel('mempV (mv)')
plt.legend()
plt.show()
