import numpy as np
from matplotlib import pyplot as plt


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


class Integrate_Fire:
    def __init__(self, I):
        self.I = I
        # v_peak > v_threshold
        self.v_peak = 2
        self.v_reset = 1
        self.v_threshold = 0

    def derivative(self, v, inputs=0):
        Dv = self.I + v * v
        return Dv

    def step(self, v, dt, inputs=0):
        v_new = rk4(dt, v, inputs, self.derivative)
        if v_new >= self.v_peak:
            v_new = self.v_reset
        return v_new


# 参数
I = -0.01
dt = 0.001
t_start = 0
t_end = 20
times = np.arange(t_start, t_end, dt)

# 初始化对象
I_and_F = Integrate_Fire(I)
v = 0.1 # 静息电位开始
v_state = []
v_reset = []
v_threshold = []
# 数值积分
for t in times:
    v_state.append(v)
    v_reset.append(1)
    v_threshold.append(0)
    v = I_and_F.step(v, dt)

# print(v_state)
plt.figure()
plt.plot(times, v_state, label='v')
# plt.plot(times, v_reset, label='v_reset')
# plt.plot(times,v_threshold, label='v_threshold')
plt.xlabel('time (ms)')
plt.ylabel('V (mv)')
plt.legend()
plt.show()