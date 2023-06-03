import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


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


class RF_Neuron:
    def __init__(self, I):
        self.GL = 0.25  # gating
        self.EL = -0  # leak potential
        self.I = I  # inject current
        self.C = 0.5  # capacitance
        self.V_half = -0
        self.K=0.25

    def derivative(self, state, inputs=0):
        v, w = state
        Dv = (-self.GL * (v - self.EL) + self.I - w) / self.C
        Dw = (v-self.V_half)/ self.K-w
        return np.array([Dv, Dw])

    def step(self, state, dt, inputs=0):
        state_new = rk4(dt, state, state, self.derivative)
        return state_new


dt = 0.001
t_start = 0
t_end = 20
times = np.arange(t_start, t_end, dt)

RF = RF_Neuron(0.2)
state = np.array([0, 1])
v_state = []
w_state = []

for t in times:
    v_state.append(state[0])
    w_state.append(state[1])
    state = RF.step(state, dt)
print(v_state[-1])
plt.figure()
plt.subplot(122)
plt.plot(times, w_state, )
plt.subplot(121)
plt.plot(v_state, w_state)
plt.show()
