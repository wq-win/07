import numpy as np
import matplotlib.pyplot as plt


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

class LIF_Neuron:
    def __init__(self,I):
        self.GL = 0.025  # gating
        self.EL = -70  # leak potential
        self.I = I  # inject current
        self.C = 0.5  # capacitance
        # v_peak > v_threshold
        self.v_peak = 30
        self.v_reset = -60
        self.v_threshold = -50

    def derivative(self, state, inputs=0):
        v = state
        Dv = (-self.GL * (v - self.EL) + self.I) / self.C
        return Dv

    def step(self,state,dt,inputs=0):
        v_new = rk4(dt,state, inputs, self.derivative)
        if self.v_threshold < v_new < 0:
            v_new = self.v_peak
        elif v_new > 0:
            v_new = self.v_reset
        return v_new


dt = 0.001
t_start = 0
t_end = 200
times = np.arange(t_start, t_end, dt)


LIF = LIF_Neuron(0.6)
v = -70
v_state = []
for t in times:
    v_state.append(v)
    v = LIF.step(v,dt)


plt.figure()
plt.plot(times,v_state,)
plt.show()
