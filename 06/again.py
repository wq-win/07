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

class Instantaneous_Na_K:
    def __init__(self, I):
        self.I = I
        self.c = 1
        self.gL = 8
        self.EL = -80
        self.gNa = 20
        # self.v_half = 1.5
        self.v_half_Na = -20
        self.kNa = 15
        self.ENa = 60
        self.gK = 10
        self.v_half_K = -25
        self.kK = 5
        self.EK = -90

        self.tao = 1

    def derivative(self, state, inputs=0):
        v, n = state

        m_inf = 1 / (1 + np.exp((self.v_half_Na - v) / self.kNa))

        n_inf = 1 / (1 + np.exp((self.v_half_K - v) / self.kNa))

        Dn = (n_inf - n) / self.tao

        Dv = (self.I - self.gL * (v - self.EL) - self.gNa * m_inf * (v - self.ENa) - self.gK * n * (
                v - self.EK)) / self.c

        return np.array([Dv, Dn])

    def step(self, state, dt, inputs=0):
        state_new = rk4(dt, state, inputs, self.derivative)
        return state_new


