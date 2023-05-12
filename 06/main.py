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


# 参数
dt = 0.001
t_start = 0
t_end = 8
times = np.arange(t_start, t_end, dt)

nulllines = np.arange(-80, 0, 0.01)
# 初始化对象
iNaK = Instantaneous_Na_K(40)
v = -90
n = 0.99
state = np.array([v, n])

v_state = []
n_state = []

# 数值积分
for t in times:
    v_state.append(state[0])
    n_state.append(state[1])

    state = iNaK.step(state, dt)
v_state = np.array(v_state)
# print(type(v_state))
n_inf = 1 / (1 + np.exp((-25 - v_state) / 15))

# print(v_state)
plt.figure()
# X = np.arange(-10, 10, 1)
# Y = np.arange(-10, 10, 1)
# U, V = np.meshgrid(X, Y)
#
# fig, ax = plt.subplots()
# q = ax.quiver(X, Y, U, V)
# ax.quiverkey(q, X=0.3, Y=1.1, U=10,
#              label='Quiver key, length = 10', labelpos='E')
plt.plot(nulllines, n_state, label='V-nullcline', linestyle='--')
plt.plot(nulllines, n_inf, label='n-nullcline', linestyle='-')
plt.plot(v_state, n_state, label='trajectories', linestyle='dotted')
plt.xlabel('v ')
plt.ylabel('n ')
plt.ylim(0, 0.7)
plt.legend()
plt.show()
