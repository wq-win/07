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
        self.c = 1.0
        self.gL = 8.0
        self.EL = -80.0
        self.gNa = 20.0
        self.v_half_Na = -20.0
        self.kNa = 15.0
        self.ENa = 60.0
        self.gK = 10.0
        self.v_half_K = -25.0
        self.kK = 5.0
        self.EK = -90.0
        self.tao = 1.0

    def derivative(self, state, inputs=0):
        v, n = state

        m_inf = 1 / (1 + np.exp((self.v_half_Na - v) / self.kNa))

        n_inf = 1 / (1 + np.exp((self.v_half_K - v) / self.kK))

        Dn = (n_inf - n) / self.tao

        Dv = (self.I - self.gL * (v - self.EL) - self.gNa * m_inf * (v - self.ENa))-(self.gK *n* (v - self.EK))

        return np.array([Dv, Dn])

    def step(self, state, dt, inputs=0):
        state_new = rk4(dt, state, inputs, self.derivative)
        return state_new

    def draw_m(self, v):
        m_inf = 1 / (1 + np.exp((self.v_half_Na - v) / self.kNa))
        m_inf =(self.I-self.gL*(v-self.EL)-self.gNa*m_inf*(v-self.ENa))/(self.gK*(v-self.EK))
        return m_inf

    def draw_n(self, v):
        n_inf = 1 / (1 + np.exp((self.v_half_K - v) / self.kK))
        return n_inf


# parameter
dt = 0.001
t_start = 0
t_end = 5
times = np.arange(t_start, t_end, dt)

# init class
iNaK=Instantaneous_Na_K(40.0)

v = np.arange(-80.0, 20.0, 0.01)
v_nullcline = iNaK.draw_m(v)
n_nullcline = iNaK.draw_n(v)

# init trajectory
state = np.array([-40, 0.045])

v_state = []
n_state = []

# stepping
for t in times:
    v_state.append(state[0])
    n_state.append(state[1])
    state = iNaK.step(state, dt)

# quiver
# a = np.arange(-80, 20, 0.01)
# b = np.arange(0, 1, 0.01)
# x, y = np.meshgrid(v, n_nullcline)
# dx = v_state
# dy = n_state
#
# # painting
# plt.figure()
# plt.quiver(x,y,dx,dy,angles='xy')
# plt.quiver(v,n_nullcline,x,y,)

plt.plot(v, v_nullcline, label='V-nullcline', linestyle='-')
plt.plot(v, n_nullcline, label='n-nullcline', linestyle='-')
plt.plot(v_state,n_state, label='trajectories', linestyle='dotted')
plt.xlabel('V')
plt.ylabel('n')
plt.ylim(0, 0.7)
plt.legend()
plt.show()
