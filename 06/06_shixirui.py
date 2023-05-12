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

C = 1 * 1e-9
I = 40 * 1e-12
gl = 8 * 1e-9
El = -80 * 1e-3
gna = 20 * 1e-9
V12na = -20 * 1e-3
kna = 15 * 1e-3
Ena = 60 * 1e-3
gk = 10 * 1e-9
V12k = -25 * 1e-3
kk = 5 * 1e-3
Ek = -90 * 1e-3

MV = lambda v: 1 / (1 + np.exp((V12na - v)/kna))
CV = lambda v, I, n: I - gl * (v - El) - gna * MV(v) * (v - Ena) - gk * n * (v - Ek)
NV = lambda v: 1 / (1 + np.exp((V12k - v) / kk))
#实现类

class Persistent_Current:
    def __init__(self, C, I):
        self.C = C
        self.I = I
    def function(self, state, inputs=0):
        V = state[0]
        n = state[1]
        DV = CV(V, self.I, n) / self.C
        Dn = NV(V) - n
        return np.array([DV, Dn])

    def step(self, state, dt, inputs=0):
        state_new = rk4(dt, state, inputs, self.function)
        return state_new

#2 V的零斜率线
dtv = 0.1 * 1e-3

V = np.arange(-89.9 * 1e-3, 20 * 1e-3, dtv)

V_nullcline = [(I - gl*(v - El) - gna*MV(v)*(v-Ena))/(gk * (v - Ek)) for v in V]

#n的零斜率线
n_nullcline = [NV(v) for v in V]
plt.plot(V, V_nullcline)
plt.plot(V, n_nullcline)
plt.legend(['V_zeroslopeline', 'n_zeroslopeline'])
plt.ylim([0, 1])
plt.xlabel('V')
plt.ylabel('n')


a = np.arange(-89.9 * 1e-3, 20 * 1e-3, 4*1e-3)
b = np.arange(0, 1, 0.04)
x, y = np.meshgrid(a, b)

dx = CV(x, I, y) / C
dy = NV(x) - y
# print(dx)
plt.quiver(x, y, dx, dy, angles='xy')
plt.plot(V, V_nullcline, label='V-zero_slopeline')
plt.plot(V, n_nullcline, label='n-zero_slopeline')
plt.ylim([0, 0.7])
plt.legend()
plt.xlabel('V')
plt.ylabel('n')
plt.show()



###
dt = 1e-3
t_end = 5
times = np.arange(0, t_end, dt)

plt.figure()
state = np.array([-40 * 1e-3, 0])
PC0 = Persistent_Current(C, I)
vs = []
ns = []
for t in times:
    vs.append(state[0])
    ns.append(state[1])
    state = PC0.step(state, dt)
plt.quiver(x, y, dx, dy)
plt.plot(V, V_nullcline, label='V_zeroslopeline')
plt.plot(V, n_nullcline, label='n_zeroslopeline')
plt.plot(vs, ns, '--', label='activities')

for i in [1, 300, 300 + len(vs)//5, 300 + 2*len(vs)//5, 300 + 3*len(vs)//5, 300 + 4*len(vs)//5]:
    plt.annotate("", xy=(vs[i], ns[i]), xytext=(vs[i-1], ns[i-1]),
            arrowprops=dict(arrowstyle="->"))


plt.ylim([0, 0.7])
plt.legend()
plt.xlabel('V')
plt.ylabel('n')
plt.show()
