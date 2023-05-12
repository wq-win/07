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


class Twice_Current:
    def __init__(self, I, Vpeak, Vreset):
        self.I = I
        self.rest = -3
        self.thresold = -2
        self.Vpeak = Vpeak
        self.Vreset = Vreset

    def function(self, state, inputs=0):
        v = state[0]
        return np.array([self.I + v ** 2])

    def step(self, state, dt, inputs=0):
        state_new = rk4(dt, state, inputs, self.function)
        return state_new

    # 3
dt = 1e-3
t_end = 20
times = np.arange(0, t_end, dt)

# 3. I = 0.5, Vpeak = 10, Vreset = -4, Vthresold = -2, Vrest = -3
plt.figure()

state = np.array([0])
TC = Twice_Current(0.5, 5, -4)
vs = []
for t in times:
    vs.append(state[0])
    state = TC.step(state, dt)
    if state[0] >= TC.Vpeak:
        state[0] = TC.Vreset
    # elif state[0] >= TC.thresold:
    #     state[0] = TC.Vpeak
    elif state[0] < TC.rest:
        state[0] = TC.rest
# print(vs)
plt.plot(times, vs)
plt.xlabel('time')
plt.ylabel('membrane potential V')

#4
dt = 1e-3
t_end = 20
times = np.arange(0, t_end, dt)

#3. I = -0.5, Vpeak = 10, Vreset = -1, Vthresold = -2, Vrest = -3
plt.figure()

state = np.array([0])
TC = Twice_Current(-0.5, 5, -1)
vs = []
for t in times:
    vs.append(state[0])
    state = TC.step(state, dt)
    if state[0] >= TC.Vpeak:
        state[0] = TC.Vreset
    # elif state[0] >= TC.thresold:
    #     state[0] = TC.Vpeak
    elif state[0] < TC.rest:
        state[0] = TC.rest
# print(vs)
plt.plot(times, vs)
plt.xlabel('time')
plt.ylabel('membrane potential V')
plt.show()
#5
dt = 1e-3
t_end = 1
times = np.arange(0, t_end, dt)

#3. I = -0.5, Vpeak = 10, Vreset = -1, Vthresold = -2, Vrest = -3
plt.figure()

state = np.array([0])
TC = Twice_Current(-0.5, 5, -2.5)
vs = []
for t in times:
    vs.append(state[0])
    state = TC.step(state, dt)
    if state[0] >= TC.Vpeak:
        state[0] = TC.Vreset
    elif state[0] >= TC.thresold:
        state[0] = TC.Vpeak
    elif state[0] < TC.rest:
        state[0] = TC.rest
# print(vs)
plt.plot(times, vs)
plt.xlabel('time')
plt.ylabel('membrane potential V')

