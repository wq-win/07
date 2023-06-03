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
        self.I = I
        self.b = -0.05
        self.omega = 0.25
        self.z_reset = 0+1j

    def derivative(self, state, inputs=0):
        z = state
        Dz = (complex(self.b, self.omega)) * z + self.I
        return Dz

    def step(self, state, dt, inputs=0):
        z_new = rk4(dt, state, inputs, self.derivative)
        if z_new.imag >= 1:
            z_new = self.z_reset
        return z_new


dt = 0.001
t_start = 0
t_end = 100
times = np.arange(t_start, t_end, dt)

RF = RF_Neuron(0.2)
z = 0 + 0j
z_real = []
z_imag = []
for t in times:
    z_real.append(z.real)
    z_imag.append(z.imag)
    z = RF.step(z, dt)

peaks, _ = find_peaks(z_imag )  # peaks = [15708,40841,65973,91106]

plt.figure()
plt.subplot(122)
plt.plot(times, z_imag,)
plt.vlines(times[peaks[0]],z_imag[peaks[0]],2)
plt.xlabel('time,ms')
plt.ylabel('y')
plt.plot(times, np.ones(len(times)),':',label='threshold')

plt.subplot(121)
plt.plot(z_real, z_imag)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(z_real, np.ones(len(z_real)),':',label='threshold')
plt.legend()
plt.show()
