import matplotlib.pyplot as plt
import numpy as np


class RF_Neuron:
    def __init__(self, ):
        self.b = -0.05
        self.omega = 0.25

    def calculate_next(self, z_x, z_y, I):
        # z = z + (complex(self.b, self.omega)) * z + 0.2
        # z = z +(complex(self.b,self.omega))*z*0.001 + 0.2
        z = complex(z_x, z_y) + (complex(self.b, self.omega)) * complex(z_x, z_y) + I
        return z.real, z.imag


dt = 0.01
t_start = 0
t_end = 4
times = np.arange(t_start, t_end, dt)

RF = RF_Neuron()
I_inputs = np.zeros(len(times))
I_input = np.array([10, 20, 130, 145, 250, 275])
for i in I_input:
    I_inputs[i] = 0.2

z_x = I_inputs * 0
z_y = I_inputs * 0

ind = 0
for I in I_inputs[0:-1]:
    z_x[ind + 1], z_y[ind + 1] = RF.calculate_next(z_x[ind], z_y[ind], I)
    ind += 1

# plt.plot(times, z_y)
plt.plot(times, z_x)
plt.plot(times, I_inputs-1)

# plt.plot(z_x, z_y)
# plt.plot(times, I_inputs)
plt.show()
