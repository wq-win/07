import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import electrocardiogram
from scipy.signal import find_peaks

bm.set_platform('cpu')

print(bp.__version__)
np.random.seed(1)

noise = np.random.normal(0, 0.1, 1000)
# x=np.arrange(-20*np.pi,20*np.pi,0.01)
x = np.linspace(-20 * np.pi, 20 * np.pi, 1000)
y = np.sin(x) + noise
peaks1,_=find_peaks(y,)

peaks, _ = find_peaks(y, width=1)

plt.xlim(-20 * np.pi, 20 * np.pi)
plt.plot(x, y, label='sin')
plt.plot(x[peaks1], y[peaks1], "x",label='original',color='r')
plt.plot(x[peaks], y[peaks], "x", label='change',color='g')
plt.axhline(linewidth=1, color='r')
plt.show()

# plt.plot(np.zeros_like(x), "--", color="gray")
