import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

bm.set_platform('cpu')


@bp.odeint(method='rk4', dt=0.01)
def VanderPol(x,t,miu):
    dydt = x

    dzdt = -x - miu*(1/3 * dydt**3- dydt)
    return dydt,dzdt


miu = 0.01

runner = bp.IntegratorRunner(
    VanderPol,
    monitors=list('yz'),
    inits=[0.25, 0.],
    args=dict(miu=miu),
    dt=0.01
)

runner.run(100.)

plt.plot(runner.mon.ts, runner.mon.y, label='y')
plt.plot(runner.mon.ts, runner.mon.z, label='z')

plt.legend()
plt.show()