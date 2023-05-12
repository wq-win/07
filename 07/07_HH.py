import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

bm.set_platform('cpu')


@bp.odeint(method='rk4', dt=0.01)
def integral(V, m, h, n, t, Iext, gNa, ENa, gK, EK, gL, EL, C):
    alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    beta = 4.0 * bm.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m

    alpha = 0.07 * bm.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bm.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h

    alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta = 0.125 * bm.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n

    # dhdt = 0.89-1.1* n

    I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
    I_K = (gK * n ** 4.0) * (V - EK)
    I_leak = gL * (V - EL)
    dVdt = (- I_Na - I_K - I_leak + Iext) / C

    return dVdt, dmdt, dhdt, dndt


Iext = 10.;   ENa = 50.;   EK = -77.;   EL = -54.387
C = 1.0;      gNa = 120.;  gK = 36.;    gL = 0.03

runner = bp.IntegratorRunner(
    integral,
    monitors=list('Vmhn'),
    inits=[0., 0., 0., 0.],
    args=dict(Iext=Iext, gNa=gNa, ENa=ENa, gK=gK, EK=EK, gL=gL, EL=EL, C=C),
    dt=0.01
)
runner.run(100.)

plt.subplot(211)
plt.plot(runner.mon.ts, runner.mon.V+75, label='V')
plt.legend()
plt.subplot(212)
plt.plot(runner.mon.ts, runner.mon.m, label='m')
plt.plot(runner.mon.ts, runner.mon.h, label='h')
plt.plot(runner.mon.ts, runner.mon.n, label='n')
plt.legend()
plt.show()