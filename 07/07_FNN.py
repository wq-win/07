import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

bm.set_platform('cpu')
a = 0.7
b = 0.8
tau = 12.5
Iext = 1.

print(bp.__version__)


# @bm.jit
@bp.odeint(dt=0.01)
def integral(V, w, t, Iext, a, b, tau):
    dw = (V + a - b * w) / tau
    dV = V - V * V * V / 3 - w + Iext
    return dV, dw


print(isinstance(integral, bp.ode.ODEIntegrator))
# hist_times = bm.arange(0, 100, 0.01)
# hist_V = []
# V, w = 0., 0.
# for t in hist_times:
#     V, w = integral(V, w, t, Iext, a, b, tau)
#     hist_V.append(V)
#
# plt.plot(hist_times, hist_V)
# plt.show()
runner = bp.IntegratorRunner(
    integral,
    monitors=['V'],
    inits=dict(V=0., w=0.),
    args=dict(a=a, b=b, tau=tau, Iext=Iext),
    dt=0.01
)
runner.run(100.)

plt.plot(runner.mon.ts, runner.mon.V)
plt.show()