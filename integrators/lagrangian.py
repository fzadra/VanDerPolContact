import numpy as np
from scipy.optimize import fsolve


def forced_lagrangian(system, tspan, p0, q0):
    """
    Integrate [system] with initial conditions [p0], [q0]
    using a second order forced lagrangian integrator.

    [tspan] is usually [np.linspace(t0, tfinal, num=steps)]
    """
    dt = tspan[1] - tspan[0]
    steps = len(tspan)
    init = [p0, q0]

    solpq = np.empty([steps, *np.shape(init)], dtype=np.float64)
    solpq[0] = np.array(init)

    for i, t in enumerate(tspan[1:]):
        p, q = np.copy(solpq[i])

        qnew = fsolve(
            lambda qnew: p + system.d1l(q, qnew, dt) + system.fm(q, qnew, t, dt),
            q + dt * p,
        )
        pnew = system.d2l(q, qnew, dt) + system.fp(q, qnew, t, dt)

        solpq[i + 1] = [pnew, qnew]

    return solpq, tspan
