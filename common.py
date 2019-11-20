import numpy as np
import matplotlib.pylab as plb


def rk4(system, init, tspan):
    """
    RungeKutta4 from matplotlib.pylab for the damped oscillator
    with damping factor a.
    """

    def _rk4(derivs, y0, t):
        """
        Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
        This is a toy implementation which may be useful if you find
        yourself stranded on a system w/o scipy.  Otherwise use
        :func:`scipy.integrate`.
        From matplotlib.pylab
        Parameters
        ----------
        y0
            initial state vector
        t
            sample times
        derivs
            returns the derivative of the system and has the
            signature ``dy = derivs(ti, yi)``
        """

        try:
            Ny = len(y0)
        except TypeError:
            yout = np.zeros((len(t),), float)
        else:
            yout = np.zeros((len(t), Ny), float)

        yout[0] = y0
        i = 0

        for i in range(len(t)-1):
            thist = t[i]
            dt = t[i+1] - thist
            dt2 = dt/2.0
            y0 = yout[i]

            k1 = np.asarray(derivs(thist, y0))
            k2 = np.asarray(derivs(thist+dt2, y0 + dt2*k1))
            k3 = np.asarray(derivs(thist+dt2, y0 + dt2*k2))
            k4 = np.asarray(derivs(thist+dt, y0 + dt*k3))
            yout[i+1] = y0 + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
        return yout
    
    def derivs(t, x):
        dq = x[:len(x)//2]
        dp = -system.Vq(x[len(x)//2:], t) - system.f(t)*x[:len(x)//2]
        return tuple(np.concatenate([dp, dq]))
    
    return _rk4(derivs, init, tspan)


# From https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def pad_and_cumsum(vec):
    cumsum = np.cumsum(vec)
    return shift(cumsum, 1, 0.0)
