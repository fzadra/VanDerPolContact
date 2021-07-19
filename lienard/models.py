import numpy as np
from scipy.misc import derivative


class Lienard:
    def __init__(self, f, fq, F, Fq):
        self.f = f
        self.fq = fq
        self.F = F
        self.Fq = Fq

    def f(self, q):
        return self.f(q)

    def fq(self, q):
        return self.fq(q)

    def V(self, q, t):
        return self.F(q, t)

    def Vq(self, q, t):
        return self.Fq(q, t)


def VanDerPol(epsilon, a=0, omega=0):
    def f(q):
        return -epsilon * (1 - q ** 2)

    def fq(q):
        return 2 * epsilon * q

    def F(q, t):
        return q - a * np.cos(omega * t)

    def Fq(q, t):
        return 1

    return Lienard(f, fq, F, Fq)


class VanDerPolLag:
    def __init__(self, epsilon, a=0, omega=0):
        self.epsilon = epsilon
        self.a = a
        self.omega = omega

    def d1l(self, q0, q1, dt):
        return -dt * (q0 + q1) / 4 - (q1 - q0) / dt

    def d2l(self, q0, q1, dt):
        return -dt * (q0 + q1) / 4 + (q1 - q0) / dt

    def fm(self, q0, q1, t, dt):
        return self.epsilon / 2 * (q1 - q0) * (
            1 - (q0 + q1) ** 2 / 4
        ) + dt / 2 * self.a * np.cos(self.omega * t)

    def fp(self, q0, q1, t, dt):
        return self.epsilon / 2 * (q1 - q0) * (1 - (q0 + q1) ** 2 / 4)


class VanDerPolLag0:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def d1l(self, q0, q1, dt):
        return -dt * q0 - (q1 - q0) / dt

    def d2l(self, q0, q1, dt):
        return (q1 - q0) / dt

    def fm(self, q0, q1, t, dt):
        return self.epsilon * (q1 - q0) * (1 - q0 ** 2) + dt * self.a * np.cos(
            self.omega * t
        )

    def fp(self, q0, q1, t, dt):
        return 0


def FitzHughNagumo(a, b, c, forcing=None, dforcing=None):
    forc = forcing if forcing is not None else (lambda _: 0.0)
    derf = (
        dforcing if dforcing is not None else (lambda t: derivative(forc, t, dx=1e-10))
    )

    def f(q):
        return b / c - c * (1 - q ** 2)

    def fq(q):
        return 2.0 * c * q

    def F(q, t):
        return -a + (1 - b) * q + (b / 3.0) * q ** 3 - (b * forc(t) + c * derf(t))

    def Fq(q, t):
        return (1 - b) + b * q ** 2

    # q <-> x, s <-> w
    def qstoy(q, s, t=0):
        return s / c - q + (q ** 3) / 3.0 - forcing(t)

    def xytos(x, y, t=0):
        return c * (x + y - (x ** 3) / 3.0 + forcing(t))

    model = Lienard(f, fq, F, Fq)
    model.qstoy = qstoy
    model.xytos = xytos

    return model
