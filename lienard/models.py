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

    
def VanDerPol(epsilon, a, omega):
    
    def f(q):
        return -epsilon*(1 - q**2)
    def fq(q):
        return 2*epsilon*q
    def F(q, t):
        return q - a*np.cos(omega*t)
    def Fq(q, t):
        return 1
    
    return Lienard(f, fq, F, Fq)


def FitzHughNagumo(a, b, c, forcing=None, dforcing=None):
    forc = forcing if forcing is not None else (lambda _: 0.0)
    derf = dforcing if dforcing is not None else (lambda t: derivative(forc, t, dx=1e-10))
    
    def f(q):
        return b/c - c*(1-q**2)
    def fq(q):
        return 2.0*c*q
    def F(q,t):
        return -a + (1-b)*q + (b/3.)*q**3 - (b*forc(t) + c*derf(t))
    def Fq(q,t):
        return (1-b) + b*q**2
    
    # q <-> x, s <-> w    
    def qstoy(q,s,t=0):
        return s/c-q+(q**3)/3.-forcing(t)
    
    def xytos(x,y,t=0):
        return c *(x+y-(x**3)/3.+forcing(t))
    
    model = Lienard(f, fq, F, Fq)
    model.qstoy = qstoy
    model.xytos = xytos
    
    return model