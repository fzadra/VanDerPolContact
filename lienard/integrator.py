import numpy as np

def D(system, dt,p,q,s,t):
    t += dt
    return p,q,s,t

def C(system, dt,p,q,s,t):
    q += s*dt
    p /= 1. + p*dt
    return p,q,s,t 

def B(system, dt,p,q,s,t):
    p -= system.Vq(q, t)*dt
    s -= system.V(q, t)*dt
    return p,q,s,t

def A(system, dt,p,q,s,t):
    f_q = system.f(q)
    fq_q = system.fq(q)
    ex = np.exp(-dt*f_q)
    p = (p + fq_q*s*dt)*ex
    s *= ex
    return p,q,s,t

def multimapintegrate(stepper, system, tspan, p0, q0, s0,j, ttol=1e-12):
    """
    Integrate [system] with initial conditions [p0], [q0]
    using the hamiltonian integrator provided in step.

    [tspan] is usually [np.linspace(t0, tfinal, num=steps)]
    """
    dt = tspan[1] - tspan[0]
    steps = len(tspan)
    init = [p0, q0]

    solpq = np.empty([steps, *np.shape(init)], dtype=np.float64)
    sols = np.empty(steps, dtype=np.float64)
    solpq[0] = np.array(init)
    sols[0] = s0

    for i in range(steps-1):
        p, q = np.copy(solpq[i])
        s = sols[i]
        t = tspan[i]
        pnew, qnew, snew, tnew = stepper(system, dt, p, q, s, t,j[i%2])
        if abs(tnew-t-dt) > ttol:
            warnings.warn(f"tnew-t-dt, dt inconsistency: {tnew-t-dt}, {dt}")
        solpq[i+1] = [pnew, qnew]
        sols[i+1] = snew

    return solpq, sols, tspan

cbabc = [(D,0.5), (C,0.5), (B,0.5), (A,1), (B,0.5), (C,0.5), (D,0.5)]
bcacb = [(D,0.5), (B,0.5), (C,0.5), (A,1), (C,0.5), (B,0.5), (D,0.5)]
abcba = [(D,0.5), (A,0.5), (B,0.5), (C,1), (B,0.5), (A,0.5), (D,0.5)]
bacab = [(D,0.5), (B,0.5), (A,0.5), (C,1), (A,0.5), (B,0.5), (D,0.5)]
cabac = [(D,0.5), (C,0.5), (A,0.5), (B,1), (A,0.5), (C,0.5), (D,0.5)]
acbca = [(D,0.5), (A,0.5), (C,0.5), (B,1), (C,0.5), (A,0.5), (D,0.5)]

mappers = [cbabc,bcacb,abcba,bacab,cabac,acbca]

def step1(system, dt, p, q, s, t, mapper=cbabc):
    for ap,coeff in mapper:
        p, q, s, t = ap(system, dt*coeff,p,q,s,t)
    return p, q, s, t

def step1b(system, dt, p, q, s, t):
    return step1(system, dt, p, q, s, t, mapper=bcacb)

# def step6(system, dt, p, q, s, t, a=ic.a_six, stepper=step1):
#     return ic.step6(system, dt, p, q, s, t, a=a, stepper=stepper)

# def step6e(system, dt, p, q, s, t, a=ic.e_six, stepper=step1):
#     return ic.step6(system, dt, p, q, s, t, a=a, stepper=stepper)