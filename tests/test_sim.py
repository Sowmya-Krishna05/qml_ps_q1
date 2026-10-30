"""Independent checks of vqc/sim.py (dense-matrix reference built with kron)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vqc.sim import StatevectorSim, DensityMatrixSim

I2 = np.eye(2); X_ = np.array([[0,1],[1,0]]); Y_ = np.array([[0,-1j],[1j,0]]); Z_ = np.diag([1.,-1])
def R(P, a): return np.cos(a/2)*I2 - 1j*np.sin(a/2)*P
def one(U, q, n): return np.kron(np.kron(np.eye(2**q), U), np.eye(2**(n-q-1)))
def cx(c, t, n):
    M = np.zeros((2**n, 2**n))
    for i in range(2**n):
        bits = [(i >> (n-1-k)) & 1 for k in range(n)]
        if bits[c]: bits[t] ^= 1
        j = sum(b << (n-1-k) for k, b in enumerate(bits)); M[j, i] = 1
    return M

def reference(ops, x, th, n, scale=1.0):
    psi = np.zeros(2**n, complex); psi[0] = 1
    P = {"rx": X_, "ry": Y_, "rz": Z_}
    for g, qs, ref in ops:
        if g == "cx": psi = cx(*qs, n) @ psi
        else:
            a = scale*x[ref[1]] if ref[0]=="x" else th[ref[1]]
            psi = one(R(P[g], a), qs[0], n) @ psi
    return psi

def test_all():
    rng = np.random.default_rng(0); n = 4
    ops = [("ry",(0,),("x",0)),("rz",(1,),("x",1)),("rx",(2,),("t",0)),("ry",(3,),("t",1)),
           ("cx",(0,1),None),("cx",(3,0),None),("cx",(2,1),None),("ry",(1,),("t",2)),("cx",(1,3),None),
           ("rx",(0,),("x",2)),("cx",(2,3),None)]
    X = rng.normal(size=(5, 3)); th = rng.normal(size=3)
    sv, dm = StatevectorSim(n), DensityMatrixSim(n)
    s = sv.run(ops, X, th); rho = dm.run(ops, X, th, noisy=False)
    zs = [Z_ if q in (0,2) else I2 for q in range(n)]
    Zop = np.kron(np.kron(np.kron(zs[0], zs[1]), zs[2]), zs[3])
    for b in range(5):
        ref = reference(ops, X[b], th, n)
        assert np.allclose(s[b].reshape(-1), ref, atol=1e-12)
        r = rho[b].reshape(2**n, 2**n)
        assert np.allclose(r, np.outer(ref, ref.conj()), atol=1e-12)
        assert np.isclose(sv.expz(s, (0,2))[b], (ref.conj() @ Zop @ ref).real)
    # noise: |0> -> single rx(0) gate -> depolarising lambda=0.01 -> <Z> = 1 - lambda ; readout flips 0.96
    dm1 = DensityMatrixSim(1); r1 = dm1.run([("rx",(0,),("t",0))], np.zeros((1,1)), np.array([0.0]))
    assert np.isclose(dm1.expz(r1,(0,),readout=False)[0], 1-0.01)
    assert np.isclose(dm1.expz(r1,(0,))[0], (1-0.01)*0.96)
    # trace preserved and 2q depolarising lambda: Bell-ish state
    ops2 = [("ry",(0,),("t",0)),("cx",(0,1),None)]
    d2 = DensityMatrixSim(2); r2 = d2.run(ops2, np.zeros((1,1)), np.array([np.pi/2]))
    assert np.isclose(np.trace(r2[0].reshape(4,4)).real, 1.0)
    z0z1 = d2.expz(r2,(0,1),readout=False)[0]
    assert np.isclose(z0z1, 1-0.05, atol=1e-9)   # Z1 is untouched by the 1q error; CX error scales it by 1-lambda2
    print("ZZ after ry,cx with noise:", z0z1)
    print("simulator tests passed")
if __name__ == "__main__": test_all()
