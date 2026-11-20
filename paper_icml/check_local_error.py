"""Numerical check of Proposition 2 (local error of additive, Lie, implicit and Strang corrections).

Toy system du/dt = nu*L u + N(u) with L diagonal negative (a discrete Laplacian) and a nonlinear N;
the frozen network is idealised as the exact nu0 flow. Requires numpy and scipy."""
import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(0)
n = 6
k = np.arange(1, n + 1)
L = np.diag(-(k ** 2).astype(float))
D = rng.normal(size=(n, n)); D = D - D.T            # skew 'advection' matrix
c = rng.normal(size=n)
N = lambda u: -u * (D @ u) + 0.3 * np.sin(u) + c * 0.1
def Nprime(u):
    return -(np.diag(D @ u) + u[:, None] * D) + 0.3 * np.diag(np.cos(u))

nu0, nu = 0.05, 0.10
B = (nu - nu0) * L

def flow(nu_, u, h, steps=4000):
    f = lambda x: nu_ * (L @ x) + N(x)
    dt = h / steps
    for _ in range(steps):
        k1 = f(u); k2 = f(u + dt / 2 * k1); k3 = f(u + dt / 2 * k2); k4 = f(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u

u0 = rng.normal(size=n) * 0.5
pred_add = 0.5 * (Nprime(u0) @ (B @ u0) + B @ N(u0))
pred_lie = 0.5 * (Nprime(u0) @ (B @ u0) - B @ N(u0))
pred_pre = -pred_lie
pred_imp = pred_lie - 0.5 * (B @ (B @ u0))
print(" h      add/h^2 pred   Lie-post/h^2 pred   Lie-pre/h^2 pred   implicit/h^2 pred   Strang/h^3")
for h in (0.04, 0.02, 0.01, 0.005):
    true = flow(nu, u0, h)
    phi0 = flow(nu0, u0, h)
    S = lambda nu_, hh=h: expm(nu_ * hh * L)
    add = phi0 + (S(nu) - S(nu0)) @ u0
    lie = expm(h * B) @ phi0
    strang = expm(h / 2 * B) @ flow(nu0, expm(h / 2 * B) @ u0, h)
    pre = flow(nu0, expm(h * B) @ u0, h)
    imp = np.linalg.solve(np.eye(n) - h * B, phi0)
    e_add, e_lie, e_str, e_pre, e_imp = true - add, true - lie, true - strang, true - pre, true - imp
    print(f"{h:6.3f}  {np.linalg.norm(e_add)/h**2:6.3f} {np.linalg.norm(pred_add):6.3f}"
          f"   {np.linalg.norm(e_lie)/h**2:6.3f} {np.linalg.norm(pred_lie):6.3f}"
          f"   {np.linalg.norm(e_pre)/h**2:6.3f} {np.linalg.norm(pred_pre):6.3f}"
          f"   {np.linalg.norm(e_imp)/h**2:6.3f} {np.linalg.norm(pred_imp):6.3f}"
          f"   {np.linalg.norm(e_str)/h**3:6.3f}")
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
print("direction match (cosine) at h=0.005: add %.6f, Lie-post %.6f, Lie-pre %.6f, implicit %.6f"
      % (cos(e_add, pred_add), cos(e_lie, pred_lie), cos(e_pre, pred_pre), cos(e_imp, pred_imp)))
