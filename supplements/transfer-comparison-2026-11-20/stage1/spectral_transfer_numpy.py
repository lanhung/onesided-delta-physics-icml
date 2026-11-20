"""NumPy reference implementations for periodic, constant-coefficient diffusion.

All arrays and predictor inputs/outputs are in PHYSICAL units. Spatial axes are
the last len(lengths) axes. Leading axes may be batches or channels. The frozen
predictor must return the full next state, not a normalized increment.

This is a CPU mathematical reference, not an optimized GPU benchmarking backend.
No checkpoint, training dataset, or manuscript neural result is included.
"""

from dataclasses import dataclass
import numpy as np

METHODS = ("frozen", "delta", "explicit", "lie_post", "lie_pre", "strang", "implicit")


@dataclass(frozen=True)
class PeriodicLaplacian:
    shape: tuple
    lengths: tuple

    def __post_init__(self):
        if len(self.shape) != len(self.lengths) or not self.shape:
            raise ValueError("One positive physical length is required per spatial axis.")
        if any(n < 2 for n in self.shape) or any(l <= 0 for l in self.lengths):
            raise ValueError("Invalid periodic grid.")
        k = [2 * np.pi * np.fft.fftfreq(n, d=l / n)
             for n, l in zip(self.shape, self.lengths)]
        k2 = sum(v * v for v in np.meshgrid(*k, indexing="ij", sparse=True))
        object.__setattr__(self, "k2", k2)
        object.__setattr__(self, "axes", tuple(range(-len(self.shape), 0)))

    def apply(self, u, multiplier):
        u = np.asarray(u)
        if tuple(u.shape[-len(self.shape):]) != tuple(self.shape):
            raise ValueError("State spatial dimensions do not match the grid.")
        if not np.issubdtype(u.dtype, np.floating):
            raise TypeError("Reference implementation expects real floating point fields.")
        out = np.fft.ifftn(np.fft.fftn(u, axes=self.axes) * multiplier, axes=self.axes)
        # All multipliers used here are real and even, preserving Hermitian symmetry.
        return out.real.astype(u.dtype, copy=False)

    def laplacian(self, u):
        return self.apply(u, -self.k2)

    def heat(self, u, amount):
        if amount < 0:
            raise ValueError("Negative heat time is excluded from the main comparison.")
        if amount == 0:
            return u.copy()
        return self.apply(u, np.exp(-amount * self.k2))


class TransferStepper:
    """One frozen predictor evaluation for every method in METHODS.

    delta_one_sided=False permits the bounded additive propagator difference for
    lower positive viscosity in diagnostic tests. Splitting/implicit/explicit
    below the reference viscosity are deliberately outside this main protocol.
    Any such diagnostic needs a separate, explicitly documented stability policy.
    """

    def __init__(self, predict, grid, h, nu0, nu, method, delta_one_sided=True):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        if h <= 0 or nu0 < 0 or nu < 0:
            raise ValueError("Require h>0 and nonnegative viscosities.")
        if nu < nu0 and method not in ("frozen", "delta"):
            raise ValueError("Main comparison requires nu >= nu0.")
        self.predict, self.grid, self.h = predict, grid, float(h)
        self.nu0, self.nu, self.method = float(nu0), float(nu), method
        self.delta_one_sided = delta_one_sided
        self.calls = 0
        z = h * (nu - nu0) * grid.k2
        if nu >= nu0:
            self.delta_symbol = np.exp(-h * nu0 * grid.k2) * np.expm1(-z)
        else:
            # Avoid 0 * inf in a two-sided diagnostic on a very fine grid.
            self.delta_symbol = -np.exp(-h * nu * grid.k2) * np.expm1(z)
        self.explicit_symbol = -z
        if nu >= nu0:
            self.extra_symbol = np.exp(-z)
            self.half_symbol = np.exp(-z / 2)
            self.implicit_symbol = 1 / (1 + z)

    def _predict(self, u):
        self.calls += 1
        y = np.asarray(self.predict(u))
        if y.shape != u.shape:
            raise ValueError("predict must return a full physical state of unchanged shape.")
        return y

    def __call__(self, u):
        u = np.asarray(u)
        if self.method == "frozen" or self.nu == self.nu0:
            return self._predict(u)
        if self.method == "delta":
            y = self._predict(u)
            if self.delta_one_sided and self.nu < self.nu0:
                return y
            return y + self.grid.apply(u, self.delta_symbol)
        if self.method == "explicit":
            return self._predict(u) + self.grid.apply(u, self.explicit_symbol)
        if self.method == "lie_post":
            return self.grid.apply(self._predict(u), self.extra_symbol)
        if self.method == "lie_pre":
            return self._predict(self.grid.apply(u, self.extra_symbol))
        if self.method == "strang":
            return self.grid.apply(self._predict(self.grid.apply(u, self.half_symbol)),
                                   self.half_symbol)
        return self.grid.apply(self._predict(u), self.implicit_symbol)


def affine_picard_cached(b, grid, h, delta_nu, omega=1.0, tol=1e-12, max_iter=2000):
    """Diagnostic solve of y=b+h*delta_nu*Delta(y), with b already cached.

    There are ZERO network evaluations inside this routine. Relaxation is not
    needed for the closed-form implicit baseline; it is used only to verify the
    algebraic equivalence where the iteration is contractive. Residual is the
    equation residual divided by ||b||, not iterate-to-iterate relative change.
    """
    if max_iter < 0 or tol <= 0:
        raise ValueError("Require max_iter>=0 and tol>0.")
    y = b.copy()
    denom = max(np.linalg.norm(b.ravel()), np.finfo(b.dtype).tiny)
    for iteration in range(max_iter + 1):
        residual = b + h * delta_nu * grid.laplacian(y) - y
        relative = float(np.linalg.norm(residual.ravel()) / denom)
        if relative <= tol:
            return y, {"converged": True, "iterations": iteration, "residual": relative}
        if not np.isfinite(relative) or iteration == max_iter:
            break
        y = y + omega * residual
    return y, {"converged": False, "iterations": iteration, "residual": relative}


def input_recast_picard(predict, u, grid, h, delta_nu, tol=1e-12, max_iter=200):
    """Diagnostic for v2's INPUT-corrected fixed-point EQUATION.

    This is plain Picard, NOT a reproduction of the paper's vector-Aitken code.
    An evaluation at y verifies its residual; every such evaluation is counted.
    Return the state whose residual was actually tested.
    """
    if max_iter < 1 or tol <= 0:
        raise ValueError("Require max_iter>=1 and tol>0.")
    y = np.asarray(predict(u))
    calls = 1
    if delta_nu == 0:
        return y, {"converged": True, "iterations": 0, "calls": calls, "residual": 0.0}
    denom = max(np.linalg.norm(y.ravel()), np.finfo(y.dtype).tiny)
    for iteration in range(1, max_iter + 1):
        g = np.asarray(predict(u + h * delta_nu * grid.laplacian(y)))
        calls += 1
        relative = float(np.linalg.norm((g - y).ravel()) / denom)
        if relative <= tol:
            return y, {"converged": True, "iterations": iteration, "calls": calls,
                       "residual": relative}
        if not np.isfinite(relative) or iteration == max_iter:
            break
        y = g
    return y, {"converged": False, "iterations": iteration, "calls": calls,
               "residual": relative}
