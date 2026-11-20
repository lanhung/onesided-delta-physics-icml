"""Check that unsuccessful diagnostic iterations report their returned residual."""
import numpy as np
from spectral_transfer_numpy import PeriodicLaplacian, affine_picard_cached, input_recast_picard


def run():
    grid = PeriodicLaplacian((16,), (1.,))
    u = np.random.default_rng(1).normal(size=(1,16))
    h, nu0, delta_nu = .02, .01, .01
    predict = lambda v: grid.heat(v, h*nu0)
    b = predict(u)
    y, diagnostic = affine_picard_cached(b, grid, h, delta_nu, max_iter=0)
    residual = np.linalg.norm(b+h*delta_nu*grid.laplacian(y)-y)/np.linalg.norm(b)
    assert diagnostic['iterations'] == 0 and not diagnostic['converged']
    assert abs(diagnostic['residual']-residual) < 1e-14
    y, diagnostic = input_recast_picard(predict, u, grid, h, delta_nu, max_iter=1)
    residual = np.linalg.norm(predict(u+h*delta_nu*grid.laplacian(y))-y)/np.linalg.norm(b)
    assert not diagnostic['converged']
    assert abs(diagnostic['residual']-residual) < 1e-14
    try:
        input_recast_picard(predict, u, grid, h, delta_nu, max_iter=0)
    except ValueError:
        pass
    else:
        raise AssertionError('The invalid iteration bound was not rejected.')
    print('Iteration-failure residual checks passed.')


if __name__ == '__main__':
    run()
