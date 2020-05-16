import numpy as np
import tensorflow as tf

# swap cp with cp
USE_CUPY = False

if USE_CUPY:
    try:
        import cupy as cp
        tf.config.set_visible_devices([], 'GPU')
    except ImportError as e:
        USE_CUPY = False
else:
    cp = np

from common_definitions import *

def calc_J(n):
    return (cp.identity(n) - 1 / n) / n ** .5


def calc_K(x: cp.ndarray, y: cp.ndarray, gamma=1.0):
    # return cp.dot(x.T, y)
    return cp.exp(-gamma*x.T**2 + -gamma*y**2 + 2*gamma*x.T*y)

import numexpr as ne

# def calc_k(x: cp.ndarray, y: cp.ndarray, gamma=1.0):
#     return cp.exp(-gamma*x.T**2 + -gamma*y**2 + 2*gamma*x.T*y).sum()
#     # return ne.evaluate('exp(A + B + C)', {
#     #     'A' : -gamma*x.T**2,
#     #     'B' : -gamma*y**2,
#     #     'C' : 2*gamma*x.T*y
#     # }).sum()


CALC_J = calc_J(2048)
CALC_J_m = cp.matmul(CALC_J, CALC_J.T)

def kernel_wasserstein_distance(u_values: np.ndarray, v_values: np.ndarray, covariate=True):
    # convert to cupy
    u_values = cp.array(u_values)
    v_values = cp.array(v_values)

    # n & m
    n = u_values.size
    m = v_values.size

    # prepare for this
    u_values = u_values[cp.newaxis]
    v_values = v_values[cp.newaxis]

    calc_K_u_v = calc_K(u_values, v_values)
    calc_K_u_u = calc_K(u_values, u_values)
    calc_K_v_v = calc_K(v_values, v_values)

    W_2 = calc_K_u_u.sum() / n ** 2 - calc_K_u_v.sum() * 2 / (n * m) \
          + calc_K_v_v.sum() / m ** 2

    if covariate:
        if n == 2048 and m == 2048:
            # pre-calculated
            J_1_m = CALC_J_m
            J_2_m = CALC_J_m
        else:
            J_1 = calc_J(n)
            J_2 = calc_J(m)
            J_1_m = cp.matmul(J_1, J_1.T)
            J_2_m = cp.matmul(J_2, J_2.T)

        W_2 += cp.trace(cp.matmul(J_1_m, calc_K_u_u)) \
               + cp.trace(cp.matmul(J_2_m, calc_K_v_v)) \
               - 2 * (cp.trace(cp.matmul(cp.matmul(calc_K_u_v, J_2_m), cp.matmul(calc_K_u_v.T, J_1_m)))) ** .5

    if USE_CUPY: cp.cuda.Stream.null.synchronize()

    return W_2

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

if __name__ == "__main__":
    a = cp.random.normal(10, .005, 2048)
    b = cp.random.normal(0, 1, 2048)
    print(np.var(a), np.mean(a))
    _foo = np.mean(a)
    a = (a - _foo) / np.std(a, axis=-1, keepdims=True) + _foo
    print(np.var(a), np.mean(a))

    import time
    import scipy.stats
    start_time = time.time()
    x = ([kernel_wasserstein_distance(a, b, True) for _ in range(1)])
    print("time spent:", time.time() - start_time)
    print(x)
    start_time = time.time()
    print(kernel_wasserstein_distance(a, b, False))
    print("time spent:", time.time() - start_time)
    print(scipy.stats.wasserstein_distance(a, b))
    print(scipy.spatial.distance.cosine(a, b))
    print(np.linalg.norm(a-b)**2)