from common_definitions import *

# swap np with cp
USE_CUPY = True
try:
    import cupy as np
except ImportError as e:
    USE_CUPY = False

def calc_J(n):
    arrow_1 = np.ones(n)
    s = arrow_1[np.newaxis, :].T / n
    J = (np.identity(n) - s * arrow_1) / n ** .5
    return J


def calc_K(x: np.ndarray, y: np.ndarray):
    return x[np.newaxis, :].T * y


def calc_k(x: np.ndarray, y: np.ndarray, gamma=None):
    x_num_features = x.shape[-1]
    y_num_features = y.shape[-1]

    if gamma is None:
        gamma = 1.0

    x = np.repeat(x.T[:, np.newaxis], y_num_features, 1)
    y = np.repeat(y[np.newaxis, :], x_num_features, 0)
    _sum = np.exp(-gamma * (x - y) ** 2).sum()
    return _sum


def kernel_wasserstein_distance(u_values: np.ndarray, v_values: np.ndarray):
    # n & m
    n = u_values.size
    m = v_values.size

    J_1 = calc_J(n)
    J_2 = calc_J(m)

    W_2 = calc_k(u_values, u_values) / n ** 2 - calc_k(u_values, v_values) * 2 / (n * m) \
          + calc_k(v_values, v_values) / m ** 2 + np.trace(J_1 @ J_1.T @ calc_K(u_values, u_values)) \
          + np.trace(J_2 @ J_2.T @ calc_K(v_values, v_values)) \
          - 2 * np.trace(calc_K(u_values, v_values) @ J_2 @ J_2.T @ calc_K(v_values, u_values) @ J_1 @ J_1.T) ** .5

    if USE_CUPY: np.cuda.Stream.null.synchronize()

    return float(W_2)