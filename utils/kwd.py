import numpy as np

# swap cp with cp
USE_CUPY = True

if USE_CUPY:
    try:
        import cupy as cp
    except ImportError as e:
        USE_CUPY = False
else:
    cp = np

def calc_J(n):
    arrow_1 = cp.ones(n)
    s = arrow_1[cp.newaxis, :].T / n
    J = (cp.identity(n) - s * arrow_1) / n ** .5
    return J


def calc_K(x: cp.ndarray, y: cp.ndarray):
    return x[cp.newaxis, :].T * y


def calc_k(x: cp.ndarray, y: cp.ndarray, gamma=None):
    x_num_features = x.shape[-1]
    y_num_features = y.shape[-1]

    if gamma is None:
        gamma = 1.0

    x = cp.repeat(x.T[..., cp.newaxis], y_num_features, 1)
    y = cp.repeat(y[..., cp.newaxis, :], x_num_features, 0)
    _sum = cp.exp(-gamma * (x - y) ** 2).sum()
    return _sum


def kernel_wasserstein_distance(u_values: np.ndarray, v_values: np.ndarray):
    # convert to cupy
    u_values = cp.array(u_values)
    v_values = cp.array(v_values)

    # n & m
    n = u_values.size
    m = v_values.size

    J_1 = calc_J(n)
    J_2 = calc_J(m)

    W_2 = calc_k(u_values, u_values) / n ** 2 - calc_k(u_values, v_values) * 2 / (n * m) \
          + calc_k(v_values, v_values) / m ** 2 + cp.trace(J_1 @ J_1.T @ calc_K(u_values, u_values)) \
          + cp.trace(J_2 @ J_2.T @ calc_K(v_values, v_values)) \
          - 2 * cp.trace(calc_K(u_values, v_values) @ J_2 @ J_2.T @ calc_K(v_values, u_values) @ J_1 @ J_1.T) ** .5

    if USE_CUPY: cp.cuda.Stream.null.synchronize()

    return float(W_2)

if __name__ == "__main__":
    a = cp.random.normal(0, 1, 2048)
    b = cp.random.normal(10, 1., 2048)
    print(kernel_wasserstein_distance(a, b))