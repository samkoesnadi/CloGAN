from common_definitions import *

def calc_J(n):
    return (cp.identity(n) - 1 / n) / n ** .5


def calc_K(x: cp.ndarray, y: cp.ndarray):
    return cp.dot(x, y.T)

import numexpr as ne

def calc_k(x: cp.ndarray, y: cp.ndarray, gamma=1.0):
    x = x[np.newaxis]
    y = y[np.newaxis]
    return cp.exp(-gamma*x.T**2 + -gamma*y**2 + 2*gamma*x.T*y).sum()
    # return ne.evaluate('exp(A + B + C)', {
    #     'A' : -gamma*x.T**2,
    #     'B' : -gamma*y**2,
    #     'C' : 2*gamma*x.T*y
    # }).sum()


CALC_J = calc_J(2048)
CALC_J_m = cp.matmul(CALC_J, CALC_J.T)

def kernel_wasserstein_distance(u_values: np.ndarray, v_values: np.ndarray, covariate=True):
    # convert to cupy
    u_values = cp.array(u_values)
    v_values = cp.array(v_values)

    # n & m
    n = u_values.size
    m = v_values.size

    W_2 = calc_k(u_values, u_values) / n ** 2 - calc_k(u_values, v_values) * 2 / (n * m) \
          + calc_k(v_values, v_values) / m ** 2

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

        # prepare for this
        u_values = u_values[:, cp.newaxis]
        v_values = v_values[:, cp.newaxis]
        calc_K_u_v = calc_K(u_values, v_values)

        W_2 += cp.trace(cp.matmul(J_1_m, calc_K(u_values, u_values))) \
               + cp.trace(cp.matmul(J_2_m, calc_K(v_values, v_values))) \
               - 2 * (cp.trace(cp.matmul(cp.matmul(calc_K_u_v, J_2_m), cp.matmul(calc_K_u_v.T, J_1_m)))) ** .5

    if USE_CUPY: cp.cuda.Stream.null.synchronize()

    return W_2


if __name__ == "__main__":
    a = cp.random.normal(0, 1, 2048)
    b = cp.random.normal(1, 1., 2048)
    import time
    start_time = time.time()
    a = ([kernel_wasserstein_distance(a, b, True) for _ in range(1)])
    print("time spent:", time.time() - start_time)
    print(a)
    print(kernel_wasserstein_distance(a, b, False))
