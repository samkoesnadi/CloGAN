from numpy.random import default_rng
from utils.welford import Welford
from datasets.cheXpert_dataset import read_dataset
from utils.visualization import *
from models.multi_label import *
from sklearn.decomposition import PCA

# swap np with cp
USE_CUPY = True
try:
    import cupy as np
except ImportError as e:
    USE_CUPY = False

N_SAMPLES = 60
FEATURES_N = 64

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



if __name__ == "__main__":
    features_nps_1 = np.load(FEATURES_NP_FILE_1 + "_dimred.npy")
    features_nps_2 = np.load(FEATURES_NP_FILE_2 + "_dimred.npy")

    # run kernel_wasserstein_distance
    rng = default_rng()
    sample_numbers = np.array(rng.choice(TRAIN_N, size=N_SAMPLES, replace=False))
    features_nps_1 = np.array(features_nps_1)  # because of cupy conversion
    features_nps_2 = np.array(features_nps_2)  # because of cupy conversion
    welford_ = Welford()

    with tqdm(total=N_SAMPLES, desc="MAIN LOOP",
              postfix=[int(0), dict(value=0)]) as t:
        for i in sample_numbers:
            for j in tqdm(range(CHESTXRAY_TRAIN_N), desc="iter for TRAIN_N"):
                if i == j: continue
                welford_(kernel_wasserstein_distance(features_nps_1[i], features_nps_2[j]))
            t.set_postfix(i_=i, value=welford_)
            t.update()

    print(welford_)
    print("k:", welford_.k)