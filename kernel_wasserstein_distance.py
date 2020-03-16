from skimage.feature import *
from skimage import data
import numpy as np
from datasets.common import *
import matplotlib.pyplot as plt
import time
import skimage.io
import skimage.transform
from scipy.stats import wasserstein_distance
import sklearn.metrics.pairwise
import math
from numpy.random import default_rng
from utils.welford import Welford


NUM_LEVELS = 128
ALL_DISTANCES = [4, 16, 32]
ALL_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # all 4 directions
PROPS_TO_MEASURE = ["dissimilarity", "homogeneity", "correlation", "energy"]
FEATURES_LEN = len(ALL_DISTANCES) * len(ALL_ANGLES) * len(PROPS_TO_MEASURE)
FEATURES_NP_FILE = "../records/chestxray14_train_input_features"
N_SAMPLES = 60


def read_resize_image(filename, num_levels):
    # img = skimage.transform.resize(
    #     skimage.io.imread(filename, as_gray=True),
    #     (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE), preserve_range=True)
    # img = skimage.io.imread(filename, as_gray=True)

    # load image
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=1)  # output grayscale image
    img = tf.image.resize(img, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))

    # process image to num of levels
    img = (img * num_levels // 256)
    return img


def calc_glcm_features(img):
    glcm = greycomatrix(img, distances=ALL_DISTANCES, angles=ALL_ANGLES, levels=NUM_LEVELS,
                        symmetric=True, normed=True)

    len_per_column = len(ALL_DISTANCES) * len(ALL_ANGLES)

    # extract features
    features_matrix = np.zeros((len(PROPS_TO_MEASURE) * len_per_column))
    for i_prop, prop in enumerate(PROPS_TO_MEASURE):
        features_matrix[i_prop * len_per_column:(i_prop + 1) * len_per_column] = greycoprops(glcm, prop).flatten()

    return features_matrix


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

    return W_2


def read_filepath_dataset(filename, dataset_path):
    dataset = read_TFRecord(filename)
    dataset = dataset.map(
        lambda data: read_resize_image(tf.strings.join([dataset_path, '/', data["image_path"]]), NUM_LEVELS),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)  # load the image

    return dataset


if __name__ == "__main__":
    # generate
    features_nps = np.zeros((TRAIN_N, FEATURES_LEN))
    for i_img, img in tqdm(enumerate(read_filepath_dataset(TRAIN_TARGET_TFRECORD_PATH, DATASET_PATH)), total=TRAIN_N):
        img = np.squeeze(img.numpy().astype(np.uint8))  # convert to numpy array
        features = calc_glcm_features(img)
        features_nps[i_img * FEATURES_LEN:(i_img + 1) * FEATURES_LEN] = features
    np.save(FEATURES_NP_FILE, features_nps)  # save it

    # load
    features_nps = np.load(FEATURES_NP_FILE + ".npy")

    # run kernel_wasserstein_distance
    rng = default_rng()
    sample_numbers = rng.choice(TRAIN_N, size=N_SAMPLES, replace=False)
    welford_ = Welford()

    for i in tqdm(sample_numbers):
        for j in range(TRAIN_N):
            if i == j: continue
            welford_(kernel_wasserstein_distance(features_nps[i], features_nps[j]))
        print(i, ":", welford_)

    print(welford_)
    print("k:", welford_.k)