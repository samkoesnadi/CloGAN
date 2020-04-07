# from common_definitions import *
import scipy.stats
from utils.utils import feature_loss
import numpy as np

if __name__ == "__main__":
    a = np.arange(32*2048).reshape((32, 2048))
    b = a[:, None, :]

    print((a-b).shape)
    print(a-b)
    print((a-b)[0], (a-b)[1])

    a = np.array([0,0,1])[np.newaxis]
    print(a.T - a)
    # features_np = np.array([np.random.normal(i, 1 / (i + 1e-9), 2048) for i in range(32)])
    #
    # for i in np.arange(0, 1, 0.05):
    #     test_label = np.random.choice(2, size=(32, 14), p=[i, 1 - i]).astype(np.float32)
    #
    #     x = feature_loss(tf.convert_to_tensor(test_label, dtype=tf.float32), tf.convert_to_tensor(features_np, dtype=tf.float32))
    #     print(x)