from common_definitions import *
import scipy.spatial
from utils.utils import FeatureLoss
import numpy as np

if __name__ == "__main__":
    print(1. - scipy.spatial.distance.cosine(np.array([0,0,0,1,0]), np.array([0,0,1,1,0])))
    print(1. - scipy.spatial.distance.cosine(np.array([0,0,0,1,0]), np.array([0,0,0,1,0])))
    print(1. - scipy.spatial.distance.cosine(np.array([0,0,0,1,0]), np.array([0,1,1,1,0])))
    print(1. - scipy.spatial.distance.cosine(np.array([0,0,0,1,0]), np.array([1,1,1,1,1])))
    print(1. - scipy.spatial.distance.cosine(np.array([0,0,0,1,0]), np.array([1,0,1,0,0])))

    # exit()
    # a = np.arange(32*2048).reshape((32, 2048))
    # b = a[:, None, :]
    #
    # print((a-b).shape)
    # print(a-b)
    # print((a-b)[0], (a-b)[1])
    #
    # a = np.array([0,0,1])[np.newaxis]
    # print(a.T - a)
    #
    # features_np_0 = np.random.normal(0, 1, 2048)
    # features_np_1 = np.random.normal(0, 1, 2048)
    feature_loss = FeatureLoss()
    for i in np.arange(0, 1, 0.05):
        features_np = np.array([np.random.normal(i, 0, 2048) for i in range(1, 33)])
        test_label = np.random.choice(2, size=(32, 14), p=[i, 1 - i]).astype(np.float32)

        x = feature_loss(tf.convert_to_tensor(test_label, dtype=tf.float32), tf.convert_to_tensor(features_np, dtype=tf.float32))
        # x = scipy.spatial.distance.cosine(features_np_0, features_np_1)
        print(x)