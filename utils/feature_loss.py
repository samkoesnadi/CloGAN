from common_definitions import *
import sklearn.metrics
import scipy.stats
import scipy.spatial
import tqdm

def loss_1(features_np, test_label):
    _batch_to_fill = features_np.shape[0]

    w = np.zeros((_batch_to_fill, _batch_to_fill))
    for i in range(_batch_to_fill):
        for j in range(i + 1, _batch_to_fill):
            w[i, j] = scipy.stats.wasserstein_distance(features_np[i], features_np[j])
            # w[i, j] = scipy.spatial.distance.cosine(features_np[i], features_np[j])
    # print(w.max(), w.min())
    # run process of calculating loss
    for key in np.eye(14, dtype=np.float32):
        index = sklearn.metrics.pairwise.cosine_similarity(test_label, key[np.newaxis, ...])  # (BATCH_SIZE, 1)
        wc = (index @ index.T) * w  # (BATCH_SIZE, BATCH_SIZE)
        wnc = ((1 - index) @ index.T) * w  # (BATCH_SIZE, BATCH_SIZE)
        loss1 = wc[wc != 0.]
        loss2 = wnc[wnc != 0.]

    return -np.log(1 - loss1.max()) - np.log(loss2.min())


def pm_W(x, y):
    return np.linalg.norm(x - y, axis=-1)
    # return (x**2 + y**2 - 2*x[:, np.newaxis, :]*y).sum(axis=-1) ** .5

def loss_2(features_np, test_label):
    # # normalize the features to 0...1
    # _foo = np.mean(features_np)
    # features_np = (features_np - _foo) / np.std(features_np) + _foo

    _batch_to_fill = features_np.shape[0]
    num_classes = test_label.shape[1]

    w = pm_W(features_np[:, np.newaxis, :], features_np)
    # w = 1 - sklearn.metrics.pairwise.cosine_similarity(features_np, features_np)
    w = np.triu(w, k=1)

    # run process of calculating loss
    keys = np.eye(num_classes)
    indexs = ((test_label / np.linalg.norm(test_label + 1e-7, axis=-1, keepdims=True)) @
             (keys / np.linalg.norm(keys + 1e-7, axis=-1, keepdims=True))).T[..., np.newaxis]  # 14 x 32

    for index in indexs:
        wc = (index @ index.T) * w  # (BATCH_SIZE, BATCH_SIZE)
        wnc = ((1 - index) @ index.T) * w  # (BATCH_SIZE, BATCH_SIZE)
        loss1 = wc
        loss2 = wnc[wnc != 0.]

    return -np.log(1 - loss1.max() + 1e-7) - np.log(loss2.min() + 1e-7)

    # indexs = indexs.T
    # wcs = np.einsum("ij,ik->ijk", indexs, indexs, optimize=True) * w  # (14, BATCH, BATCH_SIZE)
    # wncs = np.einsum("ij,ik->ijk", (1 - indexs), indexs, optimize=True) * w  # (14, BATCH_SIZE, BATCH_SIZE)
    #
    # loss1 = wcs[wcs != 0.]
    # loss2 = wncs[wncs != 0.]
    #
    # n1 += loss1.size
    # n2 += loss2.size
    # losses1 += loss1.sum()
    # losses2 += loss2.sum()


if __name__ == "__main__":
    features_np = np.array([np.random.normal(i, 1 / (i + 1e-9), 2048) for i in range(BATCH_SIZE)])
    features_np_copy = np.copy(features_np)
    _batch_to_fill = features_np.shape[0]
    w = np.zeros((_batch_to_fill, _batch_to_fill))
    for i in range(_batch_to_fill):
        for j in range(i, _batch_to_fill):
            w[i, j] = scipy.stats.wasserstein_distance(features_np[i], features_np[j])
            # w[i, j] = scipy.spatial.distance.cosine(features_np[i], features_np[j])

    # w2 = 1 - sklearn.metrics.pairwise.cosine_similarity(features_np, features_np)
    # w2 = np.triu(w2)

    # normalize the features to 0...1
    _foo = np.mean(features_np, axis=-1, keepdims=True)
    features_np = (features_np - _foo) / np.std(features_np, axis=-1, keepdims=True) + _foo
    w2 = pm_W(features_np[:, np.newaxis, :], features_np)
    # w = 1 - sklearn.metrics.pairwise.cosine_similarity(features_np, features_np)
    w2 = np.triu(w2, k=1)

    print(np.abs(w-w2).mean())

    assert np.allclose(w, w2)

    exit()

    for i in np.arange(0, 1, 0.05):
        test_label = np.random.choice(2, size=(BATCH_SIZE, 14), p=[i, 1-i])

        x = loss_1(features_np, test_label)
        y = loss_2(features_np, test_label)

        try:
            assert np.allclose(x, y, rtol=1e-3)
        except:
            print(x,y,i)

    assert np.allclose(features_np_copy, features_np)