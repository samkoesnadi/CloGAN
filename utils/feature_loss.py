from utils.kwd import kernel_wasserstein_distance
from common_definitions import *
import sklearn.metrics
import scipy.stats
import tqdm

_batch_to_fill = BATCH_SIZE


def loss_1(features_np, test_label):
    losses1 = 0
    losses2 = 0
    n1 = 0
    n2 = 0

    w = np.zeros((_batch_to_fill, _batch_to_fill))
    for i in range(_batch_to_fill):
        for j in range(i + 1, _batch_to_fill):
            w[i, j] = scipy.stats.wasserstein_distance(features_np[i], features_np[j])

    # run process of calculating loss
    for key in np.eye(14, dtype=np.float32):
        index = sklearn.metrics.pairwise.cosine_similarity(test_label, key[np.newaxis, ...])  # (BATCH_SIZE, 1)
        wc = (index @ index.T) * w  # (BATCH_SIZE, BATCH_SIZE)
        wnc = ((1 - index) @ index.T) * w  # (BATCH_SIZE, BATCH_SIZE)
        loss1 = wc[wc != 0.]
        loss2 = wnc[wnc != 0.]

        n1 += loss1.size
        n2 += loss2.size
        losses1 += loss1.sum()
        losses2 += loss2.sum()

    return losses1 / n1 / (losses2 / n2)


def pm_W(x, y):
    return np.linalg.norm(x - y, axis=-1)
    # return (x**2 + y**2 - 2*x[:, np.newaxis, :]*y).sum(axis=-1) ** .5

def loss_2(features_np, test_label, num_classes=NUM_CLASSES):
    losses1 = 0
    losses2 = 0
    n1 = 0
    n2 = 0

    # normalize the features to 0...1
    _foo = np.mean(features_np)
    features_np = (features_np - _foo) / np.std(features_np) + _foo

    _batch_to_fill = features_np.shape[0]

    w = np.triu(pm_W(features_np[:, np.newaxis, :], features_np), k=1)

    # run process of calculating loss
    keys = np.identity(num_classes)
    indexs = (test_label / np.linalg.norm(test_label + tf.keras.backend.epsilon(), axis=-1, keepdims=True)) @ \
             (keys / np.linalg.norm(keys + tf.keras.backend.epsilon(), axis=-1, keepdims=True))  # 14 x 32

    # for index in indexs:
    #     wc = (index @ index.T) * w  # (BATCH_SIZE, BATCH_SIZE)
    #     wnc = ((1 - index) @ index.T) * w  # (BATCH_SIZE, BATCH_SIZE)
    #
    #     loss1 = wc[wc != 0.]
    #     loss2 = wnc[wnc != 0.]
    #
    #     n1 += loss1.size
    #     n2 += loss2.size
    #     losses1 += loss1.sum()
    #     losses2 += loss2.sum()

    indexs = indexs.T
    wcs = np.einsum("ij,ik->ijk", indexs, indexs, optimize=True) * w  # (14, BATCH, BATCH_SIZE)
    wncs = np.einsum("ij,ik->ijk", (1 - indexs), indexs, optimize=True) * w  # (14, BATCH_SIZE, BATCH_SIZE)

    loss1 = wcs[wcs != 0.]
    loss2 = wncs[wncs != 0.]

    n1 += loss1.size
    n2 += loss2.size
    losses1 += loss1.sum()
    losses2 += loss2.sum()

    return losses1 / n1 / (losses2 / n2)


if __name__ == "__main__":
    features_np = np.array([np.random.normal(i, 1 / (i + 1e-9), 2048) for i in range(BATCH_SIZE)])

    for i in np.arange(0, 1, 0.05):
        test_label = np.random.choice(2, size=(BATCH_SIZE, 14), p=[i, 1-i])

        x = loss_1(features_np, test_label)
        y = loss_2(features_np, test_label)

        try:
            assert np.allclose(x, y, rtol=1e-3)
        except:
            print(x,y,i)