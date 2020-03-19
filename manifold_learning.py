from sklearn.manifold import Isomap, TSNE
from common_definitions import *
import time
from utils.visualization import *
from datasets.cheXpert_dataset import read_dataset
from models.multi_label import model_binaryXE_mid

def _np_to_binary(np_array):
    return int("".join(str(int(x)) for x in np_array), 2)


if __name__ == "__main__":
    model = model_binaryXE_mid()
    model.load_weights("networks/chexpert.hdf5" if TRAIN_CHEXPERT else "networks/chestxray14.hdf5")

    test_dataset = read_dataset(
        CHEXPERT_TEST_TARGET_TFRECORD_PATH if EVAL_CHEXPERT else CHESTXRAY_TEST_TARGET_TFRECORD_PATH,
        CHEXPERT_DATASET_PATH if EVAL_CHEXPERT else CHESTXRAY_DATASET_PATH, evaluation_mode=True)

    _test_n = CHEXPERT_TEST_N if EVAL_CHEXPERT else CHESTXRAY_TEST_N

    _color_label = np.zeros(_test_n)
    _feature_nps = np.zeros((_test_n, 2048))
    for i_test, (input, label) in tqdm(enumerate(test_dataset)):
        label = label.numpy()
        label = list(map(_np_to_binary, label))
        _color_label[i_test * BATCH_SIZE : (i_test+1) * BATCH_SIZE] = label
        _feature_nps[i_test * BATCH_SIZE : (i_test+1) * BATCH_SIZE] = model.predict(input)[1]

    start_time = time.time()
    embedding = TSNE(n_components=2, init='pca', random_state=0, n_jobs=-1)
    X_embedded = embedding.fit_transform(_feature_nps)
    print("time spent for manifold learning:", time.time()-start_time)

    # sketch it
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=_color_label, cmap=plt.cm.Spectral)
    plt.axis('tight')

    plt.savefig("report/results/manifold_learning.png", bbox_inches="tight")