from numpy.random import default_rng
from utils.welford import Welford
from datasets.cheXpert_dataset import read_dataset
from utils.visualization import *
from models.multi_label import *
from sklearn.decomposition import PCA
from utils.kwd import *

GENERATE_FEATURE = False
PROCESS_DIMRED = False

N_SAMPLES = 60
FEATURES_N = 64


if __name__ == "__main__":
    test_dataset = read_dataset(
        CHEXPERT_TEST_TARGET_TFRECORD_PATH if EVAL_CHEXPERT else CHESTXRAY_TEST_TARGET_TFRECORD_PATH,
        CHEXPERT_DATASET_PATH if EVAL_CHEXPERT else CHESTXRAY_DATASET_PATH, evaluation_mode=True)

    _test_n = CHEXPERT_TEST_N if EVAL_CHEXPERT else CHESTXRAY_TEST_N

    # run chexpert first
    print("run chexpert first")
    model = model_binaryXE_mid()
    model.load_weights("networks/chexpert.hdf5")

    _feature_nps_1 = np.zeros((_test_n, 2048))
    for i_test, (input, label) in tqdm(enumerate(test_dataset)):
        _feature_nps_1[i_test * BATCH_SIZE : (i_test+1) * BATCH_SIZE] = model.predict(input)[1]

    # now run chestxray14
    print("now run chestxray14")
    model.load_weights("networks/chestxray14.hdf5")

    _feature_nps_2 = np.zeros((_test_n, 2048))
    for i_test, (input, label) in tqdm(enumerate(test_dataset)):
        _feature_nps_2[i_test * BATCH_SIZE : (i_test+1) * BATCH_SIZE] = model.predict(input)[1]

    # swap np with cp
    USE_CUPY = True
    try:
        import cupy as np
    except ImportError as e:
        USE_CUPY = False

    # run the kwd
    welford_ = Welford()
    with tqdm(total=_test_n, desc="MAIN LOOP") as t:
        for i_test in range(_test_n):
            welford_(kernel_wasserstein_distance(_feature_nps_1[i_test], _feature_nps_2[i_test], USE_CUPY))
            t.set_postfix(i_=i_test, value=welford_)
            t.update()

    print(welford_)
    print("k:", welford_.k)