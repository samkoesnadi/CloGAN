from utils.welford import Welford
from datasets.cheXpert_dataset import read_dataset
from utils.visualization import *
from models.multi_label import *
from sklearn.decomposition import PCA
from utils.kwd import kernel_wasserstein_distance
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import chebyshev
from scipy.spatial.distance import cosine
from scipy.spatial.distance import mahalanobis

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

    # run the kwd
    welford_kwd = Welford()
    welford_wd = Welford()
    welford_chebysev = Welford()
    welford_cosine = Welford()

    with tqdm(total=_test_n, desc="MAIN LOOP") as t:
        for i_test in range(_test_n):
            welford_kwd(kernel_wasserstein_distance(_feature_nps_1[i_test], _feature_nps_2[i_test]))
            welford_wd(wasserstein_distance(_feature_nps_1[i_test], _feature_nps_2[i_test]))
            welford_chebysev(chebyshev(_feature_nps_1[i_test], _feature_nps_2[i_test]))
            welford_cosine(cosine(_feature_nps_1[i_test], _feature_nps_2[i_test]))

            t.set_postfix(i_=i_test, value=welford_kwd)
            t.update()

    print("kwd:", welford_kwd)
    print("wd:", welford_wd)
    print("chebysev:", welford_chebysev)
    print("cosine:", welford_cosine)

    print("--- k:", welford_kwd.k)