from utils.welford import Welford
from datasets.cheXpert_dataset import read_dataset
from utils.visualization import *
from models.multi_label import *
from models.multi_class import model_MC_SVM
from sklearn.decomposition import PCA
from utils.kwd import *

GENERATE_FEATURE = True
PROCESS_DIMRED = True  # make it smaller dimension

N_SAMPLES = 60
FEATURES_N = 64

if __name__ == "__main__":
    if GENERATE_FEATURE:
        if not USE_SVM:
            model = model_binaryXE_mid(use_patient_data=USE_PATIENT_DATA)
            model.load_weights(MODEL_CHEXPERT_PATH if EVAL_CHEXPERT else MODEL_CHESTXRAY_PATH)
        else:
            model = model_MC_SVM(with_feature=True)
            model.load_weights(MODEL_SVM_PATH)

        # get the dataset
        test_dataset = read_dataset(
            CHEXPERT_TEST_TARGET_TFRECORD_PATH if EVAL_CHEXPERT else CHESTXRAY_TEST_TARGET_TFRECORD_PATH,
            CHEXPERT_DATASET_PATH if EVAL_CHEXPERT else CHESTXRAY_DATASET_PATH, shuffle=False,
            use_patient_data=USE_PATIENT_DATA)

        _test_n = TEST_N

        features_nps = np.zeros((_test_n, 2048))
        # get the ground truth labels
        for i_d, (test_img, _) in tqdm(enumerate(test_dataset), total=math.ceil(_test_n / BATCH_SIZE)):
            _batch_to_fill = test_img.shape[0] if not USE_PATIENT_DATA else test_img["input_img"].shape[0]

            # Evaluate the model on the test data using `evaluate`
            features_nps[i_d*BATCH_SIZE: i_d*BATCH_SIZE + _batch_to_fill] = model.predict(test_img)[1]

        np.save(FEATURES_NP_FILE, features_nps)  # save it

    if PROCESS_DIMRED:  # process dimred
        # load
        features_nps = np.load(FEATURES_NP_FILE + ".npy")

        print("### Running PCA... ###")
        _pca = PCA(n_components=FEATURES_N)

        features_nps = _pca.fit_transform()
        np.save(FEATURES_NP_FILE + "_dimred", features_nps)
    else:
        features_nps = np.load(FEATURES_NP_FILE + "_dimred.npy")

    # run kernel_wasserstein_distance
    sample_numbers = np.array(np.random.choice(TEST_N, size=N_SAMPLES, replace=False))
    features_nps = np.array(features_nps)  # because of cupy conversion
    welford_ = Welford()

    with tqdm(total=N_SAMPLES, desc="MAIN LOOP",
              postfix=[int(0), dict(value=0)]) as t:
        for i in sample_numbers:
            for j in tqdm(range(TEST_N), desc="iter for TEST_N"):
                if i == j: continue
                welford_(kernel_wasserstein_distance(features_nps[i], features_nps[j]))
            t.set_postfix(i_=i, value=welford_)
            t.update()

    print(welford_)
    print("k:", welford_.k)