from utils.kwd import *
from utils.welford import Welford
from datasets.cheXpert_dataset import read_dataset
from utils.visualization import *
from models.multi_label import *
from models.multi_class import model_MC_SVM
from sklearn.decomposition import PCA
import sklearn.metrics
import scipy.spatial
import scipy

PRINT_PREDICTION = False

FEATURES_N = 234
BATCH_SIZE = FEATURES_N

if __name__ == "__main__":
    if not USE_SVM:
        model = model_binaryXE_mid(use_patient_data=USE_PATIENT_DATA)
    else:
        model = model_MC_SVM(with_feature=True)

    if LOAD_WEIGHT_BOOL:
        target_model_weight, _ = get_max_acc_weight(MODELCKP_PATH)
        if target_model_weight:  # if weight is Found
            model.load_weights(target_model_weight)
        else:
            print("[Load weight] No weight is found")

    # get the dataset
    test_dataset = read_dataset(
        CHEXPERT_TEST_TARGET_TFRECORD_PATH if EVAL_CHEXPERT else CHESTXRAY_TEST_TARGET_TFRECORD_PATH,
        CHEXPERT_DATASET_PATH if EVAL_CHEXPERT else CHESTXRAY_DATASET_PATH, shuffle=False,
        use_patient_data=USE_PATIENT_DATA,
        evaluation_mode=True,
        batch_size=BATCH_SIZE)

    _test_n = TEST_N

    welford_ = Welford()
    end_res = 0
    losses1 = 0
    losses2 = 0
    losses3 = 0
    n1 = 0
    n2 = 0
    # get the ground truth labels
    maxi = -np.inf
    mini = np.inf
    avg_feature = Welford()
    for i_d, (test_img, test_label) in tqdm(enumerate(test_dataset), total=math.ceil(_test_n / BATCH_SIZE)):
        _batch_to_fill = test_img.shape[0] if not USE_PATIENT_DATA else test_img["input_img"].shape[0]
        # if _batch_to_fill != BATCH_SIZE: continue  # TODO: this is just temporary ugly fix

        # Evaluate the model on the test data using `evaluate`
        predictions = model.predict(test_img)
        features_np = predictions[1]  # without actual label
        maxi = features_np.max() if maxi < features_np.max() else maxi
        mini = features_np.min() if mini > features_np.min() else mini
        avg_feature.update(features_np.mean())
        # make the dimension smaller
        # features_np = _pca.fit_transform(features_np)

        # # normalize the features to 0...1
        # _foo = np.mean(features_np)
        # features_np = (features_np - _foo) / np.std(features_np) + _foo

        w = np.zeros((_batch_to_fill, _batch_to_fill))
        for i in tqdm(range(_batch_to_fill)):
            for j in range(i + 1, _batch_to_fill):
                # w[i, j] = kernel_wasserstein_distance(features_np[i], features_np[j], covariate=False)
                w[i, j] = scipy.stats.wasserstein_distance(features_np[i], features_np[j])
        print(w.max(), w.min())
        # run process of calculating loss
        for key in np.eye(5, dtype=np.float32):
            index = sklearn.metrics.pairwise.cosine_similarity(
                predictions[0][:, TRAIN_FIVE_CATS_INDEX] if PRINT_PREDICTION else test_label,
                key[np.newaxis, ...])  # (BATCH_SIZE, 1)
            wc = (index @ index.T) * w  # (BATCH_SIZE, BATCH_SIZE)
            wnc = ((1 - index) @ index.T) * w  # (BATCH_SIZE, BATCH_SIZE)

            loss1 = np.tanh(wc[wc != 0.])
            loss2 = np.tanh(wnc[wnc != 0.])

            n1 += loss1.size
            n2 += loss2.size
            losses1 += loss1.max()
            losses2 += loss2.min()

            losses3 += -np.log(1 - loss1.max()) - np.log(loss2.min())

    print(losses1 / n1)
    print(losses2 / n2)
    print(losses1 / n1 / (losses2 / n2))
    print(losses3 / 5)
    print("Max val in feature:", maxi, ", mini val in feature:", mini, "avg:", avg_feature)
