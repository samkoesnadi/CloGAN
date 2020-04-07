from utils.kwd import *
from utils.feature_loss import loss_2
from utils.utils import _feature_loss
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
        batch_size=TEST_N)
    # test_dataset = read_dataset(TRAIN_TARGET_TFRECORD_PATH, DATASET_PATH, use_patient_data=USE_PATIENT_DATA, use_feature_loss=False, evaluation_mode=True, batch_size=BATCH_SIZE).take(100)

    welford_ = Welford()
    losses3 = 0
    n = 0
    # get the ground truth labels
    maxi = -np.inf
    mini = np.inf
    avg_feature = Welford()
    for i_d, (test_img, test_label) in tqdm(enumerate(test_dataset)):
        _batch_to_fill = test_img.shape[0] if not USE_PATIENT_DATA else test_img["input_img"].shape[0]
        # if _batch_to_fill != BATCH_SIZE: continue  # TODO: this is just temporary ugly fix

        # Evaluate the model on the test data usin  g `evaluate`
        predictions = model.predict(test_img)
        features_np = predictions[1]  # without actual label
        maxi = features_np.max() if maxi < features_np.max() else maxi
        mini = features_np.min() if mini > features_np.min() else mini
        avg_feature.update(features_np.mean())
        # make the dimension smaller
        # features_np = _pca.fit_transform(features_np)

        # calculate
        loss = _feature_loss(test_label, features_np)
        losses3 += loss
        n += 1

    print(losses3 / n)
    print("Max val in feature:", maxi, ", mini val in feature:", mini, "avg:", avg_feature)
