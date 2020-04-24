from utils.kwd import *
from utils.feature_loss import loss_2
from utils.welford import Welford
from datasets.cheXpert_dataset import read_dataset
from models.multi_label import *
from models.multi_class import model_MC_SVM
from sklearn.decomposition import PCA
import sklearn.metrics
import scipy.spatial
import scipy
from utils.utils import *

PRINT_PREDICTION = False

if __name__ == "__main__":
    if not USE_SVM:
        model = model_binaryXE_mid(use_patient_data=USE_PATIENT_DATA, use_wn=USE_WN)
    else:
        model = model_MC_SVM(with_feature=True)

    if LOAD_WEIGHT_BOOL:
        target_model_weight, _ = get_max_acc_weight(MODELCKP_PATH)
        if target_model_weight:  # if weight is Found
            model.load_weights(target_model_weight)
        else:
            print("[Load weight] No weight is found")

    # get the dataset
    _chexpert_test_dataset = read_dataset(
        CHEXPERT_TEST_TARGET_TFRECORD_PATH,
        CHEXPERT_DATASET_PATH, shuffle=False,
        use_patient_data=USE_PATIENT_DATA,
        evaluation_mode=True,
        eval_five_cats_index=[2, 5, 6, 8, 10],
        batch_size=CHEXPERT_TEST_N)

    _chestxray_test_dataset = read_dataset(
        CHESTXRAY_TEST_TARGET_TFRECORD_PATH,
        CHESTXRAY_DATASET_PATH, shuffle=False,
        use_patient_data=USE_PATIENT_DATA,
        evaluation_mode=True,
        eval_five_cats_index=[1, 9, 8, 0, 2],
        batch_size=CHESTXRAY_TEST_N)

    _loss_sum = 0.
    _raw_mean_sum = 0.
    _raw_var_sum = 0.
    _loss_n = 0.

    _chexpert_index_sd = tf.Variable(tf.zeros((CHEXPERT_TEST_N, 5)))
    _chestxray_index_sd = tf.Variable(tf.zeros((CHESTXRAY_TEST_N, 5)))
    chexpert_featureStrength = FeatureStrength(num_classes=5, _indexs=_chexpert_index_sd, _kalman_update_alpha=0.2)
    chestxray_featureStrength = FeatureStrength(num_classes=5, _indexs=_chestxray_index_sd, _kalman_update_alpha=0.2)

    for i_d, (test_img, test_label) in tqdm(enumerate(_chexpert_test_dataset)):
        _batch_to_fill = test_img.shape[0] if not USE_PATIENT_DATA else test_img["input_img"].shape[0]

        # Evaluate the model on the test data usin  g `evaluate`
        predictions = model.predict(test_img)
        features_np = predictions[1]  # without actual label

        # calculate index
        _bs = test_label.shape[0]
        _chexpert_index_sd[:_bs].assign(calc_indexs(5, predictions[0][..., [2, 5, 6, 8, 10]]))
        # _chexpert_index_sd[:_bs].assign(calc_indexs(5, test_label))

        # calculate feature strength
        chexpert_featureStrength(tf.convert_to_tensor(features_np, dtype=tf.float32))

    for i_d, (test_img, test_label) in tqdm(enumerate(_chestxray_test_dataset)):
        _batch_to_fill = test_img.shape[0] if not USE_PATIENT_DATA else test_img["input_img"].shape[0]

        # Evaluate the model on the test data usin  g `evaluate`
        predictions = model.predict(test_img)
        features_np = predictions[1]  # without actual label

        # calculate index
        _bs = test_label.shape[0]
        # _chestxray_index_sd[:_bs].assign(calc_indexs(5, predictions[0][..., [1, 9, 8, 0, 2]]))
        _chestxray_index_sd[:_bs].assign(calc_indexs(5, test_label))

        # calculate feature strength
        chestxray_featureStrength(tf.convert_to_tensor(features_np, dtype=tf.float32))


    # l2-dist means inter dataset
    mean_strength = tf.linalg.norm(chexpert_featureStrength.features_mean - chestxray_featureStrength.features_mean
                                   , ord=2, axis=-1)

    # l2-dist vars inter dataset
    var_strength = tf.linalg.norm(chexpert_featureStrength.features_var - chestxray_featureStrength.features_var,
                                  ord=2, axis=-1)

    print("Mean strength", mean_strength)
    print("Var strength", var_strength)
    print("avgs Mean strength", tf.reduce_mean(mean_strength))
    print("avgs Var strength", tf.reduce_mean(var_strength))
