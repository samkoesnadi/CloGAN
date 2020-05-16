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
from models.gan import GANModel

FEATURE_LAYER_NAME = 1

if __name__ == "__main__":
    if USE_SVM:
        model = model_MC_SVM()
    elif USE_DOM_ADAP_NET:
        model = GANModel()
        # to initiate the graph
        model.call_w_features(tf.zeros((1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1)))
    else:
        model = model_binaryXE(use_patient_data=USE_PATIENT_DATA)
        model.call_w_features = model.call

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
        drop_remainder=False)

    _chestxray_test_dataset = read_dataset(
        CHESTXRAY_TEST_TARGET_TFRECORD_PATH,
        CHESTXRAY_DATASET_PATH, shuffle=False,
        use_patient_data=USE_PATIENT_DATA,
        evaluation_mode=True,
        eval_five_cats_index=[1, 9, 8, 0, 2],
        drop_remainder=False)

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
        predictions = model.call_w_features(test_img)
        # features_np = tf.reduce_mean(predictions[1], axis=[1,2])  # without actual label

        # calculate index
        _bs = test_label.shape[0]
        _chexpert_index_sd[:_bs].assign(5, predictions[0].numpy()[..., [2, 5, 6, 8, 10]])
        # _chexpert_index_sd[:_bs].assign(calc_indexs(5, test_label))

        # calculate feature strength
        chexpert_featureStrength(predictions[FEATURE_LAYER_NAME])

    for i_d, (test_img, test_label) in tqdm(enumerate(_chestxray_test_dataset)):
        _batch_to_fill = test_img.shape[0] if not USE_PATIENT_DATA else test_img["input_img"].shape[0]

        # Evaluate the model on the test data usin  g `evaluate`
        predictions = model.call_w_features(test_img)
        # features_np = tf.reduce_mean(predictions[1], axis=[1,2])  # without actual label

        # calculate index
        _bs = test_label.shape[0]
        _chestxray_index_sd[:_bs].assign(5, predictions[0].numpy()[..., [1, 9, 8, 0, 2]])
        # _chestxray_index_sd[:_bs].assign(calc_indexs(5, test_label))

        # calculate feature strength
        chestxray_featureStrength(predictions[FEATURE_LAYER_NAME])


    # l2-dist means inter dataset
    mean_strength = tf.linalg.norm(chexpert_featureStrength.features_mean - chestxray_featureStrength.features_mean
                                   , ord=2, axis=-1) / 2048**.5

    # l2-dist vars inter dataset
    var_strength = tf.linalg.norm(chexpert_featureStrength.features_var - chestxray_featureStrength.features_var,
                                  ord=2, axis=-1) / 2048**.5

    print("Inter-dataset-mean", mean_strength)
    print("avgs Inter-dataset-Mean", tf.reduce_mean(mean_strength))

    print("Inter-dataset-Variance", var_strength)
    print("avgs Inter-dataset-Variance", tf.reduce_mean(var_strength))
