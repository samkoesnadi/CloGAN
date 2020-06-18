"""
Three Robust Feature Mapping metrics (Cluster Centroid, Cluster Variance, and Inter-cluster Centroid)
"""

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

USE_PREDICTION = False

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
    test_dataset = read_dataset(
        CHEXPERT_TEST_TARGET_TFRECORD_PATH if EVAL_CHEXPERT else CHESTXRAY_TEST_TARGET_TFRECORD_PATH,
        CHEXPERT_DATASET_PATH if EVAL_CHEXPERT else CHESTXRAY_DATASET_PATH, shuffle=False,
        use_patient_data=USE_PATIENT_DATA,
        evaluation_mode=False,
        use_preprocess_img=True,
        drop_remainder=False,
        batch_size=CHEXPERT_TEST_N
    )

    welford_ = Welford()
    _loss_sum = Welford()
    _raw_mean_sum = Welford()
    _raw_imean_sum = Welford()
    _raw_var_sum = Welford()
    _loss_n = 0.

    _label_entropy_sum = 0.
    _label_entropy_n = 0.

    # get the ground truth labels
    maxi = -np.inf
    mini = np.inf
    avg_feature = Welford()

    _index_sd = tf.Variable(tf.zeros((TEST_N, 5)))
    featureStrength = FeatureMetric(num_classes=5, _indexs=_index_sd, _kalman_update_alpha=1)

    for i_d, (test_img, test_label) in tqdm(enumerate(test_dataset)):
        _batch_to_fill = test_img.shape[0] if not USE_PATIENT_DATA else test_img["input_img"].shape[0]

        # Evaluate the model on the test data usin  g `evaluate`
        predictions = model.call_w_features(test_img)
        features_np = predictions[1]  # without actual label
        maxi = tf.reduce_max(features_np) if maxi < tf.reduce_max(features_np) else maxi
        mini = tf.reduce_min(features_np) if mini > tf.reduce_min(features_np) else mini
        avg_feature.update(tf.reduce_mean(features_np))

        # label's entropy
        _predict_label = predictions[0].numpy() if USE_PREDICTION else test_label.numpy()
        _label_entropy = (_predict_label * tf.math.log(_predict_label + tf.keras.backend.epsilon()) +
                          (1 - _predict_label) * tf.math.log(1. - _predict_label + tf.keras.backend.epsilon())) / \
                         tf.math.log(.5)
        _label_entropy_sum += tf.reduce_sum(_label_entropy)
        _label_entropy_n += tf.cast(tf.size(_label_entropy), dtype=tf.float32)

        # calculate index
        _bs = test_label.shape[0]
        _index_sd[:_bs].assign(_predict_label[:, EVAL_FIVE_CATS_INDEX])
        # _index_sd[:_bs].assign(calc_indexs(5, test_label))

        # calculate feature strength
        raw_loss, (raw_mean_s, raw_var_s, raw_imean_s) = featureStrength(features_np)

        _loss_sum(raw_loss)
        _raw_mean_sum(raw_mean_s.numpy())
        _raw_imean_sum(raw_imean_s.numpy())
        _raw_var_sum(raw_var_s.numpy())

        # _loss_sum += tf.reduce_sum(raw_loss)
        # _raw_mean_sum += tf.reduce_sum(raw_mean_s)
        # _raw_imean_sum += tf.reduce_sum(raw_imean_s)
        # _raw_var_sum += tf.reduce_sum(raw_var_s)
        _loss_n += tf.cast(tf.size(raw_loss), dtype=tf.float32)

    print("Loss", (_loss_sum.mean) * 100)
    print("RFM", (1 - (_loss_sum.mean)) * 100)
    print("Raw mean", (_raw_mean_sum.mean) * 100)
    print("Raw imean", (_raw_imean_sum.mean) * 100)
    print("Raw var", (_raw_var_sum.mean) * 100)
    print("Label entropy", (_label_entropy_sum / _label_entropy_n).numpy())
    print("Max val in feature:", maxi, ", mini val in feature:", mini, "avg:", avg_feature)
