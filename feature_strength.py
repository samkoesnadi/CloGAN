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
    test_dataset = read_dataset(
        CHEXPERT_TEST_TARGET_TFRECORD_PATH if EVAL_CHEXPERT else CHESTXRAY_TEST_TARGET_TFRECORD_PATH,
        CHEXPERT_DATASET_PATH if EVAL_CHEXPERT else CHESTXRAY_DATASET_PATH, shuffle=False,
        use_patient_data=USE_PATIENT_DATA,
        evaluation_mode=True,
        batch_size=TEST_N)

    welford_ = Welford()
    _loss_sum = 0.
    _raw_mean_sum = 0.
    _raw_imean_sum = 0.
    _raw_var_sum = 0.
    _loss_n = 0.

    _label_entropy_sum = 0.
    _label_entropy_n = 0.

    # get the ground truth labels
    maxi = -np.inf
    mini = np.inf
    avg_feature = Welford()

    _index_sd = tf.Variable(tf.zeros((TEST_N, 5)))
    featureStrength = FeatureMetric(num_classes=5, _indexs=_index_sd)

    for i_d, (test_img, test_label) in tqdm(enumerate(test_dataset)):
        _batch_to_fill = test_img.shape[0] if not USE_PATIENT_DATA else test_img["input_img"].shape[0]

        # Evaluate the model on the test data usin  g `evaluate`
        predictions = model.predict(test_img)
        features_np = tf.reduce_mean(predictions[1], axis=[1,2])  # without actual label
        maxi = features_np.max() if maxi < features_np.max() else maxi
        mini = features_np.min() if mini > features_np.min() else mini
        avg_feature.update(features_np.mean())

        # label's entropy
        _predict_label = predictions[0]
        _label_entropy = (_predict_label * tf.math.log(_predict_label + tf.keras.backend.epsilon()) +
                          (1 - _predict_label) * tf.math.log(1. - _predict_label + tf.keras.backend.epsilon())) / \
                         tf.math.log(.5)
        _label_entropy_sum += tf.reduce_sum(_label_entropy)
        _label_entropy_n += tf.cast(tf.size(_label_entropy), dtype=tf.float32)

        # calculate index
        _bs = test_label.shape[0]
        _index_sd[:_bs].assign(calc_indexs(5, _predict_label[..., EVAL_FIVE_CATS_INDEX]))
        # _index_sd[:_bs].assign(calc_indexs(5, test_label))

        # calculate feature strength
        raw_loss, (raw_mean_s, raw_var_s, raw_imean_s) = featureStrength(tf.convert_to_tensor(features_np, dtype=tf.float32))

        _loss_sum += tf.reduce_sum(raw_loss)
        _raw_mean_sum += tf.reduce_sum(raw_mean_s)
        _raw_imean_sum += tf.reduce_sum(raw_imean_s)
        _raw_var_sum += tf.reduce_sum(raw_var_s)
        _loss_n += tf.cast(tf.size(raw_loss), dtype=tf.float32)

    print("Loss", (_loss_sum / _loss_n).numpy())
    print("Raw mean", (_raw_mean_sum / _loss_n).numpy())
    print("Raw imean", (_raw_imean_sum / _loss_n).numpy())
    print("Raw var", (_raw_var_sum / _loss_n).numpy())
    print("Label entropy", (_label_entropy_sum / _label_entropy_n).numpy())
    print("Max val in feature:", maxi, ", mini val in feature:", mini, "avg:", avg_feature)
