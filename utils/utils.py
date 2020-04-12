import os
import glob
import re
import numpy as np
from common_definitions import tf, THRESHOLD_SIGMOID, IMAGE_INPUT_SIZE, K_SN, NUM_CLASSES, CLR_MAXLR, CLR_BASELR, \
    CLR_PATIENCE, TRAIN_FIVE_CATS_INDEX, EVAL_FIVE_CATS_INDEX, DISTANCE_METRIC, SELECT_INTRATER_CLASS, SELECT_CLOSER_OR_FURTHER
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from utils._auc import AUC


def pm_W(x, y=None, from_diff=True):
    if from_diff:
        norm = tf.linalg.norm(x[:, None, :] + tf.zeros(tf.shape(x)), ord=1, axis=-1)
    else:
        norm = tf.linalg.norm(x[:, None, :] - y, ord=1, axis=-1)

    return norm / 2048


class FeatureLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes=NUM_CLASSES, use_moving_average=True, alpha=.1, **kwargs):
        super().__init__(**kwargs)
        self._mean_features = tf.random.normal((num_classes, 2048))
        self._num_classes = num_classes
        self._alpha = alpha
        self._moving_average_bool = use_moving_average

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, 2048], dtype=tf.float32)))
    def call(self, _y_true, _features):
        _epsilon = tf.keras.backend.epsilon()  # epsilon of keras

        _num_classes = self._num_classes

        # run process of calculating loss
        keys = tf.eye(_num_classes)
        _indexs = tf.matmul(_y_true / (tf.norm(_y_true, axis=-1, keepdims=True) + _epsilon),
                            keys / (tf.norm(keys, axis=-1, keepdims=True) + _epsilon))  # 32 x 14
        _indexs_ones = tf.matmul(_y_true, keys)  # contains all ones, 32 x 14

        # prepare for indexs later
        indexs = tf.transpose(_indexs)[..., None]  # 14 x 32 x 1
        indexs_ones = tf.transpose(_indexs_ones)[..., None]  # 14 x 32 x 1

        _mean_features = tf.reduce_sum(indexs_ones * _features[None, ...], axis=1) / (
                tf.reduce_sum(indexs_ones, axis=1) + _epsilon)  # 14x2048
        if self._moving_average_bool:
            # compute the difference between the new mean and the old mean
            _mean_diff = _mean_features - self._mean_features

            # update the global mean features
            self._mean_features = self._mean_features + self._alpha * _mean_diff
        else:
            self._mean_features = _mean_features

        # calculate the distance matrix / heat map
        # _features = _features - tf.reduce_sum(
        #     _indexs_ones[..., None] * (_mean_features - self._mean_features)[None, ...], axis=1) / (
        #                     tf.reduce_sum(_indexs_ones[..., None], axis=1) + _epsilon)
        if SELECT_INTRATER_CLASS:
            _centered_features = _features - tf.reduce_sum(_indexs_ones[..., None] * _mean_features) / (
                                tf.reduce_sum(_indexs_ones[..., None], axis=1) + _epsilon)
            w = pm_W(_centered_features, _centered_features)  # 32x32
        else:
            _diff_features = tf.reduce_sum(
                _indexs_ones[..., None] * self._mean_features[None, ...], axis=1) / (
                                tf.reduce_sum(_indexs_ones[..., None], axis=1) + _epsilon)  # 32x2048
            w = pm_W(_diff_features, from_diff=True)  # 32x32

        # if DISTANCE_METRIC == "cosine":
        #     w = tf.keras.losses.cosine_similarity(_features[:, None, :], _features) + 1
        # else:
        #     pass

        # scale it
        w = 1. - tf.math.exp(-w)

        # calculate w loss ... (further is better)
        w_loss = - tf.math.log((w if SELECT_CLOSER_OR_FURTHER else (1. - w)) + _epsilon)

        _mask = 1. - tf.linalg.band_part(tf.ones_like(w), -1, 0)
        # w = tf.boolean_mask(w, _mask)

        _losses = 0.
        _n = 0.
        for _i in range(_num_classes):
            i_c = indexs_ones[0] @ tf.transpose(indexs[0])  # bsxbs

            # either this
            i_c = (i_c + tf.transpose(i_c)) / 2.
            # # or further reduce the value of similar but not the same classes labels
            # i_c = i_c * tf.transpose(i_c)

            i_nc = 1. - i_c

            # mask the indicator matrices
            i_c = i_c * _mask
            i_nc = i_nc * _mask

            # calculate the log loss
            _selected_indicator_matrix = i_nc if SELECT_INTRATER_CLASS else i_c
            _loss = w_loss * _selected_indicator_matrix
            _losses += tf.reduce_sum(_loss)
            _n += tf.reduce_sum(_selected_indicator_matrix)

        return _losses / (_n + _epsilon)


# @tf.function
# def _feature_loss(_y_true, _features):
#     _epsilon = tf.keras.backend.epsilon()  # epsilon of keras
#
#     # # normalize the features to 0...1 , this leads to unwanted fixed distance, because the distributions are now similar
#     # _foo = tf.math.reduce_mean(_features)
#     # _features = (_features - _foo) / tf.math.reduce_std(_features)
#
#     _num_classes = _y_true.shape[1]
#
#     # calculate the distance matrix / heat map
#     if DISTANCE_METRIC == "cosine":
#         w = tf.keras.losses.cosine_similarity(_features[:, None, :], _features) + 1
#     else:
#         w = pm_W(_features[:, None, :], _features)
#
#     _mask = 1. - tf.linalg.band_part(tf.ones_like(w), -1, 0)
#     # w = tf.boolean_mask(w, _mask)
#
#     w = w * _mask
#     # w = 1. - tf.exp(-w)  # set range to 0...1
#     # return -tf.math.log(w + _epsilon)
#
#     # wrong things because of sup and inf
#     # return tf.reduce_mean(w)
#     # w = tf.math.abs(tf.math.tanh(w))
#     # return - 2 * tf.math.log(w + _epsilon)
#
#     # run process of calculating loss
#     keys = tf.eye(_num_classes)
#     indexs = tf.transpose(tf.matmul(_y_true / tf.norm(tf.math.add(_y_true, _epsilon), axis=-1, keepdims=True),
#                                     keys / tf.norm(tf.math.add(keys, _epsilon), axis=-1, keepdims=True)))[
#         ..., None]  # 14 x 32 x 1
#
#     _losses = 0.
#     _n = 0.
#     for index in indexs:
#         index_T = tf.transpose(tf.ones_like(index))
#         index_inv = 1 - index
#
#         # index calculation
#         i_c = index @ index_T
#         i_nc = index_inv @ index_T
#         _foo = tf.ones_like(index) @ tf.transpose(index_inv)
#         # i_nc_inv = _foo + i_c
#         # i_c_inv = _foo + i_nc
#
#         wc = i_c * w  # (BATCH_SIZE, BATCH_SIZE)
#         wnc = i_nc * w  # (BATCH_SIZE, BATCH_SIZE)
#
#         # make the masked value one
#         # wc += i_c_inv
#         # wnc += i_nc_inv
#
#         # loss1 = tf.reduce_mean(wc)
#         # loss2 = tf.reduce_mean(wnc)
#
#         # scale it
#         wnc = wnc / (tf.math.reduce_max(wnc) + _epsilon)
#
#         wnc = wnc[wnc != 0.]
#
#         # calculate the log loss
#         _loss = - tf.math.log(wnc)
#
#         # tf.math.log(1. - wnc + _epsilon)
#         # _losses += tf.math.abs(tf.reduce_sum(_loss) - 2 * tf.math.log(2.))
#         _losses += tf.reduce_max(_loss)
#         _n += tf.cast(tf.size(_loss), dtype=tf.float32)
#
#     return _losses / _num_classes
#
#
# @tf.function(input_signature=(tf.TensorSpec(shape=[None, NUM_CLASSES], dtype=tf.float32),
#                               tf.TensorSpec(shape=[None, 2048], dtype=tf.float32)))
# def feature_loss(y_true, feature):
#     return _feature_loss(y_true, feature)


class AUC_five_classes(AUC):
    def __init__(self, **kwargs):
        super().__init__(num_classes=5, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(tf.gather(y_true, TRAIN_FIVE_CATS_INDEX, axis=-1),
                             tf.gather(y_pred, TRAIN_FIVE_CATS_INDEX, axis=-1), sample_weight)


def f1(y_true, y_pred):  # taken from old keras source code
    # threshold y_pred
    y_pred = tf.cast(tf.math.greater_equal(y_pred, tf.cast(THRESHOLD_SIGMOID, tf.float32)), tf.float32)

    true_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)), 0)
    possible_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)), 0)
    predicted_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)), 0)
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1_val


def custom_sigmoid(x):
    """
	This functions fit SVM case because it is close to max points on {-1,1}
	"""
    return 1 / (1 + tf.math.exp(-2 * tf.math.exp(1.) * x))


def f1_svm(y_true, y_pred):
    y_pred = custom_sigmoid(y_pred)
    return f1(y_true, y_pred)


class AUC_SVM(tf.keras.metrics.AUC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = custom_sigmoid(y_pred)
        super().update_state(y_true, y_pred, sample_weight)


def f1_mc(y_true, y_pred):
    return f1(y_true[-1, 1::2], y_pred[-1, 1::2])


class AUC_MC(tf.keras.metrics.AUC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true[-1, 1::2], y_pred[-1, 1::2], sample_weight)


def get_and_mkdir(path):
    dir_modelckp = os.path.dirname(path)

    if not os.path.exists(dir_modelckp):
        os.makedirs(dir_modelckp)
    return dir_modelckp


def get_max_acc_weight(path):
    dir_modelckp = get_and_mkdir(path)

    model_weight_files = sorted(glob.glob(dir_modelckp + "/*.hdf5"), reverse=True)

    if len(model_weight_files) == 0:
        return False, 0

    max_epoch = None
    max_acc = None
    target_weight_file = None

    # look for target weight
    for mw_file in model_weight_files:
        basename = os.path.basename(mw_file)
        epoch_acc = re.search(r"[.](.*)[-](.*)[.]hdf5", basename)
        epoch = int(epoch_acc.group(1))
        acc = float(epoch_acc.group(2))

        if max_epoch is None:
            max_epoch = epoch
            max_acc = acc
            target_weight_file = mw_file
        else:
            if acc > max_acc:
                max_epoch = epoch
                max_acc = acc
                target_weight_file = mw_file

    return target_weight_file, max_epoch


def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in tqdm(range(number_dim)):
        try:
            weights[i] = compute_class_weight('balanced', [0., 1.], y_true[:, i])
        except ValueError:
            weights[i] = np.ones(2)
    return weights


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return tf.keras.backend.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * tf.keras.backend.binary_crossentropy(y_true,
                                                                                                                 y_pred),
            axis=-1)

    return weighted_loss


from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import smart_cond


def _maybe_convert_labels(y_true):
    """Converts binary labels into -1/1."""
    are_zeros = math_ops.equal(y_true, 0)
    are_ones = math_ops.equal(y_true, 1)
    is_binary = math_ops.reduce_all(math_ops.logical_or(are_zeros, are_ones))

    def _convert_binary_labels():
        # Convert the binary labels to -1 or 1.
        return 2. * y_true - 1.

    updated_y_true = smart_cond.smart_cond(is_binary,
                                           _convert_binary_labels, lambda: y_true)
    return updated_y_true


def squared_hinge(y_true, y_pred, reduction_bool=True):
    """Computes the squared hinge loss between `y_true` and `y_pred`.
	Args:
	y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
	  If binary (0 or 1) labels are provided we will convert them to -1 or 1.
	y_pred: The predicted values.
	Returns:
	Tensor with one scalar loss entry per sample.
	"""
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = _maybe_convert_labels(y_true)

    if reduction_bool:
        return tf.keras.backend.mean(
            math_ops.square(math_ops.maximum(1. - y_true * y_pred, 0.)), axis=-1)
    else:
        return math_ops.square(math_ops.maximum(1. - y_true * y_pred, 0.))


def get_square_hinge_weighted_loss(weights):
    # Different Error Costs
    def weighted_loss(y_true, y_pred):
        return tf.keras.backend.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** y_true) * squared_hinge(y_true, y_pred,
                                                                                        reduction_bool=False), axis=-1)

    return weighted_loss


import math


# learning rate schedule
def step_decay(epoch):
    initial_lrate = CLR_MAXLR
    drop = 0.6
    epochs_drop = CLR_PATIENCE
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def _np_to_binary(np_array):
    return int("".join(str(int(x)) for x in np_array), 2)


if __name__ == "__main__":
    # img = read_image_and_preprocess("../sample/00002032_012.png")
    a = np.random.randint(0, 2, size=10)
