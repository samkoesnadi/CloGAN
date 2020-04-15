import os
import glob
import re
import numpy as np
from common_definitions import tf, THRESHOLD_SIGMOID, IMAGE_INPUT_SIZE, K_SN, NUM_CLASSES, CLR_MAXLR, CLR_BASELR, \
    CLR_PATIENCE, TRAIN_FIVE_CATS_INDEX, EVAL_FIVE_CATS_INDEX, DISTANCE_METRIC, LABELS_COUPLE_INDEX, BATCH_SIZE
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from utils._auc import AUC


def pm_W(x, y=None, from_diff=True):
    if from_diff:
        norm = tf.linalg.norm(x[:, None, :] + tf.zeros(tf.shape(x)), ord=1, axis=-1)
    else:
        norm = tf.linalg.norm(x[:, None, :] - y, ord=1, axis=-1)

    return norm / 2048


def allclose(x, y, rtol=1e-5, atol=1e-8):
    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)


def _calc_indexs(_num_classes, _y_true, _epsilon=tf.keras.backend.epsilon()):
    # run process of calculating loss
    keys = tf.eye(_num_classes)
    _indexs = tf.matmul(_y_true / (tf.norm(_y_true, axis=-1, keepdims=True) + _epsilon),
                        keys / (tf.norm(keys, axis=-1, keepdims=True) + _epsilon))  # 32 x 14
    _indexs_ones = tf.matmul(_y_true, keys)  # contains all ones, 32 x 14

    return _indexs, _indexs_ones


class myBinaryXE(tf.keras.losses.BinaryCrossentropy):
    def __init__(self, num_classes=NUM_CLASSES, bs=BATCH_SIZE, *args, **kwargs):
        super(myBinaryXE, self).__init__(*args, **kwargs)
        self.num_classes = num_classes

        # for SD. TODO: check if the data actually transferred
        self.indexs = tf.Variable(tf.zeros((bs, num_classes)), dtype=tf.float32)
        self.indexs_ones = tf.Variable(tf.zeros((bs, num_classes)), dtype=tf.float32)

    def call(self, y_true, y_pred):
        _bs = tf.shape(y_true)[0]

        _indexs, _indexs_ones = _calc_indexs(self.num_classes, y_true)

        self.indexs[0:_bs].assign(_indexs)
        self.indexs_ones[0:_bs].assign(_indexs_ones)

        return super(myBinaryXE, self).call(y_true, y_pred)


class FeatureLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes=NUM_CLASSES, use_moving_average=True, net_model=None, alpha=.1,
                 _feloss_alpha=(1, 1, 1), _index_var=None, _index_ones_var=None, **kwargs):
        super().__init__(**kwargs)
        if _feloss_alpha is None:
            _feloss_alpha = [1., 1., 1.]
        self._mean_features = tf.random.normal((num_classes, 2048))
        self._num_classes = num_classes
        self._alpha = alpha
        self._moving_average_bool = use_moving_average

        # for td feLoss, lets make it zero shot
        self.model = net_model
        self._feloss_alpha = _feloss_alpha
        self._feloss_alpha_ones = list(map(lambda x: 1. if x != 0 else 0., self._feloss_alpha))
        self._td_mean_features = tf.random.normal((num_classes, 2048))

        # for sd
        self._indexs = _index_var
        self._indexs_ones = _index_ones_var

        self._epsilon = tf.keras.backend.epsilon()  # epsilon of keras

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None],
                                                dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, 2048], dtype=tf.float32)))
    def call(self, _td_x, _features):
        _epsilon = self._epsilon
        _num_classes = self._num_classes
        _bs = tf.shape(_features)[0]

        # changed the _td_x shape
        _td_x = tf.reshape(_td_x, (_bs, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1))

        # with tf.compat.v1.variable_scope('model', reuse=True):
        _td_y_pred, _td_features = self.model(_td_x)

        # calculate the indexs
        _indexs = self._indexs[0:_bs]
        _indexs_ones = self._indexs_ones[0:_bs]
        _td_indexs, _td_indexs_ones = _calc_indexs(_num_classes, _td_y_pred)

        # have the transpose one
        indexs = tf.transpose(_indexs)[..., None]  # 14 x 32 x 1
        indexs_ones = tf.transpose(_indexs_ones[0:_bs])[..., None]  # 14 x 32 x 1

        td_indexs = tf.transpose(_td_indexs)[..., None]  # 14 x 32 x 1
        td_indexs_ones = tf.transpose(_td_indexs_ones)[..., None]  # 14 x 32 x 1

        _mean_features = (indexs_ones[..., 0] @ _features) / (
                tf.reduce_sum(indexs_ones, axis=1) + _epsilon)  # 14x2048
        _td_mean_features = (td_indexs_ones[..., 0] @ _td_features) / (
                tf.reduce_sum(td_indexs_ones, axis=1) + _epsilon)  # 7x2048

        if self._moving_average_bool:
            # compute the difference between the new mean and the old mean
            _mean_diff = _mean_features - self._mean_features
            _td_mean_diff = _td_mean_features - self._td_mean_features

            # update the global mean features
            self._mean_features = self._mean_features + self._alpha * _mean_diff
            self._td_mean_features = self._td_mean_features + self._alpha * _td_mean_diff
        else:
            self._mean_features = _mean_features
            self._td_mean_features = _td_mean_features

        # calculate the distance matrix / heat map
        _diff_features = tf.reduce_sum(
            _indexs_ones[..., None] * self._mean_features[None, ...], axis=1) / (
                                 tf.reduce_sum(_indexs_ones[..., None], axis=1) + _epsilon)  # 32x2048

        _centered_features = _features - _diff_features

        _td_diff_features = tf.reduce_sum(
            _td_indexs_ones[..., None] * self._td_mean_features[None, ...], axis=1) / (
                                    tf.reduce_sum(_td_indexs_ones[..., None], axis=1) + _epsilon)  # 32x2048

        w_inter = pm_W(_diff_features, from_diff=True)  # 32x32
        w_intra = pm_W(_centered_features, _centered_features)  # 32x32
        w_td = pm_W(_diff_features, _td_diff_features, from_diff=False)  # 32x32

        # if DISTANCE_METRIC == "cosine":
        #     w = tf.keras.losses.cosine_similarity(_features[:, None, :], _features) + 1
        # else:
        #     pass

        # scale it
        w_inter = 1. - tf.math.exp(-w_inter)
        w_intra = 1. - tf.math.exp(-w_intra)
        w_td = 1. - tf.math.exp(-w_td)

        # calculate w loss ... (further is better)
        w_inter_loss = - tf.math.log(1. - w_inter + _epsilon)
        w_intra_loss = - tf.math.log(w_intra + _epsilon)
        w_td_loss = - tf.math.log(1. - w_td + _epsilon)

        _mask = 1. - tf.linalg.band_part(tf.ones_like(w_inter_loss), -1, 0)

        _losses = 0.
        _n = 0.
        for _i in range(_num_classes):
            # for intra
            i_c = indexs_ones[_i] @ tf.transpose(indexs[_i])  # bsxbs
            i_c = (i_c + tf.transpose(i_c)) / 2.

            # for inter
            i_nc = 1. - i_c

            # for td
            _i_c_td = td_indexs_ones[_i] @ tf.transpose(indexs[_i])  # bsxbs
            _i_c_td = _i_c_td + tf.transpose(_i_c_td)  # bsxbs
            _i_c2_td = indexs_ones[_i] @ tf.transpose(td_indexs[_i])  # bsxbs
            _i_c2_td = _i_c2_td + tf.transpose(_i_c2_td)  # bsxbs

            i_c_td = (_i_c_td + _i_c2_td) / 2

            # mask the indicator matrices

            i_c_td = i_c_td * _mask
            i_c = i_c * _mask
            i_nc = i_nc * _mask

            # calculate the log loss
            _loss = self._feloss_alpha[2] * w_td_loss * i_c_td + \
                    self._feloss_alpha[1] * w_inter_loss * i_nc + \
                    self._feloss_alpha[0] * w_intra_loss * i_c
            _losses += tf.reduce_sum(_loss)
            _n += tf.reduce_sum(self._feloss_alpha_ones[2] * i_c_td + \
                                self._feloss_alpha_ones[1] * i_nc + \
                                self._feloss_alpha_ones[0] * i_c)

        return _losses / (_n + _epsilon)


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
