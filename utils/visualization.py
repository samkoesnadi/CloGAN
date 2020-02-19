"""
GradCAM++, ROC, AUC
"""

from common_definitions import *
from keras import backend as K
import skimage
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import math
from utils.utils import *
import csv
from scipy import interp
from scipy.ndimage.interpolation import zoom

def convert_to_RGB(dz):
    norm = plt.Normalize()
    colors = plt.cm.jet(norm(dz))
    return skimage.color.rgba2rgb(colors)

def grad_cam_plus(input_model, img, layer_name, use_svm=False, use_multi_class=True):
    cams = np.zeros((NUM_CLASSES_CHEXPERT, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))

    for i in tqdm(range(NUM_CLASSES_CHEXPERT), desc="Generate tensorboard's IMAGE"):
        cls = i
        y_c = input_model.output[0, cls*2+1] if use_multi_class else input_model.output[0, cls]
        y_c = custom_sigmoid(y_c) if use_svm else y_c

        conv_output = input_model.get_layer(layer_name).output

        grads = K.gradients(y_c, conv_output)[0]

        first = tf.math.exp(y_c)*grads
        second = tf.math.exp(y_c)*grads*grads
        third = tf.math.exp(y_c)*grads*grads*grads

        gradient_function = K.function([input_model.input], [y_c,first,second,third, conv_output, grads])
        y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([img])
        global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num/alpha_denom

        weights = np.maximum(conv_first_grad[0], 0.0)

        alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)

        alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))

        deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
        #print deep_linearization_weights
        grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = zoom(cam, IMAGE_INPUT_SIZE / cam.shape[0])
        cam = cam / np.max(cam)  # scale 0 to 1.0
        # cam = resize(cam, (224,224))

        cams[i] = cam

    return cams

def smoothen(_fpr, threshold_len, extra_len):
    avg_fpr = np.copy(_fpr)
    temp_fpr = np.zeros(threshold_len + 2 * extra_len)
    temp_fpr[extra_len:threshold_len + extra_len] = _fpr
    temp_fpr[threshold_len + extra_len:] += 1

    for i_f in range(extra_len, threshold_len + extra_len):
        avg_fpr[i_f - extra_len] = np.average(temp_fpr[i_f - extra_len:i_f + extra_len + 1])

    return avg_fpr

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_best_trade_off(_fpr, _tpr, threshold_len, extra_len):
    _fpr = smoothen(_fpr, threshold_len, extra_len)
    _tpr = smoothen(_tpr, threshold_len, extra_len)

    avg_fpr = np.copy(_fpr)
    temp_fpr = np.zeros(threshold_len + 2 * extra_len)
    temp_fpr[extra_len:threshold_len + extra_len] = _fpr
    temp_fpr[threshold_len + extra_len:] += 1

    temp_tpr = np.zeros(threshold_len + 2 * extra_len)
    temp_tpr[extra_len:threshold_len + extra_len] = _tpr
    temp_tpr[threshold_len + extra_len:] += 1

    for i_f in range(extra_len, threshold_len + extra_len):
        a = math.atan2((temp_tpr[i_f] - temp_tpr[i_f - extra_len]), (temp_fpr[i_f] - temp_fpr[i_f - extra_len]))
        b = math.atan2((temp_tpr[i_f + extra_len] - temp_tpr[i_f]), (temp_fpr[i_f + extra_len] - temp_fpr[i_f]))
        avg_fpr[i_f - extra_len] = np.average([a, b])

    stable_index = np.argwhere(avg_fpr == find_nearest(avg_fpr, np.mean(avg_fpr)))[
        0, 0]  # IMPORTANT! because this determine which degree to pick

    return stable_index

def calculate_roc_auc(labels, predictions):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()

    for i in range(NUM_CLASSES_CHEXPERT):
        fpr[i], tpr[i], _thresholds = roc_curve(labels[:, i], predictions[:, i])

        tpr[i] = np.where(np.isnan(tpr[i]), 1., tpr[i])  # IMPORTANT! because nan here might better be large number

        _fpr = fpr[i]
        _tpr = tpr[i]

        threshold_len = len(_thresholds)
        extra_len = 2  # how many extra on the left and right

        stable_index = get_best_trade_off(_fpr, _tpr, threshold_len, extra_len)
        thresholds[i] = ((_thresholds[stable_index] + (_thresholds[stable_index+1] if stable_index + 1 < threshold_len else 0. )) / 2, _fpr[stable_index], _tpr[stable_index])

        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES_CHEXPERT)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES_CHEXPERT):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= NUM_CLASSES_CHEXPERT

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc, thresholds

def plot_roc(labels, predictions, **kwargs):
    fpr, tpr, roc_auc, thresholds = calculate_roc_auc(labels, predictions)

    get_and_mkdir(ROC_RESULTS_PATH)

    # print the AUC and thresholds
    with open(AUC_RESULTS_PATH, 'a', newline='') as csvfile:
        fieldnames = ['type', 'value1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'type': datetime.datetime.now().strftime("%Y%m%d-%H%M%S")})
        writer.writerow({'type': 'AUC', 'value1': str(roc_auc)})
        writer.writerow({'type': 'thresholds', 'value1': str(thresholds)})
        writer.writerow({"type": "F1", "value1": str(f1(labels, predictions).numpy())})
        writer.writerow({'type': '------------------'})

    # print all ROC plot
    for i in fpr:
        _i_toprint = i+1 if isinstance(i, int) else i
        plt.figure()
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkred',
               lw=lw, label=str(_i_toprint) + ' (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='lightblue', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic')
        plt.legend(loc='lower right')
        plt.savefig(ROC_RESULTS_PATH % str(_i_toprint))



def Xception_gradcampp(model, img, use_svm=False, use_multi_class=False):
    return grad_cam_plus(model, img, 'block14_sepconv2_act', use_svm, use_multi_class)


if __name__ == "__main__":

    train_labels = np.random.randint(2, size=(100, NUM_CLASSES_CHEXPERT))
    train_predictions_baseline = np.random.sample(size=(100, NUM_CLASSES_CHEXPERT))
    train_predictions_baseline = train_labels
    plot_roc(train_labels, train_predictions_baseline)
    # plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')