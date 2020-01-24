import os
import glob
import re
import numpy as np
from common_definitions import tf, THRESHOLD_SIGMOID, IMAGE_INPUT_SIZE, pos, neg
import skimage.io
import skimage.transform
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

def f1(y_true, y_pred): #taken from old keras source code
	# threshold y_pred
	y_pred = tf.cast(tf.math.greater_equal(y_pred, tf.cast(THRESHOLD_SIGMOID, tf.float32)), tf.float32)

	true_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)), 0)
	possible_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)), 0)
	predicted_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)), 0)
	precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
	recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
	f1_val = 2*(precision*recall)/(precision+recall+tf.keras.backend.epsilon())
	return f1_val

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

def read_image_and_preprocess(filename):
	img = skimage.io.imread(filename, True)
	img = skimage.transform.resize(img, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))

	return img

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
		return tf.keras.backend.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*tf.keras.backend.binary_crossentropy(y_true, y_pred), axis=-1)
	return weighted_loss

if __name__ == "__main__":
	img = read_image_and_preprocess("../sample/00002032_012.png")