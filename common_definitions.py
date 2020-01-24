# common imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import datetime
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# common global variables
IMAGE_INPUT_SIZE = 320  # this is because of Xception
NUM_CLASSES = 14
LOAD_WEIGHT_BOOL = False
DROPOUT_N = 0.25
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()
USE_CLASS_WEIGHT = True
USE_REDUCELR = False

# for training
BUFFER_SIZE = 1600
BATCH_SIZE = 16
TOTAL_EPOCHS = 20
SUB_EPOCHS = 1  # subepoch. So the total applied epochs will be MAX_EPOCHS*SUB_EPOCHS
LEARNING_RATE = 1e-3

TENSORBOARD_LOGDIR = "./logs/kusdaNet/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# callbacks hyperparameter
REDUCELR_FACTOR = .5
REDUCELR_PATIENCE = 3
REDUCELR_MINLR = 1e-3

MODELCKP_PATH = "./checkpoints/model_weights.{epoch:02d}-{val_auc:.2f}.hdf5"  # do not change the format of basename
MODELCKP_BEST_ONLY = True


# for validation
THRESHOLD_SIGMOID = 0.5
SAMPLE_FILENAME = "./sample/00002032_012.png"

# for evaluation
ROC_RESULTS_PATH = "./report/results/ROC_%s.png"
AUC_RESULTS_PATH = "./report/results/AUC.txt"

# cheXpert dataset
CHEXPERT_TRAIN_TARGET_TFRECORD_PATH = './cheXpert_datasets/CheXpert_train.tfrecord'
CHEXPERT_VALID_TARGET_TFRECORD_PATH = './cheXpert_datasets/CheXpert_valid.tfrecord'
CHEXPERT_TEST_TARGET_TFRECORD_PATH = './cheXpert_datasets/CheXpert_test.tfrecord'
CHEXPERT_DATASET_PATH = "../datasets"

CHEXPERT_TRAIN_N = 201073
CHEXPERT_VAL_N = 22341
CHEXPERT_TEST_N = 234

	# statistics training
pos = dict()
pos[0] = 20155
pos[1] = 20857
pos[2] = 31635
pos[3] = 100087
pos[4] = 9603
pos[5] = 58583
pos[6] = 38255
pos[7] = 22401
pos[8] = 60412
pos[9] = 20324
pos[10] = 88072
pos[11] = 5536
pos[12] = 8666
pos[13] = 105344

neg = dict()
neg[0] = 180918
neg[1] = 180216
neg[2] = 169438
neg[3] = 100986
neg[4] = 191470
neg[5] = 142490
neg[6] = 162818
neg[7] = 178672
neg[8] = 140661
neg[9] = 180749
neg[10] = 113001
neg[11] = 195537
neg[12] = 192407
neg[13] = 95729


# calculation
MAX_EPOCHS = TOTAL_EPOCHS * SUB_EPOCHS
SUB_CHEXPERT_TRAIN_N = ceil(CHEXPERT_TRAIN_N / SUB_EPOCHS)

CHEXPERT_CLASS_WEIGHT = \
	np.array([[ 0.55570203,  4.98816671],
	[ 0.55786667,  4.82027617],
	[ 0.59335273 , 3.17801486],
	[ 0.99554889  ,1.00449109],
	[ 0.52507704, 10.46928043],
	[ 0.70556881 , 1.71613779],
	[ 0.6174778   ,2.62806169],
	[ 0.56268749,  4.48803625],
	[ 0.71474325 , 1.66418096],
	[ 0.55622161  ,4.94668864],
	[ 0.88969567,  1.14152625],
	[ 0.51415589 ,18.16049494],
	[ 0.52251997, 11.60125779],
	[ 1.05021989 , 0.9543638 ]])