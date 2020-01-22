# common imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import datetime
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# common global variables
IMAGE_INPUT_SIZE = 299  # this is because of Xception
NUM_CLASSES = 14
LOAD_WEIGHT_BOOL = True
DROPOUT_N = 0.2
KERNEL_INITIALIZER = tf.keras.initializers.glorot_uniform()
USE_CLASS_WEIGHT = True

# for training
BUFFER_SIZE = 1600
BATCH_SIZE = 16
TOTAL_EPOCHS = 10
SUB_EPOCHS = 2  # subepoch. So the total applied epochs will be MAX_EPOCHS*SUB_EPOCHS
LEARNING_RATE = 1e-2

TENSORBOARD_LOGDIR = "../logs/kusdaNet/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# callbacks hyperparameter
REDUCELR_FACTOR = .2
REDUCELR_PATIENCE = 2
REDUCELR_MINLR = 1e-5

MODELCKP_PATH = "../checkpoints/model_weights.{epoch:02d}-{val_f1:.2f}.hdf5"  # do not change the format of basename
MODELCKP_BEST_ONLY = True


# for validation
THRESHOLD_SIGMOID = 0.5
SAMPLE_FILENAME = "../sample/00002032_012.png"

# for evaluation
ROC_RESULTS_PATH = "../report/results/ROC_%s.png"
AUC_RESULTS_PATH = "../report/results/AUC.txt"

# cheXpert dataset
CHEXPERT_TRAIN_TARGET_TFRECORD_PATH = '../cheXpert_datasets/CheXpert_train.tfrecord'
CHEXPERT_VALID_TARGET_TFRECORD_PATH = '../cheXpert_datasets/CheXpert_valid.tfrecord'
CHEXPERT_TEST_TARGET_TFRECORD_PATH = '../cheXpert_datasets/CheXpert_test.tfrecord'
CHEXPERT_DATASET_PATH = "../../datasets"

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