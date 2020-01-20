# common imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# common global variables
IMAGE_INPUT_SIZE = 299  # this is because of Xception
NUM_CLASSES = 14

# for training
BUFFER_SIZE = 1600
BATCH_SIZE = 16
MAX_EPOCHS = 6
LEARNING_RATE = 1e-4

# for validation
THRESHOLD_SIGMOID = 0.5

# cheXpert dataset
CHEXPERT_TRAIN_TARGET_TFRECORD_PATH = '../cheXpert_datasets/CheXpert_train.tfrecord'
CHEXPERT_VALID_TARGET_TFRECORD_PATH = '../cheXpert_datasets/CheXpert_valid.tfrecord'
CHEXPERT_TEST_TARGET_TFRECORD_PATH = '../cheXpert_datasets/CheXpert_test.tfrecord'
CHEXPERT_DATASET_PATH = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets"

CHEXPERT_TRAIN_N = 201073
CHEXPERT_VAL_N = 22341
CHEXPERT_TEST_N = 234