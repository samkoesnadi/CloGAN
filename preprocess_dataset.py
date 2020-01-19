"""
Convert valid to test, train split it to 10% validation, 90% training.
The validation should have more or less same amount of data from each diseases.
"""

from common_definitions import *

dataset_path = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets"
train_file = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets/CheXpert-v1.0-small/train.csv"
valid_file = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets/CheXpert-v1.0-small/valid.csv"

