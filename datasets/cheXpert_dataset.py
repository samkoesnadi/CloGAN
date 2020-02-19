"""
CheXpert
Convert valid to test, train split it to 10% validation, 90% training.
The validation should have more or less same amount of data from each diseases.

Dataset will be stored in tf.data.Dataset. Patient data and image will be combined.
"""

from common_definitions import *
from datasets.common import *
import csv
np.random.seed(666)


# all the variables
dataset_path = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets"
train_csv_file = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets/CheXpert-v1.0-small/train.csv"
valid_csv_file = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets/CheXpert-v1.0-small/valid.csv"

train_target_tfrecord_path = '../cheXpert_datasets/CheXpert_train.tfrecord'
valid_target_tfrecord_path = '../cheXpert_datasets/CheXpert_valid.tfrecord'
test_target_tfrecord_path = '../cheXpert_datasets/CheXpert_test.tfrecord'

def read_CheXpert_csv(csv_path, statistics=True):
	"""

	:param csv_path:
	:return: (keys, (image, patient data, label))
	"""

	# read train csv
	paths = []  # save path of image

	with open(csv_path, newline='') as train_csv:
		csvreader = csv.reader(train_csv, delimiter=',')

		total_row = (sum(1 for _ in csvreader)) - 1  # total for every row in csv minus one for the key
		train_csv.seek(0)

		patient_datas = np.zeros((total_row, 4))
		labels = np.zeros((total_row, 14))

		for i_row, row in enumerate(csvreader):
			if not i_row:
				# parse the keys to dict
				path_key = row[0]
				patient_data_key = row[1:5]
				labels_key = row[5:]
			else:
				# save data
				paths.append(row[0])

				patient_data = (row[1:5])
				patient_data[0] = convert_to_half_plus_one(patient_data[0], "Female", "Male")  # sex
				patient_data[1] = min(float(patient_data[1]), 100.) / 100.  # maximum age is 100 (range output is 0. to 1.)
				patient_data[2] = convert_to_half_plus_one(patient_data[2], "Frontal", "Lateral")  # f/l
				patient_data[3] = convert_to_half_plus_one(patient_data[3], "AP", "PA")  # f/l
				patient_datas[i_row-1] = patient_data

				label = row[5:]
				label = map(lambda l: 0 if l == '' else float(l), label)
				label = map(lambda l: 1 if l == -1 else l, label)  # U-Ones
				labels[i_row-1] = list(label)

	if statistics : statisticsCheXpert(labels)

	return (path_key, patient_data_key, labels_key), (paths, patient_datas, labels), total_row


if __name__ == "__main__":
	# # get valid and train set
	# (_, _, labels_key), (paths, patient_datas, labels), total_row = read_CheXpert_csv(train_csv_file)
	# valids, trains = seperate_train_valid(paths, patient_datas, labels, total_row)
	#
	# write_csv_to_tfrecord(valids, valid_target_tfrecord_path)
	# write_csv_to_tfrecord(trains, train_target_tfrecord_path)
	#
	# get test set
	# (_, _, labels_key), tests, total_row = read_CheXpert_csv(valid_csv_file)
	# write_csv_to_tfrecord(tests, test_target_tfrecord_path)

	# # reading
	# train_dataset = read_dataset(test_target_tfrecord_path, dataset_path)
	# for image_features in train_dataset.take(1):
	# 	print(image_features)

	# calculate K_SN
	train_dataset = read_dataset(CHEXPERT_TRAIN_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	for i in train_dataset.take(1):
		print(i[1].shape)
	# print(calculate_K_SN(train_dataset))

	# calculate class weight
	# train_labels = []
	# # get the ground truth labels
	# for _, train_label in tqdm(train_dataset):
	# 	train_labels.extend(train_label)
	# train_labels = np.array(train_labels)
	# print(calculating_class_weights(train_labels))