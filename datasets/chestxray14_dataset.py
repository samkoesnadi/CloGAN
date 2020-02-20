"""
ChestXray14 API
"""

from common_definitions import *
from datasets.common import *
import csv

# all the variables
dataset_path = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets/chestXray14/images"
train_csv_file = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets/chestXray14/train_val_list.txt"
test_csv_file = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets/chestXray14/test_list.txt"
data_csv_file = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets/chestXray14/Data_Entry_2017.csv"

train_target_tfrecord_path = 'cheXray14_datasets/CheXray14_train.tfrecord'
valid_target_tfrecord_path = 'cheXray14_datasets/CheXray14_valid.tfrecord'
test_target_tfrecord_path = 'cheXray14_datasets/CheXray14_test.tfrecord'


def read_CheXray14_csv(csv_path, statistics=True):
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
		labels = np.zeros((total_row, NUM_CLASSES))

		# parse the keys to dict
		path_key = "Image Index"
		patient_data_key = ["Patient Gender", "Patient Age", "Frontal/Lateral", "AP/PA"]
		labels_key = CHESTXRAY_LABELS_KEY

		next(csvreader)  # to remove the header
		for i_row, row in enumerate(csvreader):
			image_index = row[0]
			label = row[1].split("|")
			gender = row[5]
			age = row[4]
			view_position = row[6]

			# save data
			paths.append(image_index)

			patient_data = []
			patient_data.append(convert_to_half_plus_one(gender, "F", "M"))  # sex
			patient_data.append(min(float(age), 100.) / 100.)  # maximum age is 100 (range output is 0. to 1.)
			patient_data.append(0.)  # f/l
			patient_data.append(convert_to_half_plus_one(view_position, "AP", "PA"))  # f/l
			patient_datas[i_row-1] = patient_data

			temp = np.zeros(NUM_CLASSES)

			try: label.remove("No Finding")
			except: pass # remove no finding

			temp[list(map(lambda x: labels_key.index(x), label))] = 1.
			labels[i_row] = temp

	if statistics : statisticsCheXpert(labels, NUM_CLASSES)

	return (path_key, patient_data_key, labels_key), (paths, patient_datas, labels), total_row

if __name__ == "__main__":
	# save to TFRecord
	with open(train_csv_file, "r") as f:  # train csv
		train_paths = [line.strip() for line in f]

	with open(test_csv_file, "r") as f:  # test csv
		test_paths = [line.strip() for line in f]

	(_, _, labels_key), (paths, patient_datas, labels), total_row = read_CheXray14_csv(data_csv_file, statistics=False)

	dict_patient_datas = dict(zip(paths, patient_datas))
	dict_labels = dict(zip(paths, labels))

	def get_patient_datas(paths):
		return np.array([dict_patient_datas[t_p] for t_p in paths])

	def get_labels(paths):
		return np.array([dict_labels[t_p] for t_p in paths])

	# write TFrecords valids and trains
	valids, trains = seperate_train_valid(np.array(train_paths), get_patient_datas(train_paths), get_labels(train_paths), len(train_paths))

	write_csv_to_tfrecord(valids, valid_target_tfrecord_path)
	write_csv_to_tfrecord(trains, train_target_tfrecord_path)

	# write TFrecords tests
	write_csv_to_tfrecord((np.array(test_paths), get_patient_datas(test_paths), get_labels(test_paths)), test_target_tfrecord_path)

	train_dataset = read_dataset(train_target_tfrecord_path, dataset_path, num_class=NUM_CLASSES)
	for i in train_dataset.take(1):
		print(i)