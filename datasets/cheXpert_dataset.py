"""
CheXpert
Convert valid to test, train split it to 10% validation, 90% training.
The validation should have more or less same amount of data from each diseases.

Dataset will be stored in tf.data.Dataset. Patient data and image will be combined.
"""

from common_definitions import *
import csv
import os
import pandas as pd
np.random.seed(666)

# all the variables
dataset_path = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets"
train_csv_file = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets/CheXpert-v1.0-small/train.csv"
valid_csv_file = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/datasets/CheXpert-v1.0-small/valid.csv"

train_target_tfrecord_path = '../cheXpert_datasets/CheXpert_train.tfrecord'
valid_target_tfrecord_path = '../cheXpert_datasets/CheXpert_valid.tfrecord'
test_target_tfrecord_path = '../cheXpert_datasets/CheXpert_test.tfrecord'

VALID_RATIO = 10/100

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def load_image(img_path):
	# load image
	img = tf.io.read_file(img_path)
	img = tf.image.decode_jpeg(img, channels=1)  # output grayscale image
	img = tf.image.resize(img, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
	img /= 255.

	return img

def serialize_example(feature0, feature1, feature2):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'image_path': _bytes_feature(feature0),
      'patient_data': _float_feature(feature1),
      'label': _float_feature(feature2),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def tf_serialize_example(f0,f1,f2):
  tf_string = tf.py_function(
    serialize_example,
    (f0,f1,f2),  # pass these args to the above function.
    tf.string)      # the return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar

def convert_to_half_plus_one(variable, half_one_val, plus_one_val):
	return .5 if variable == half_one_val else 1 if variable == plus_one_val else 0

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

def statisticsCheXpert(labels, labels_key=['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']):
	totals = np.zeros((14,2))

	for i in range(14):
		label = labels[:,i]

		total_positive = label.sum()
		total_negative = (label==0).sum()
		print ("%s: pos. = %d ; neg. = %d" % (labels_key[i], total_positive, total_negative))

		totals[i] = (total_positive, total_negative)

	df = pd.DataFrame(totals, index=labels_key, columns=["pos.", "neg."])
	df.plot.bar()
	plt.show()

	return totals

def seperate_train_valid(paths, patient_datas, labels, total_row):
	paths = paths[:,np.newaxis]
	ori_trains = np.concatenate([paths,patient_datas,labels], axis=1)
	np.random.shuffle(ori_trains)  # shuffle the array

	valids = ori_trains[:int(total_row*VALID_RATIO)]
	trains = ori_trains[int(total_row*VALID_RATIO):]

	return (valids[:,0], valids[:,1:5].astype(float), valids[:,5:].astype(float)), (trains[:,0], trains[:,1:5].astype(float), trains[:,5:].astype(float))

def write_csv_to_tfrecord(data, target_path):
	paths, patient_datas, labels = data

	# create tf data Dataset
	dataset = tf.data.Dataset.from_tensor_slices((paths, patient_datas, labels))
	serialized_features_dataset = dataset.map(tf_serialize_example)

	writer = tf.data.experimental.TFRecordWriter(target_path)

	print("Start writing to %s" % target_path)
	writer.write(serialized_features_dataset)
	print("Writing successful")

def read_TFRecord(filename):
	filenames = [filename]
	raw_dataset = tf.data.TFRecordDataset(filenames)

	# Create a description of the features.
	feature_description = {
		'image_path': tf.io.FixedLenFeature([], tf.string, default_value=''),
		'patient_data': tf.io.FixedLenFeature([4], tf.float32, default_value=[0]*4),
		'label': tf.io.FixedLenFeature([14], tf.float32, default_value=[0]*14),
	}

	def _parse_function(example_proto):
		# Parse the input `tf.Example` proto using the dictionary above.
		return tf.io.parse_single_example(example_proto, feature_description)

	parsed_dataset = raw_dataset.map(_parse_function)

	return parsed_dataset

def read_dataset(filename, dataset_path, image_only=True):
	dataset = read_TFRecord(filename)
	dataset = dataset.map(lambda data: (load_image(tf.strings.join([dataset_path, '/', data["image_path"]])), data["patient_data"], data["label"]), num_parallel_calls=tf.data.experimental.AUTOTUNE)  # load the image
	dataset = dataset.map(lambda image, _, label: (image, label), num_parallel_calls=tf.data.experimental.AUTOTUNE) if image_only else dataset  # if image only throw away patient data
	dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)  # shuffle and batch with length of padding according to the the batch

	return dataset


if __name__ == "__main__":
	# # get valid and train set
	# (_, _, labels_key), (paths, patient_datas, labels), total_row = read_CheXpert_csv(train_csv_file)
	# valids, trains = seperate_train_valid(paths, patient_datas, labels, total_row)
	#
	# write_csv_to_tfrecord(valids, valid_target_tfrecord_path)
	# write_csv_to_tfrecord(trains, train_target_tfrecord_path)
	#
	# get test set
	(_, _, labels_key), tests, total_row = read_CheXpert_csv(valid_csv_file)
	# write_csv_to_tfrecord(tests, test_target_tfrecord_path)

	# # reading
	# train_dataset = read_dataset(test_target_tfrecord_path, dataset_path)
	# for image_features in train_dataset.take(1):
	# 	print(image_features)