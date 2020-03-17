import os
import pandas as pd
from common_definitions import *
from tqdm import tqdm
import skimage.io
import skimage.transform

def statisticsCheXpert(labels, num_class=14, labels_key=LABELS_KEY):
	totals = np.zeros((num_class,2))

	for i in range(num_class):
		label = labels[:,i]

		total_positive = label.sum()
		total_negative = (label==0).sum()
		print ("%s: pos. = %d ; neg. = %d" % (labels_key[i], total_positive, total_negative))

		totals[i] = (total_positive, total_negative)

	df = pd.DataFrame(totals, index=labels_key, columns=["pos.", "neg."])
	df.plot.bar()
	plt.show()

	return totals


def convert_to_half_plus_one(variable, half_one_val, plus_one_val):
	return .5 if variable == half_one_val else 1 if variable == plus_one_val else 0


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

def sparsity_norm(a):
	mask = tf.where(a > 0., 1., 0.)
	norm_m = tf.reduce_sum(mask)
	return K_SN * a / norm_m

def seperate_train_valid(paths, patient_datas, labels, total_row):
	paths = paths[:,np.newaxis]
	ori_trains = np.concatenate([paths,patient_datas,labels], axis=1)
	np.random.shuffle(ori_trains)  # shuffle the array

	valids = ori_trains[:int(total_row*VALID_RATIO)]
	trains = ori_trains[int(total_row*VALID_RATIO):]

	return (valids[:,0], valids[:,1:5].astype(float), valids[:,5:].astype(float)), (trains[:,0], trains[:,1:5].astype(float), trains[:,5:].astype(float))

def calculate_K_SN(train_dataset):
	sum_norm = 0.
	n_mask = 0.
	for img, _ in tqdm(train_dataset):
		mask = tf.where(img > 0., 1., 0.)
		norm_m = tf.reduce_sum(mask)

		sum_norm += norm_m
		n_mask += 1.
	return sum_norm / (n_mask * BATCH_SIZE)

def load_image(img_path):
	# load image
	img = tf.io.read_file(img_path)
	img = tf.image.decode_jpeg(img, channels=1)  # output grayscale image
	img = tf.image.resize(img, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
	img /= 255.

	# sparsity normalization
	img = sparsity_norm(img) if USE_SPARSITY_NORM and K_SN != 1. else img

	return img

def read_image_and_preprocess(filename, use_sn=False):
	img = skimage.io.imread(filename, True)
	img = skimage.transform.resize(img, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))

	# sparsity normalization
	img = sparsity_norm(img) if use_sn and K_SN != 1. else img

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


def write_csv_to_tfrecord(data, target_path):
	paths, patient_datas, labels = data

	# create tf data Dataset
	dataset = tf.data.Dataset.from_tensor_slices((paths, patient_datas, labels))
	serialized_features_dataset = dataset.map(tf_serialize_example)

	writer = tf.data.experimental.TFRecordWriter(target_path)

	print("Start writing to %s" % target_path)
	writer.write(serialized_features_dataset)
	print("Writing successful")

def read_TFRecord(filename, num_class=14):
	filenames = [filename]
	raw_dataset = tf.data.TFRecordDataset(filenames)

	# Create a description of the features.
	feature_description = {
		'image_path': tf.io.FixedLenFeature([], tf.string, default_value=''),
		'patient_data': tf.io.FixedLenFeature([4], tf.float32, default_value=[0]*4),
		'label': tf.io.FixedLenFeature([num_class], tf.float32, default_value=[0]*num_class),
	}

	def _parse_function(example_proto):
		# Parse the input `tf.Example` proto using the dictionary above.
		return tf.io.parse_single_example(example_proto, feature_description)

	parsed_dataset = raw_dataset.map(_parse_function)

	return parsed_dataset

def read_dataset(filename, dataset_path, image_only=True, num_class=14, evaluation_mode=False, shuffle=True):
	if evaluation_mode:
		dataset = read_TFRecord(filename, num_class)
		dataset = dataset.map(lambda data: (
		load_image(tf.strings.join([dataset_path, '/', data["image_path"]])), data["patient_data"], data["label"]),
							  num_parallel_calls=tf.data.experimental.AUTOTUNE)  # load the image
		dataset = dataset.map(lambda image, _, label: (image, tf.gather(label, tf.constant(EVAL_FIVE_CATS_INDEX))),
							  num_parallel_calls=tf.data.experimental.AUTOTUNE) if image_only else dataset  # if image only throw away patient data
	else:
		dataset = read_TFRecord(filename, num_class)
		dataset = dataset.map(lambda data: (
		load_image(tf.strings.join([dataset_path, '/', data["image_path"]])), data["patient_data"], data["label"]),
							  num_parallel_calls=tf.data.experimental.AUTOTUNE)  # load the image
		dataset = dataset.map(lambda image, _, label: (image, label), num_parallel_calls=tf.data.experimental.AUTOTUNE) if image_only else dataset  # if image only throw away patient data
	dataset = dataset.shuffle(BUFFER_SIZE) if shuffle else dataset
	dataset = dataset.batch(BATCH_SIZE)  # shuffle and batch with length of padding according to the the batch

	return dataset

def convert_ones_to_multi_classes(label):
	return np.array([[0,1] if l else [1,0] for l in label], dtype=np.float32).flatten()

@tf.function(input_signature=(tf.TensorSpec(shape=[NUM_CLASSES], dtype=tf.float32),))
def mapping(label):
	y = tf.numpy_function(convert_ones_to_multi_classes, [label], tf.float32)
	y.set_shape(NUM_CLASSES * 2)

	return y

def read_dataset_multi_class(filename, dataset_path, image_only=True, num_class=14):
	dataset = read_TFRecord(filename, num_class)
	dataset = dataset.map(lambda data: (load_image(tf.strings.join([dataset_path, '/', data["image_path"]])), data["patient_data"], mapping(data["label"])),
	                      num_parallel_calls=tf.data.experimental.AUTOTUNE)  # load the image
	dataset = dataset.map(lambda image, _, label: (image, label), num_parallel_calls=tf.data.experimental.AUTOTUNE) if image_only else dataset  # if image only throw away patient data
	dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)  # shuffle and batch with length of padding according to the the batch

	return dataset