"""
Train normal model with binary XE as loss function
"""
from common_definitions import *
from datasets.cheXpert_dataset import read_dataset
from utils.utils import *
from utils.visualization import *
from models.multi_label import *

if __name__ == "__main__":
	if LOAD_WEIGHT_BOOL:
		model = tf.keras.models.load_model(SAVED_MODEL_PATH, custom_objects={'weighted_loss': get_weighted_loss(CHEXPERT_CLASS_WEIGHT), 'f1': f1})
	else:
		model = model_binaryXE()


	# get the dataset
	# train_dataset = read_dataset(CHEXPERT_TRAIN_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	# val_dataset = read_dataset(CHEXPERT_VALID_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	test_dataset = read_dataset(CHEXPERT_TEST_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)

	test_labels = []
	# get the ground truth labels
	for _, test_label in test_dataset:
		test_labels.extend(test_label)
	test_labels = np.array(test_labels)


	# Evaluate the model on the test data using `evaluate`
	results = model.predict(test_dataset,
	                         # steps=ceil(CHEXPERT_TEST_N / BATCH_SIZE),
	                         verbose=1)

	print("F1: ", np.mean(f1(test_labels, results).numpy()))

	plot_roc(test_labels, results)

	results = model.evaluate(test_dataset,
	                         # steps=ceil(CHEXPERT_TEST_N / BATCH_SIZE)
	                         )
	print('test loss, test f1, test auc:', results)