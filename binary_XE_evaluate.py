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

	test_label_nps = np.empty((CHEXPERT_TEST_N, NUM_CLASSES), dtype=np.float32)
	test_img_nps = np.empty((CHEXPERT_TEST_N, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1), dtype=np.float32)
	# get the ground truth labels
	for i_d, (test_img, test_label) in enumerate(test_dataset):
		test_label_nps[i_d*BATCH_SIZE:i_d*BATCH_SIZE + BATCH_SIZE] = test_label
		test_img_nps[i_d*BATCH_SIZE:i_d*BATCH_SIZE + BATCH_SIZE] = test_img


	# Evaluate the model on the test data using `evaluate`
	results = model.predict(test_img_nps,
	                         # steps=ceil(CHEXPERT_TEST_N / BATCH_SIZE),
	                         verbose=1)

	print("F1: ", np.mean(f1(test_label_nps, results).numpy()))

	plot_roc(test_label_nps, results)

	results = model.evaluate(test_dataset,
	                         # steps=ceil(CHEXPERT_TEST_N / BATCH_SIZE)
	                         )
	print('test loss, test f1, test auc:', results)