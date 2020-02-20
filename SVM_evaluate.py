"""
Train normal model with binary XE as loss function
"""
from common_definitions import *
from datasets.cheXpert_dataset import read_dataset
from utils.utils import *
from utils.visualization import *
from models.multi_class import *

if __name__ == "__main__":
	# if LOAD_WEIGHT_BOOL:
	# 	model = tf.keras.models.load_model(SAVED_MODEL_PATH, custom_objects={'weighted_loss': get_square_hinge_weighted_loss(CHEXPERT_CLASS_WEIGHT),
	# 	                                                                     'f1': f1_svm,
	# 	                                                                     'AUC_SVM': AUC_SVM})
	# else:
	# 	model = model_MC_SVM()

	if USE_CLASS_WEIGHT:
		_loss = get_square_hinge_weighted_loss(CHEXPERT_CLASS_WEIGHT)
	else:
		_loss = tf.keras.losses.SquaredHinge()

	_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, amsgrad=True)

	f1_svm.__name__ = "f1"
	_metrics = {"predictions" : [f1_svm, AUC_SVM(name="auc")]}  # give recall for metric it is more accurate

	model = model_MC_SVM()
	if LOAD_WEIGHT_BOOL:
		target_model_weight, _ = get_max_acc_weight(MODELCKP_PATH)
		if target_model_weight:  # if weight is Found
			model.load_weights(target_model_weight)
		else:
			print("[Load weight] No weight is found")

	model.compile(optimizer=_optimizer,
	              loss=_loss,
	              metrics=_metrics)

	# get the dataset
	# train_dataset = read_dataset(CHEXPERT_TRAIN_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	# val_dataset = read_dataset(CHEXPERT_VALID_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	test_dataset = read_dataset(TEST_TARGET_TFRECORD_PATH, DATASET_PATH)

	test_label_nps = np.empty((TEST_N, NUM_CLASSES), dtype=np.float32)
	test_img_nps = np.empty((TEST_N, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1), dtype=np.float32)
	# get the ground truth labels
	for i_d, (test_img, test_label) in enumerate(test_dataset):
		test_label_nps[i_d*BATCH_SIZE:i_d*BATCH_SIZE + BATCH_SIZE] = test_label
		test_img_nps[i_d*BATCH_SIZE:i_d*BATCH_SIZE + BATCH_SIZE] = test_img


	# Evaluate the model on the test data using `evaluate`
	results = custom_sigmoid(model.predict(test_img_nps,
	                         # steps=ceil(CHEXPERT_TEST_N / BATCH_SIZE),
	                         verbose=1)).numpy()

	print("F1: ", np.mean(f1(test_label_nps, results).numpy()))

	plot_roc(test_label_nps, results)

	results = model.evaluate(test_dataset,
	                         # steps=ceil(CHEXPERT_TEST_N / BATCH_SIZE)
	                         )
	print('test loss, test f1, test auc:', results)