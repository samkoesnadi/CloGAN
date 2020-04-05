"""
Train normal model with binary XE as loss function
"""
from common_definitions import *
from datasets.cheXpert_dataset import read_dataset
from utils.utils import *
from utils.visualization import *
from models.multi_label import *
from models.multi_class import *

if __name__ == "__main__":
	if USE_SVM:
		model = model_MC_SVM()
	else:
		model = model_binaryXE(use_patient_data=USE_PATIENT_DATA)

	if LOAD_WEIGHT_BOOL:
		target_model_weight, _ = get_max_acc_weight(MODELCKP_PATH)
		if target_model_weight:  # if weight is Found
			model.load_weights(target_model_weight)
		else:
			print("[Load weight] No weight is found")

	# get the dataset
	test_dataset = read_dataset(CHEXPERT_TEST_TARGET_TFRECORD_PATH if EVAL_CHEXPERT else CHESTXRAY_TEST_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH if EVAL_CHEXPERT else CHESTXRAY_DATASET_PATH, use_patient_data=USE_PATIENT_DATA, evaluation_mode=True)

	_test_n = CHEXPERT_TEST_N if EVAL_CHEXPERT else CHESTXRAY_TEST_N

	results = np.zeros((_test_n, 5), dtype=np.float32)
	test_label_nps = np.zeros((_test_n, 5), dtype=np.float32)
	# get the ground truth labels
	for i_d, (test_img, test_label) in tqdm(enumerate(test_dataset)):
		_batch_to_fill = test_img.shape[0] if not USE_PATIENT_DATA else test_img["input_img"].shape[0]

		# Evaluate the model on the test data using `evaluate`
		result = model.predict(test_img)
		result = result[:, TRAIN_FIVE_CATS_INDEX]

		results[i_d * BATCH_SIZE : i_d * BATCH_SIZE + _batch_to_fill] = result
		test_label_nps[i_d * BATCH_SIZE : i_d * BATCH_SIZE + _batch_to_fill] = test_label

	# start calculating metrics
	print("F1: ", np.mean(f1(test_label_nps, results).numpy()))

	roc_auc = plot_roc(test_label_nps, results, True)

	print("test auc:", roc_auc, ", auc macro:", sum(roc_auc)/len(roc_auc))