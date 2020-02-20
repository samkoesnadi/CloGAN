"""
Train normal model with binary XE as loss function
"""
from common_definitions import *
from datasets.cheXpert_dataset import *
from utils.utils import *
from utils.visualization import *
from models.multi_class import *
import skimage
import skimage.color


target_filename = "./sample/00002032_006.png"
target_filename = "~/Downloads/pneumonia3.jpg"
if __name__ == "__main__":
	model = model_MC_SVM()
	if LOAD_WEIGHT_BOOL:
		target_model_weight, _ = get_max_acc_weight(MODELCKP_PATH)
		if target_model_weight:  # if weight is Found
			model.load_weights(target_model_weight)
		else:
			print("[Load weight] No weight is found")

	# the data
	_image = read_image_and_preprocess(target_filename, use_sn=True)
	image_ori = skimage.color.gray2rgb(read_image_and_preprocess(target_filename, use_sn=False))

	image = np.reshape(_image, (-1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1))

	prediction = custom_sigmoid(model.predict(image)).numpy()

	gradcampps = Xception_gradcampp(model, image, use_svm=True)

	results = np.zeros((NUM_CLASSES, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3))

	for i_g, gradcampp in enumerate(gradcampps):
		gradcampp = convert_to_RGB(gradcampp)

		result = .5 * image_ori + .5 * gradcampp
		results[i_g] = result


	max_row_n = ceil(math.sqrt(NUM_CLASSES + 1))
	f, axarr = plt.subplots(max_row_n, max_row_n)

	# show input / ori image
	axarr[0, 0].imshow(image_ori)
	axarr[0, 0].set_title("Input")
	axarr[0, 0].axis('off')

	for i_r, result in enumerate(results, 1):
		axarr[i_r // max_row_n, i_r % max_row_n].imshow(result)
		axarr[i_r // max_row_n, i_r % max_row_n].set_title(LABELS_KEY[i_r - 1] + "({:.2f})".format(prediction[0, i_r - 1]))
		axarr[i_r // max_row_n, i_r % max_row_n].axis('off')

	plt.grid(b=None)
	plt.axis('off')
	plt.show()

