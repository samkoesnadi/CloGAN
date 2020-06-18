"""
Predict symptoms in chest X-ray image with exclusively CloGAN model's architecture
"""

from common_definitions import *
from datasets.cheXpert_dataset import *
from utils.utils import *
from utils.visualization import *
from models.multi_label import *
import skimage
import skimage.color
from models.gan import *
from models.multi_class import *
import time

target_filename = "resources/effusion.png"
if __name__ == "__main__":
	model = GANModel()
	# to initiate the graph
	model.call_w_features(tf.zeros((1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1)))

	target_model_weight = "resources/model.hdf5"
	model.load_weights(target_model_weight)

	model = tf.keras.Model(inputs=model.input_layer, outputs=model.call_w_everything(model.input_layer))

	# iteration for inputing image to be predicted
	while True:
		target_filename = input("Enter chest X-ray image file path:")

		# exit command
		if target_filename == "exit":
			break

		# the data
		try:
			_image = read_image_and_preprocess(target_filename, use_sn=False, use_preprocess_img=True)
		except:
			# detect file not found
			print("Image file can not be processed. Please try again!")
			continue

		image_ori = skimage.color.gray2rgb(read_image_and_preprocess(target_filename, use_sn=False, use_preprocess_img=False))

		image = np.reshape(_image, (-1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1))

		# predict the image
		start_time = time.time()
		prediction = np.squeeze(model.predict(image)[0])
		print("Time spent on predicting:", time.time()-start_time, "s")

		# generate GradCAM++
		gradcampps = Xception_gradcampp(model, image)

		results = np.zeros((NUM_CLASSES, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3))

		for i_g, gradcampp in enumerate(gradcampps):
			gradcampp = convert_to_RGB(gradcampp)

			result = .7 * image_ori + .3 * (gradcampp-gradcampp.min())/(gradcampp.max()-gradcampp.min())
			results[i_g] = result


		max_row_n = ceil(math.sqrt(NUM_CLASSES + 1))
		f, axarr = plt.subplots(max_row_n, max_row_n)

		# show input / ori image
		axarr[0, 0].imshow(image_ori)
		axarr[0, 0].set_title("Input")
		axarr[0, 0].axis('off')

		for i_r, result in enumerate(results, 1):
			axarr[i_r // max_row_n, i_r % max_row_n].imshow(result)
			axarr[i_r // max_row_n, i_r % max_row_n].set_title(LABELS_KEY[i_r - 1] + "({:.2f})".format(prediction[i_r - 1]))
			axarr[i_r // max_row_n, i_r % max_row_n].axis('off')

		plt.grid(b=None)
		plt.axis('off')
		plt.show()