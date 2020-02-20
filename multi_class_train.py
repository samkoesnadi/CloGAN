"""
Train normal model with binary XE as loss function
"""
from common_definitions import *
from datasets.cheXpert_dataset import read_dataset, read_image_and_preprocess, read_dataset_multi_class
from utils.utils import *
from utils.visualization import *
from models.multi_class import *
import skimage.color
from utils.cylical_learning_rate import CyclicLR

if __name__ == "__main__":
	model = model_MC_softmax()

	# get the dataset
	train_dataset = read_dataset_multi_class(TRAIN_TARGET_TFRECORD_PATH, DATASET_PATH)
	val_dataset = read_dataset_multi_class(VALID_TARGET_TFRECORD_PATH, DATASET_PATH)
	test_dataset = read_dataset_multi_class(TEST_TARGET_TFRECORD_PATH, DATASET_PATH)

	@tf.function
	def multi_class_loss(y_true, y_pred):
		y_true = tf.reshape(y_true, [-1, NUM_CLASSES, 2])
		y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES, 2])

		y_together = tf.stack([y_true, y_pred], 2)
		y_together = tf.transpose(y_together, [1,2,0,3])

		sum = tf.constant(0.)
		for y in y_together:
			sum += tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(y[0], y[1]))
		return sum / NUM_CLASSES
	_loss = multi_class_loss

	_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, amsgrad=True)

	f1_mc.__name__ = "f1"
	_metrics = {"predictions" : [f1_mc, AUC_MC(name="auc")]}  # give recall for metric it is more accurate

	clr = CyclicLR(base_lr=CLR_BASELR, max_lr=CLR_MAXLR,
				   step_size=CLR_PATIENCE*ceil(TRAIN_N / BATCH_SIZE), mode='triangular')

	model_ckp = tf.keras.callbacks.ModelCheckpoint(MODELCKP_PATH,
	                                               monitor="val_auc",
	                                               verbose=1,
	                                               save_best_only=MODELCKP_BEST_ONLY,
	                                               save_weights_only=True,
	                                               mode="max")
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc',
													  verbose=1,
													  patience=int(CLR_PATIENCE * 2.5),
													  mode='max',
													  restore_best_weights=True)

	init_epoch = 0
	if LOAD_WEIGHT_BOOL:
		target_model_weight, init_epoch = get_max_acc_weight(MODELCKP_PATH)
		if target_model_weight:  # if weight is Found
			model.load_weights(target_model_weight)
		else:
			print("[Load weight] No weight is found")


	model.compile(optimizer=_optimizer,
                  loss=_loss,
                  metrics=_metrics)

	file_writer_cm = tf.summary.create_file_writer(TENSORBOARD_LOGDIR + '/cm')
	# define image logging
	def log_gradcampp(epoch, logs):
		_image = read_image_and_preprocess(SAMPLE_FILENAME, use_sn=True)
		image_ori = skimage.color.gray2rgb(read_image_and_preprocess(SAMPLE_FILENAME, use_sn=False))

		image = np.reshape(_image, (-1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1))

		prediction = model.predict(image)[0]
		prediction = prediction[1::2]  # get the true prediction

		prediction_dict = {LABELS_KEY[i]: prediction[i] for i in range(prediction.size)}

		lr = logs["lr"] if "lr" in logs else LEARNING_RATE

		gradcampps = Xception_gradcampp(model, image, use_svm=False, use_multi_class=True)

		results = np.zeros((NUM_CLASSES, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3))

		for i_g, gradcampp in enumerate(gradcampps):

			gradcampp = convert_to_RGB(gradcampp)

			result = .5 * image_ori + .5 * gradcampp
			results[i_g] = result

		# Log the gradcampp as an image summary.
		with file_writer_cm.as_default():
			tf.summary.text("Patient 0 prediction:", str(prediction_dict), step=epoch,
			                description="Prediction from sample file")
			tf.summary.image("Patient 0", results, max_outputs=NUM_CLASSES, step=epoch, description="GradCAM++ per classes")
			tf.summary.scalar("epoch_lr", lr, step=epoch)


	tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGDIR,
				                                     histogram_freq=1,
				                                     write_grads=True,
				                                     write_graph=False,
				                                     write_images=False)

	# Define the per-epoch callback.
	cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_gradcampp)

	# _callbacks = [clr, model_ckp, tensorboard_cbk, cm_callback, early_stopping]  # callbacks list
	_callbacks = [model_ckp, tensorboard_cbk, cm_callback, early_stopping]  # callbacks list

	# start training
	model.fit(train_dataset,
	          epochs=MAX_EPOCHS,
	          validation_data=val_dataset,
	          initial_epoch=init_epoch,
	          callbacks=_callbacks,
	          verbose=1)

	# Evaluate the model on the test data using `evaluate`
	results = model.evaluate(test_dataset,
	                         # steps=ceil(CHEXPERT_TEST_N / BATCH_SIZE)
	                         )
	print('test loss, test f1, test auc:', results)

	# Save the entire model to a HDF5 file.
	# The '.h5' extension indicates that the model shuold be saved to HDF5.
	get_and_mkdir(SAVED_MODEL_PATH)
	model.save(SAVED_MODEL_PATH)