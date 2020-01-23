"""
Train normal model with binary XE as loss function
"""
from common_definitions import *
from datasets.cheXpert_dataset import read_dataset
from utils.utils import *
from utils.visualization import *
from models.multi_label import *

if __name__ == "__main__":
	model = model_binaryXE()

	_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, amsgrad=True)
	_loss = tf.keras.losses.BinaryCrossentropy()
	_metrics = {"predictions" : [f1, tf.keras.metrics.AUC()]}  # give recall for metric it is more accurate

	# all callbacks
	# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=REDUCELR_FACTOR,
	#                                                  verbose=1,
	#                                                  patience=REDUCELR_PATIENCE,
	#                                                  min_lr=REDUCELR_MINLR,
	#                                                  mode="max")
	model_ckp = tf.keras.callbacks.ModelCheckpoint(MODELCKP_PATH,
	                                               monitor="val_auc",
	                                               verbose=1,
	                                               save_best_only=MODELCKP_BEST_ONLY,
	                                               save_weights_only=True,
	                                               mode="max")
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc',
													  verbose=1,
													  patience=SUB_EPOCHS * 2,
													  mode='max',
													  restore_best_weights=True)
	tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGDIR,
	                                                 histogram_freq=1,
	                                                 write_grads=True,
													 write_graph=False,
	                                                 write_images=False)


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

	# get the dataset
	train_dataset = read_dataset(CHEXPERT_TRAIN_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	val_dataset = read_dataset(CHEXPERT_VALID_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	test_dataset = read_dataset(CHEXPERT_TEST_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)

	file_writer_cm = tf.summary.create_file_writer(TENSORBOARD_LOGDIR + '/cm')
	# define image logging
	def log_gradcampp(epoch, logs):
		_image = read_image_and_preprocess(SAMPLE_FILENAME)
		image = np.reshape(_image, (-1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1))

		gradcampps = Xception_gradcampp(model, image)

		results = np.zeros((NUM_CLASSES, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3))

		for i_g, gradcampp in enumerate(gradcampps):

			image = skimage.color.gray2rgb(_image)
			gradcampp = convert_to_RGB(gradcampp)

			result = .5 * image + .5 * gradcampp
			results[i_g] = result

		# Log the gradcampp as an image summary.
		with file_writer_cm.as_default():
			tf.summary.image("Patient 0", results, max_outputs=NUM_CLASSES, step=epoch, description="GradCAM++ per classes")

	# Define the per-epoch callback.
	cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_gradcampp)

	_callbacks = [model_ckp, early_stopping, tensorboard_cbk, cm_callback]  # callbacks list

	class_weight = None
	if USE_CLASS_WEIGHT:
		# class weight
		class_weight = dict()
		total = sum(pos.values())
		for p in pos:
			weight_for_1 = (1 / pos[p]) * (total) / NUM_CLASSES  # num classes as the total is calculated from all the positives
			class_weight[p] = weight_for_1

	# start training
	model.fit(train_dataset,
	          epochs=MAX_EPOCHS,
	          validation_data=val_dataset,
	          class_weight=class_weight,
	          # validation_steps=ceil(CHEXPERT_VAL_N / BATCH_SIZE),
	          initial_epoch=init_epoch,
	          # steps_per_epoch=ceil(SUB_CHEXPERT_TRAIN_N / BATCH_SIZE),
	          callbacks=_callbacks,
	          verbose=1)

	# # start training
	# model.fit(train_dataset,
	#           epochs=MAX_EPOCHS,
	#           validation_data=val_dataset,
	#           class_weight=class_weight,
	#           validation_steps=ceil(CHEXPERT_VAL_N / BATCH_SIZE),
	#           initial_epoch=init_epoch,
	#           steps_per_epoch=ceil(SUB_CHEXPERT_TRAIN_N / BATCH_SIZE),
	#           use_multiprocessing=True,
	#           callbacks=_callbacks)

	# Evaluate the model on the test data using `evaluate`
	results = model.evaluate(test_dataset,
	                         # steps=ceil(CHEXPERT_TEST_N / BATCH_SIZE)
	                         )
	print('test loss, test f1, test auc:', results)