"""
Train normal model with binary XE as loss function
"""
from common_definitions import *
from datasets.cheXpert_dataset import read_dataset, read_image_and_preprocess
from utils.utils import *
from utils.visualization import *
from models.multi_label import *
import skimage.color


if __name__ == "__main__":
	model = model_binaryXE()

	# get the dataset
	train_dataset = read_dataset(CHEXPERT_TRAIN_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	val_dataset = read_dataset(CHEXPERT_VALID_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	test_dataset = read_dataset(CHEXPERT_TEST_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)

	if USE_CLASS_WEIGHT:
		# train_labels = []
		# # get the ground truth labels
		# for _, train_label in tqdm(train_dataset):
		# 	train_labels.extend(train_label)
		# train_labels = np.array(train_labels)
		# print(calculating_class_weights(train_labels))

		_loss = get_weighted_loss(CHEXPERT_CLASS_WEIGHT)
	else:
		_loss = tf.keras.losses.BinaryCrossentropy()

	_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, amsgrad=True)
	_metrics = {"predictions" : [f1, tf.keras.metrics.AUC()]}  # give recall for metric it is more accurate

	# all callbacks
	# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=REDUCELR_FACTOR,
	#                                                  verbose=1,
	#                                                  patience=REDUCELR_PATIENCE,
	#                                                  min_lr=REDUCELR_MINLR,
	#                                                  mode="max")

	def step_decay(epoch):
		initial_lrate = LEARNING_RATE
		drop = REDUCELR_FACTOR
		epochs_drop = REDUCELR_PATIENCE
		lrate = initial_lrate * math.pow(drop,
		                                 math.floor((1 + epoch) / epochs_drop))
		return max(lrate, REDUCELR_MINLR)
	lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)

	model_ckp = tf.keras.callbacks.ModelCheckpoint(MODELCKP_PATH,
	                                               monitor="val_auc",
	                                               verbose=1,
	                                               save_best_only=MODELCKP_BEST_ONLY,
	                                               save_weights_only=True,
	                                               mode="max")
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc',
													  verbose=1,
													  patience=SUB_EPOCHS * 4,
													  mode='max',
													  restore_best_weights=True)


	class TensorBoardWrapper(tf.keras.callbacks.TensorBoard):
		'''Sets the self.validation_data property for use with TensorBoard callback.'''

		def __init__(self, batch_gen, nb_steps, b_size, **kwargs):
			super(TensorBoardWrapper, self).__init__(**kwargs)
			self.batch_gen = batch_gen  # The generator.
			self.nb_steps = nb_steps  # Number of times to call next() on the generator.
			self.batch_size = b_size

		def on_train_begin(self, logs):
			# Fill in the `validation_data` property. Obviously this is specific to how your generator works.
			# Below is an example that yields images and classification tags.
			# After it's filled in, the regular on_epoch_end method has access to the validation_data.
			imgs, tags = None, None
			for s, (ib, tb) in enumerate(self.batch_gen):
				if imgs is None and tags is None:
					imgs = np.zeros(((self.nb_steps * self.batch_size,) + ib.shape[1:]), dtype=np.float32)
					tags = np.zeros(((self.nb_steps * self.batch_size,) + tb.shape[1:]), dtype=np.float32)
				imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
				tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb

			self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]

			return super(TensorBoardWrapper, self).on_train_begin(logs)


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

		gradcampps = Xception_gradcampp(model, image)

		results = np.zeros((NUM_CLASSES, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3))

		for i_g, gradcampp in enumerate(gradcampps):

			gradcampp = convert_to_RGB(gradcampp)

			result = .5 * image_ori + .5 * gradcampp
			results[i_g] = result

		# Log the gradcampp as an image summary.
		with file_writer_cm.as_default():
			tf.summary.image("Patient 0", results, max_outputs=NUM_CLASSES, step=epoch, description="GradCAM++ per classes")


	tensorboard_cbk = TensorBoardWrapper(val_dataset, ceil(CHEXPERT_VAL_N / BATCH_SIZE), BATCH_SIZE,
	                                     log_dir=TENSORBOARD_LOGDIR,
	                                     histogram_freq=1,
	                                     write_grads=True,
	                                     write_graph=False,
	                                     write_images=False)

	# Define the per-epoch callback.
	cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_gradcampp)

	_callbacks = [model_ckp, early_stopping, tensorboard_cbk, cm_callback]  # callbacks list

	if USE_REDUCELR:
		_callbacks.append(lrate)

	# start training
	model.fit(train_dataset,
	          epochs=MAX_EPOCHS,
	          validation_data=val_dataset,
	          validation_steps=ceil(CHEXPERT_VAL_N / BATCH_SIZE),
	          initial_epoch=init_epoch,
	          steps_per_epoch=ceil(SUB_CHEXPERT_TRAIN_N / BATCH_SIZE),
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