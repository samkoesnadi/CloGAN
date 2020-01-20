"""
Normal model with binary XE as loss function
"""
from common_definitions import *
from datasets.cheXpert_dataset import read_dataset
from math import ceil

if __name__ == "__main__":
	input_layer = tf.keras.layers.Input(shape=(IMAGE_INPUT_SIZE,IMAGE_INPUT_SIZE,1))
	image_section_model = tf.keras.applications.xception.Xception(include_top=False, weights=None, pooling="avg", input_tensor=input_layer)
	image_section_layer = image_section_model.output

	output_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid", name="predictions")(image_section_layer)

	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

	_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
	_loss = tf.keras.losses.BinaryCrossentropy()
	_metric = tf.keras.metrics.BinaryAccuracy(threshold=THRESHOLD_SIGMOID)  # give recall for metric it is more accurate

	model.compile(optimizer=_optimizer,
                  loss=_loss,
                  metrics=[_metric])

	# get the dataset
	train_dataset = read_dataset(CHEXPERT_TRAIN_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	val_dataset = read_dataset(CHEXPERT_VALID_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	test_dataset = read_dataset(CHEXPERT_TEST_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)

	model.fit(train_dataset,
	          epochs=MAX_EPOCHS,
	          validation_data=val_dataset,
	          validation_steps=ceil(CHEXPERT_VAL_N / BATCH_SIZE),
	          steps_per_epoch=ceil(CHEXPERT_TRAIN_N / BATCH_SIZE))  ## ?

	# Evaluate the model on the test data using `evaluate`
	results = model.evaluate(test_dataset,
	                         steps=ceil(CHEXPERT_TEST_N / BATCH_SIZE))
	print('test loss, test acc:', results)