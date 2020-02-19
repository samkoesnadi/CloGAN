"""
Normal model with binary XE as loss function
"""
from common_definitions import *

def model_MC_SVM():
	input_layer = tf.keras.layers.Input(shape=(IMAGE_INPUT_SIZE,IMAGE_INPUT_SIZE,1), name="input")
	image_section_model = tf.keras.applications.xception.Xception(include_top=False, weights=None, pooling="avg", input_tensor=input_layer)
	image_section_layer = image_section_model.output

	image_section_layer = tf.keras.layers.Dropout(DROPOUT_N)(image_section_layer)
	output_layer = [tf.keras.layers.Dense(1, activation="linear", kernel_regularizer=tf.keras.regularizers.l2(SVM_KERNEL_REGULARIZER), kernel_initializer=KERNEL_INITIALIZER)(image_section_layer) for _ in range(NUM_CLASSES_CHEXPERT)]

	output_layer = tf.keras.layers.Concatenate(name='predictions')(output_layer)

	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	return model

def model_MC_softmax():
	input_layer = tf.keras.layers.Input(shape=(IMAGE_INPUT_SIZE,IMAGE_INPUT_SIZE,1), name="input")
	image_section_model = tf.keras.applications.xception.Xception(include_top=False, weights=None, pooling="avg", input_tensor=input_layer)
	image_section_layer = image_section_model.output

	image_section_layer = tf.keras.layers.Dropout(DROPOUT_N)(image_section_layer)
	output_layer = [tf.keras.layers.Dense(2, activation="softmax", kernel_initializer=KERNEL_INITIALIZER)(image_section_layer) for _ in range(NUM_CLASSES_CHEXPERT)]

	output_layer = tf.keras.layers.Concatenate(name='predictions')(output_layer)

	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	return model