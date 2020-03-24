"""
Normal model with binary XE as loss function
"""
from common_definitions import *

def raw_model_binaryXE(use_patient_data=False):
	input_layer = tf.keras.layers.Input(shape=(IMAGE_INPUT_SIZE,IMAGE_INPUT_SIZE,1), name="input_img")
	image_section_model = tf.keras.applications.xception.Xception(include_top=False, weights=None, pooling="avg", input_tensor=input_layer)

	if use_patient_data:
		# process semantic
		input_semantic = tf.keras.layers.Input(shape=(4), name="input_semantic")
		int_semantic = tf.keras.layers.Dense(48, activation=tf.nn.leaky_relu, kernel_initializer=KERNEL_INITIALIZER)(input_semantic)

		image_section_1536 = image_section_model.layers[128].output
		image_section_2000 = tf.keras.layers.SeparableConv2D(2000, (3,3), padding="same")(image_section_1536)
		image_section_2000 = tf.keras.layers.BatchNormalization()(image_section_2000)
		image_section_2000 = tf.keras.layers.LeakyReLU()(image_section_2000)
		image_section_layer_feature = tf.keras.layers.GlobalAveragePooling2D()(image_section_2000)
		feature_vectors = tf.keras.layers.Concatenate()([image_section_layer_feature, int_semantic])
	else:
		feature_vectors = image_section_model.output

	image_section_layer = tf.keras.layers.Dropout(DROPOUT_N)(feature_vectors)
	output_layer = tf.keras.layers.Dense(NUM_CLASSES, kernel_initializer=KERNEL_INITIALIZER)(image_section_layer)
	output_layer = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(output_layer)

	if use_patient_data:
		return input_layer, input_semantic, output_layer, feature_vectors
	else:
		return input_layer, output_layer, feature_vectors

def model_binaryXE(use_patient_data=False):
	if use_patient_data:
		input_layer, input_semantic, output_layer, _ = raw_model_binaryXE(use_patient_data)

		model = tf.keras.Model(inputs=[input_layer, input_semantic], outputs=[output_layer])
	else:
		input_layer, output_layer, _ = raw_model_binaryXE()

		model = tf.keras.Model(inputs=input_layer, outputs=[output_layer])
	return model

def model_binaryXE_mid(use_patient_data=False):
	if use_patient_data:
		input_layer, input_semantic, output_layer, image_section_layer_feature = raw_model_binaryXE()

		model = tf.keras.Model(inputs=[input_layer, input_semantic], outputs=[output_layer, image_section_layer_feature])
	else:
		input_layer, output_layer, image_section_layer_feature = raw_model_binaryXE()

		model = tf.keras.Model(inputs=input_layer, outputs=[output_layer, image_section_layer_feature])
	return model