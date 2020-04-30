"""
Normal model with binary XE as loss function
"""
from common_definitions import *
from utils.weightnorm import WeightNormalization

def raw_model_binaryXE_gan():
    input_layer = tf.keras.layers.Input(shape=(IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1), name="input_img")
    image_section_model = tf.keras.applications.xception.Xception(include_top=False, weights=None, pooling=None,
                                                                  input_tensor=input_layer)
    image_feature_vectors = image_section_model.output

    # add dropout if needed
    if DROPOUT_N != 0.:
        image_feature_vectors = tf.keras.layers.Dropout(DROPOUT_N)(image_feature_vectors)

    # # add regularizer as the experiments show that it results positively when the avg value of image_feature_vectors is small
    # image_feature_vectors = tf.keras.layers.ActivityRegularization(l2=ACTIVIY_REGULARIZER_VAL)(image_feature_vectors)

    feature_vectors_1 = tf.keras.layers.Conv2D(NUM_CLASSES, kernel_size=1, name="predictions_1")(image_feature_vectors)

    feature_vectors_2 = tf.keras.layers.Conv2D(NUM_CLASSES, kernel_size=1, name="predictions_2")(image_feature_vectors)

    feature_vectors = tf.keras.layers.Add()([feature_vectors_1, feature_vectors_2])
    # feature_vectors = tf.keras.layers.Activation("sigmoid", dtype='float32', name='predictions')(feature_vectors)

    output_layer = tf.keras.layers.GlobalAveragePooling2D()(feature_vectors)
    output_layer = tf.keras.layers.Activation("sigmoid", dtype='float32', name='predictions')(output_layer)

    return input_layer, output_layer, feature_vectors, feature_vectors_1, feature_vectors_2


def model_binaryXE_mid_gan():
    input_layer, output_layer, feature_vectors, feature_vectors_1, feature_vectors_2 = raw_model_binaryXE_gan()

    model = tf.keras.Model(inputs=input_layer, outputs=[output_layer, feature_vectors, feature_vectors_1, feature_vectors_2])
    return model
