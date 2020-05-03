"""
Normal model with binary XE as loss function
"""
from common_definitions import *

def xception_end_layer_block(inp, filters, name):
    sep_conv = tf.keras.layers.SeparableConv2D(filters, kernel_size=3, name=name, padding="same",
                                               kernel_initializer=KERNEL_INITIALIZER, use_bias=False)(inp)
    _bn = tf.keras.layers.BatchNormalization(name=name+"_bn")(sep_conv)
    _act = tf.keras.layers.Activation(GLOBAL_ACTIVATION, name=name+"_act")(_bn)

    return _act

def raw_model_binaryXE_gan():
    input_layer = tf.keras.layers.Input(shape=(IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1), name="input_img")
    image_section_model = tf.keras.applications.xception.Xception(include_top=False, weights=None, pooling=None,
                                                                  input_tensor=input_layer)
    _add_layer = image_section_model.get_layer("add_11").output

    source_sep_conv1_act = xception_end_layer_block(_add_layer, 1536, "block14_sepconv1")
    source_sep_conv2 = tf.keras.layers.SeparableConv2D(2048, kernel_size=3, name="block14_sepconv2", padding="same",
                                                       kernel_initializer=KERNEL_INITIALIZER, use_bias=False)(source_sep_conv1_act)

    target_sep_conv1_act = xception_end_layer_block(_add_layer, 1536, "block14_sepconv1_target")
    target_sep_conv2 = tf.keras.layers.SeparableConv2D(2048, kernel_size=3, name="block14_sepconv2_target", padding="same",
                                                       kernel_initializer=KERNEL_INITIALIZER, use_bias=False)(target_sep_conv1_act)

    # post-process the image features
    _bn = tf.keras.layers.BatchNormalization(name="block14_sepconv2_bn")(source_sep_conv2)  # the input can be from source or mixed
    _act = tf.keras.layers.Activation(GLOBAL_ACTIVATION, name="block14_sepconv2_act")(_bn)

    image_section_layer = tf.keras.layers.GlobalAveragePooling2D()(_act)

    output_layer_dense = tf.keras.layers.Dense(NUM_CLASSES, kernel_initializer=KERNEL_INITIALIZER)

    output_layer = tf.keras.layers.Activation("sigmoid", name='predictions')(output_layer_dense(image_section_layer))

    return input_layer, output_layer, source_sep_conv2, target_sep_conv2


def model_binaryXE_mid_gan():
    layers = raw_model_binaryXE_gan()

    model = tf.keras.Model(inputs=layers[0],
                           outputs=layers[1:])
    return model


def model_binaryXE_gan():
    layers = raw_model_binaryXE_gan()

    model = tf.keras.Model(inputs=layers[0], outputs=layers[1])
    return model
