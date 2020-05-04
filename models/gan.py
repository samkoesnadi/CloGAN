"""
Normal model with binary XE as loss function
"""
from common_definitions import *


# def xception_end_layer_block(inp, filters, name):
#     sep_conv = tf.keras.layers.SeparableConv2D(filters, kernel_size=3, name=name, padding="same",
#                                                kernel_initializer=KERNEL_INITIALIZER, use_bias=False)(inp)
#     _bn = tf.keras.layers.BatchNormalization(name=name+"_bn")(sep_conv)
#     _act = tf.keras.layers.Activation(GLOBAL_ACTIVATION, name=name+"_act")(_bn)
#
#     return _act

# def raw_model_binaryXE_gan():
#     input_layer = tf.keras.layers.Input(shape=(IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1), name="input_img")
#     image_section_model = tf.keras.applications.xception.Xception(include_top=False, weights=None, pooling=None,
#                                                                   input_tensor=input_layer)
#     _add_layer = image_section_model.get_layer("add_11").output
#
#     source_sep_conv1_act = xception_end_layer_block(_add_layer, 1536, "block14_sepconv1")
#     source_sep_conv2 = tf.keras.layers.SeparableConv2D(2048, kernel_size=3, name="block14_sepconv2", padding="same",
#                                                        kernel_initializer=KERNEL_INITIALIZER, use_bias=False)(source_sep_conv1_act)
#
#     target_sep_conv1_act = xception_end_layer_block(_add_layer, 1536, "block14_sepconv1_target")
#     target_sep_conv2 = tf.keras.layers.SeparableConv2D(2048, kernel_size=3, name="block14_sepconv2_target", padding="same",
#                                                        kernel_initializer=KERNEL_INITIALIZER, use_bias=False)(target_sep_conv1_act)
#
#     # post-process the image features
#     _bn = tf.keras.layers.BatchNormalization(name="block14_sepconv2_bn")(source_sep_conv2)  # the input can be from source or mixed
#     _act = tf.keras.layers.Activation(GLOBAL_ACTIVATION, name="block14_sepconv2_act")(_bn)
#
#     image_section_layer = tf.keras.layers.GlobalAveragePooling2D()(_act)
#
#     output_layer_dense = tf.keras.layers.Dense(NUM_CLASSES, kernel_initializer=KERNEL_INITIALIZER)
#
#     output_layer = tf.keras.layers.Activation("sigmoid", name='predictions')(output_layer_dense(image_section_layer))
#
#     return input_layer, output_layer, source_sep_conv2, target_sep_conv2

class EndBlock(tf.keras.layers.Layer):
    def __init__(self, filters, name, **kwargs):
        super().__init__(**kwargs)
        self.sep_conv = tf.keras.layers.SeparableConv2D(filters, kernel_size=3, padding="same",
                                                        kernel_initializer=KERNEL_INITIALIZER, use_bias=False)

        self._bn = tf.keras.layers.BatchNormalization(name=name + "_bn")
        self._act = tf.keras.layers.Activation(GLOBAL_ACTIVATION, name=name + "_act")
        self._name = name
        self._weights = self.sep_conv.weights

    def call(self, inputs, training=False, **kwargs):
        sep_conv = self.sep_conv(inputs)
        _bn = self._bn(sep_conv, training)
        _act = self._act(_bn)

        return _act


class GANModel(tf.keras.Model):
    def __init__(self):
        super(GANModel, self).__init__()

        self.input_layer = tf.keras.layers.Input(shape=(IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1), name="input_img")

        image_section_model = tf.keras.applications.xception.Xception(include_top=False, weights=None, pooling=None,
                                                                      input_tensor=self.input_layer)
        self._add_layer = image_section_model.get_layer("add_11").output

        self.shared_model = tf.keras.Model(inputs=self.input_layer, outputs=self._add_layer)

        self.source_sep_conv1_act = EndBlock(1536, "block14_sepconv1")
        self.source_sep_conv2 = tf.keras.layers.SeparableConv2D(2048, kernel_size=3, name="block14_sepconv2",
                                                                padding="same",
                                                                kernel_initializer=KERNEL_INITIALIZER, use_bias=False)

        self.target_sep_conv1_act = EndBlock(1536, "block14_sepconv1_target")
        self.target_sep_conv2 = tf.keras.layers.SeparableConv2D(2048, kernel_size=3, name="block14_sepconv2_target",
                                                                padding="same",
                                                                kernel_initializer=KERNEL_INITIALIZER, use_bias=False)

        # post-process the image features
        self._bn = tf.keras.layers.BatchNormalization(
            name="block14_sepconv2_bn")  # the input can be from source or mixed
        self._act = tf.keras.layers.Activation(GLOBAL_ACTIVATION, name="block14_sepconv2_act")

        self.image_section_layer = tf.keras.layers.GlobalAveragePooling2D()

        self.final_do = tf.keras.layers.Dropout(DROPOUT_N)

        self.output_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid", name='predictions',
                                                  kernel_initializer=KERNEL_INITIALIZER)

    def call_w_features(self, inputs, training=False, dont_stop_gradient_shared=True):
        shared_layer = self.shared_model(inputs, training) if dont_stop_gradient_shared else \
            tf.stop_gradient(self.shared_model(inputs, training))

        source_sep_conv1_act = self.source_sep_conv1_act(shared_layer, training)
        source_sep_conv2 = self.source_sep_conv2(source_sep_conv1_act)

        target_sep_conv1_act = self.target_sep_conv1_act(shared_layer, training)
        target_sep_conv2 = self.target_sep_conv2(target_sep_conv1_act)

        # post-process the image features
        _bn = self._bn(source_sep_conv2, training)  # the input can be from source or mixed
        _act = self._act(_bn)

        # _act += self._act(self._bn(target_sep_conv2, training))
        # _act /= 2

        image_section_layer = self.image_section_layer(_act)

        final_do = self.final_do(image_section_layer, training)

        output_layer = self.output_layer(final_do)

        return output_layer, source_sep_conv2, target_sep_conv2

    def call(self, inputs, training=False, **kwargs):
        return self.call_w_features(inputs, training=training, dont_stop_gradient_shared=True)["predictions"]


# def model_binaryXE_mid_gan():
#     layers = raw_model_binaryXE_gan()
#
#     model = tf.keras.Model(inputs=layers[0],
#                            outputs=layers[1:])
#     return model
#
#
# def model_binaryXE_gan():
#     layers = raw_model_binaryXE_gan()
#
#     model = tf.keras.Model(inputs=layers[0], outputs=layers[1])
#     return model
