from common_definitions import *


def make_discriminator_model():
    input_1 = tf.keras.Input(shape=2048)

    hidden_1 = tf.keras.layers.Dense(6144, use_bias=False)(input_1)
    hidden_1_bn = tf.keras.layers.BatchNormalization()(hidden_1)
    hidden_1_act = tf.keras.layers.Activation(GLOBAL_ACTIVATION)(hidden_1_bn)

    hidden = tf.keras.layers.Dropout(DROPOUT_N)(hidden_1_act)

    output = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(hidden)

    return tf.keras.Model(inputs=input_1, outputs=output)


if __name__ == "__main__":
    pass
