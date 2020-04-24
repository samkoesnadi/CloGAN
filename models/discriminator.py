from common_definitions import *


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1))

    return model


if __name__ == "__main__":
    pass
