from common_definitions import *


def make_discriminator_model():
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Reshape((-1, 1)))
    #
    # model.add(tf.keras.layers.Conv1D(1, 5))
    # model.add(tf.keras.layers.BatchNormalization(axis=0))
    # model.add(tf.keras.layers.LeakyReLU())
    #
    # model.add(tf.keras.layers.Conv1D(1, 5))
    # model.add(tf.keras.layers.BatchNormalization(axis=0))
    # model.add(tf.keras.layers.LeakyReLU())
    #
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dropout(DROPOUT_N))
    # model.add(tf.keras.layers.Dense(1))

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(1024, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(GLOBAL_ACTIVATION))

    model.add(tf.keras.layers.Dense(1))

    return model


if __name__ == "__main__":
    pass
