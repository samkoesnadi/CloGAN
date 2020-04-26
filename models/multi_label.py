"""
Normal model with binary XE as loss function
"""
from common_definitions import *
from utils.weightnorm import WeightNormalization

def raw_model_binaryXE(use_patient_data=False, use_wn=USE_WN):
    input_layer = tf.keras.layers.Input(shape=(IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1), name="input_img")
    image_section_model = tf.keras.applications.xception.Xception(include_top=False, weights=None, pooling=None,
                                                                  input_tensor=input_layer)
    image_feature_vectors = image_section_model.output

    # # add regularizer as the experiments show that it results positively when the avg value of image_feature_vectors is small
    # image_feature_vectors = tf.keras.layers.ActivityRegularization(l2=ACTIVIY_REGULARIZER_VAL)(image_feature_vectors)
    image_feature_vectors = tf.identity(image_feature_vectors, name="image_feature_vectors")  # to change the name

    if use_patient_data:
        # process semantic
        input_semantic = tf.keras.layers.Input(shape=PAT_DATA_SIZE, name="input_semantic")

        # Apply Batch Normalization to convert the range ro mean 0 and std 1
        int_semantic = tf.keras.layers.BatchNormalization()(input_semantic)
        int_semantic = tf.keras.layers.Dropout(0.2)(int_semantic) if USE_DROPOUT_PAT_DATA else int_semantic

        feature_vectors = tf.keras.layers.Concatenate()([image_feature_vectors, int_semantic])
    else:
        feature_vectors = image_feature_vectors

    image_section_layer = tf.keras.layers.GlobalAveragePooling2D()(feature_vectors)

    if USE_CONV1D:
        image_section_layer = tf.expand_dims(image_section_layer, -1)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(1, 5))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv1D(1, 5))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Flatten())

        image_section_layer = model(image_section_layer)

    # add dropout if needed
    if DROPOUT_N != 0.:
        image_section_layer = tf.keras.layers.Dropout(DROPOUT_N)(image_section_layer)

    output_layer_dense = tf.keras.layers.Dense(NUM_CLASSES, kernel_initializer=KERNEL_INITIALIZER)

    if use_wn:
        weight_normalized = WeightNormalization(output_layer_dense, data_init=True)(
            image_section_layer)

        output_layer = tf.keras.layers.Activation("sigmoid", dtype='float32', name='predictions')(weight_normalized)
    else:
        output_layer = tf.keras.layers.Activation("sigmoid", dtype='float32', name='predictions')(output_layer_dense(image_section_layer))

    if use_patient_data:
        return (input_layer, input_semantic), output_layer, feature_vectors
    else:
        return input_layer, output_layer, feature_vectors


def model_binaryXE(use_patient_data=False, use_wn=USE_WN):
    input_layer, output_layer, _ = raw_model_binaryXE(use_patient_data=use_patient_data, use_wn=use_wn)

    model = tf.keras.Model(inputs=input_layer, outputs=[output_layer])
    return model


def model_binaryXE_mid(use_patient_data=False, use_wn=USE_WN):
    input_layer, output_layer, image_section_layer_feature = raw_model_binaryXE(use_patient_data=use_patient_data,
                                                                                use_wn=use_wn)

    model = tf.keras.Model(inputs=input_layer, outputs=[output_layer, image_section_layer_feature])
    return model
