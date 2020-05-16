"""
Train normal model with binary XE as loss function
"""
from common_definitions import *
from datasets.cheXpert_dataset import read_dataset, read_image_and_preprocess
from utils.utils import *
from utils.visualization import *
from models.multi_label import *
import skimage.color
from utils.cylical_learning_rate import CyclicLR
from utils._auc import AUC


if __name__ == "__main__":
    model = model_binaryXE(USE_PATIENT_DATA, USE_WN)

    # get the dataset
    train_dataset = read_dataset(TRAIN_TARGET_TFRECORD_PATH, DATASET_PATH, use_augmentation=USE_AUGMENTATION,
                                 use_patient_data=USE_PATIENT_DATA, use_feature_loss=False)
    val_dataset = read_dataset(VALID_TARGET_TFRECORD_PATH, DATASET_PATH, use_patient_data=USE_PATIENT_DATA,
                               use_feature_loss=False)
    test_dataset = read_dataset(TEST_TARGET_TFRECORD_PATH, DATASET_PATH, use_patient_data=USE_PATIENT_DATA,
                                use_feature_loss=False)

    clr = CyclicLR(base_lr=CLR_BASELR, max_lr=CLR_MAXLR,
                   step_size=CLR_PATIENCE * ceil(TRAIN_N / BATCH_SIZE), mode='triangular')

    _losses = []

    # _XEloss = get_weighted_loss(CHEXPERT_CLASS_WEIGHT)
    _XEloss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    _losses.append(_XEloss)

    _optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, amsgrad=True)
    _metrics = {"predictions": [f1, AUC(name="auc", multi_label=True, num_classes=NUM_CLASSES)]}  # give recall for metric it is more accurate

    model_ckp = tf.keras.callbacks.ModelCheckpoint(MODELCKP_PATH,
                                                   monitor="val_auc",
                                                   verbose=1,
                                                   save_best_only=MODELCKP_BEST_ONLY,
                                                   save_weights_only=True,
                                                   mode="max")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                      verbose=1,
                                                      patience=int(CLR_PATIENCE * 2.5),
                                                      mode='max',
                                                      restore_best_weights=True)

    init_epoch = 0
    if LOAD_WEIGHT_BOOL:
        target_model_weight, init_epoch = get_max_acc_weight(MODELCKP_PATH)
        if target_model_weight:  # if weight is Found
            model.load_weights(target_model_weight)
        else:
            print("[Load weight] No weight is found")

    file_writer_cm = tf.summary.create_file_writer(TENSORBOARD_LOGDIR + '/cm')


    # define image logging
    def log_gradcampp(epoch, logs):
        _image = read_image_and_preprocess(SAMPLE_FILENAME, use_sn=True)
        image_ori = skimage.color.gray2rgb(read_image_and_preprocess(SAMPLE_FILENAME, use_sn=False))

        image = np.reshape(_image, (-1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1))

        patient_data = np.array([SAMPLE_PATIENT_DATA])
        if USE_PATIENT_DATA:
            prediction = model.predict({"input_img": image, "input_semantic": patient_data})[0]
        else:
            prediction = model.predict(image)[0]

        prediction_dict = {LABELS_KEY[i]: prediction[i] for i in range(NUM_CLASSES)}

        lr = logs["lr"] if "lr" in logs else LEARNING_RATE

        gradcampps = Xception_gradcampp(model, image, patient_data=patient_data, use_feature_loss=False)

        results = np.zeros((NUM_CLASSES, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3))

        for i_g, gradcampp in enumerate(gradcampps):
            gradcampp = convert_to_RGB(gradcampp)

            result = .5 * image_ori + .5 * gradcampp
            results[i_g] = result

        # Log the gradcampp as an image summary.
        with file_writer_cm.as_default():
            tf.summary.text("Patient 0 prediction:", str(prediction_dict), step=epoch,
                            description="Prediction from sample file")
            tf.summary.image("Patient 0", results, max_outputs=NUM_CLASSES, step=epoch,
                             description="GradCAM++ per classes")
            tf.summary.scalar("epoch_lr", lr, step=epoch)


    tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGDIR,
                                                     histogram_freq=1,
                                                     write_grads=True,
                                                     write_graph=False,
                                                     write_images=False)

    # Define the per-epoch callback.
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_gradcampp)
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)

    _callbacks = [clr if USE_CLR else lrate, tensorboard_cbk, model_ckp, early_stopping]  # callbacks list

    model.compile(optimizer=_optimizer,
                  loss=_losses,
                  metrics=_metrics
                  )

    # start training
    model.fit(train_dataset,
              epochs=MAX_EPOCHS,
              validation_data=val_dataset,
              initial_epoch=init_epoch,
              # steps_per_epoch=2,
              callbacks=_callbacks,
              verbose=1)

    # Evaluate the model on the test data using `evaluate`
    results = model.evaluate(test_dataset,
                             # steps=ceil(CHEXPERT_TEST_N / BATCH_SIZE)
                             )
    print('test loss, test f1, test auc:', results)
