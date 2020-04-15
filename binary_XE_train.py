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

# global local vars
TARGET_DATASET_FILENAME = CHESTXRAY_TRAIN_TARGET_TFRECORD_PATH
TARGET_DATASET_PATH = CHESTXRAY_DATASET_PATH

if __name__ == "__main__":
    model = model_binaryXE_mid(USE_PATIENT_DATA) if USE_FEATURE_LOSS else model_binaryXE(USE_PATIENT_DATA)

    # get the dataset
    train_dataset = read_dataset(TRAIN_TARGET_TFRECORD_PATH, DATASET_PATH, use_augmentation=USE_AUGMENTATION,
                                 use_patient_data=USE_PATIENT_DATA, use_feature_loss=USE_FEATURE_LOSS,
                                 secondary_filename=TARGET_DATASET_FILENAME,
                                 secondary_dataset_path=TARGET_DATASET_PATH)
    val_dataset = read_dataset(VALID_TARGET_TFRECORD_PATH, DATASET_PATH, use_patient_data=USE_PATIENT_DATA,
                               use_feature_loss=USE_FEATURE_LOSS,
                               secondary_filename=TARGET_DATASET_FILENAME,
                               secondary_dataset_path=TARGET_DATASET_PATH)
    test_dataset = read_dataset(TEST_TARGET_TFRECORD_PATH, DATASET_PATH, use_patient_data=USE_PATIENT_DATA,
                                use_feature_loss=USE_FEATURE_LOSS,
                                secondary_filename=TARGET_DATASET_FILENAME,
                                secondary_dataset_path=TARGET_DATASET_PATH)

    clr = CyclicLR(base_lr=CLR_BASELR, max_lr=CLR_MAXLR,
                   step_size=CLR_PATIENCE * ceil(TRAIN_N / BATCH_SIZE), mode='triangular')

    _losses = []

    # _XEloss = get_weighted_loss(CHEXPERT_CLASS_WEIGHT)
    _XEloss = myBinaryXE(num_classes=NUM_CLASSES, from_logits=False)
    _losses.append(_XEloss)

    if USE_FEATURE_LOSS:
        _losses.append(FeatureLoss(num_classes=NUM_CLASSES, alpha=FeL_ALPHA, net_model=model, _feloss_alpha=RATIO_INTRATERTD,
                                   _index_var=_XEloss.indexs, _index_ones_var=_XEloss.indexs_ones))

    # a = model(np.random.normal(0, 1, (1,224,224,1)))
    # # test featureloss
    # a = _losses[1](tf.convert_to_tensor(np.random.choice(2, size=(32, 14), p=[0.5, 0.5]).astype(np.float32)),
    #                tf.convert_to_tensor(np.random.normal(0, 1, (32, 2048)), dtype=tf.float32))
    # print(a)
    # exit()

    _optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, amsgrad=True)
    _metrics = {"predictions": [f1, AUC(name="auc", multi_label=True, num_classes=NUM_CLASSES)]}  # give recall for metric it is more accurate

    model_ckp = tf.keras.callbacks.ModelCheckpoint(MODELCKP_PATH,
                                                   monitor="val_predictions_auc" if USE_FEATURE_LOSS else "val_auc",
                                                   verbose=1,
                                                   save_best_only=MODELCKP_BEST_ONLY,
                                                   save_weights_only=True,
                                                   mode="max")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_predictions_auc" if USE_FEATURE_LOSS else 'val_auc',
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

        if USE_FEATURE_LOSS:
            prediction = prediction[0]

        prediction_dict = {LABELS_KEY[i]: prediction[i] for i in range(NUM_CLASSES)}

        lr = logs["lr"] if "lr" in logs else LEARNING_RATE

        gradcampps = Xception_gradcampp(model, image, patient_data=patient_data, use_feature_loss=USE_FEATURE_LOSS)

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


    class FECallback(tf.keras.callbacks.Callback):
        def __init__(self, pred_loss, feature_loss, base_feature_loss_ratio=0.5):
            super().__init__()
            self.pred_loss = pred_loss
            self.feature_loss = feature_loss
            self.base_feature_loss_ratio = base_feature_loss_ratio

            self._lowest_pred_loss = 0.

        # customize your behavior
        def on_train_batch_end(self, batch, logs=None):
            _cur_pred_loss = logs["predictions_loss"]
            # _cur_feat_loss = logs["tf_op_layer_image_feature_vectors_loss"]

            if batch == 0:
                self._lowest_pred_loss = _cur_pred_loss

            # self.feature_loss.assign(min((self._lowest_pred_loss / _cur_pred_loss), 1.) ** 2 * self.base_feature_loss_ratio)

            # TODO: check if this is valid
            if self._lowest_pred_loss < _cur_pred_loss:
                self.feature_loss.assign(0.)
            else:
                self.feature_loss.assign(self.base_feature_loss_ratio)

            logs["rat_feL"] = self.feature_loss

            # update the lowerst pred loss
            self._lowest_pred_loss = self._lowest_pred_loss + UPDATE_LOSS_SCHEDULER_ALPHA * (_cur_pred_loss - self._lowest_pred_loss)

    _callbacks = []

    if USE_FEATURE_LOSS:
        _callbacks.append(FECallback(RATIO_LOSSES[0], RATIO_LOSSES[1], base_feature_loss_ratio=BASE_FELOSS_RAT))

    _callbacks.extend([clr if USE_CLR else lrate, tensorboard_cbk, model_ckp, early_stopping])  # callbacks list



    model.compile(optimizer=_optimizer,
                  loss=_losses,
                  metrics=_metrics,
                  loss_weights={"predictions": RATIO_LOSSES[0], "tf_op_layer_image_feature_vectors": RATIO_LOSSES[1]} if USE_FEATURE_LOSS else {"predictions": 1}
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
