"""
Train normal model with binary XE as loss function
"""
from tensorflow.python.keras.callbacks import configure_callbacks
from tqdm import tqdm

from _callbacks import get_callbacks
from datasets.cheXpert_dataset import read_dataset
from models.discriminator import make_discriminator_model
from models.gan import *
from utils._auc import AUC
from utils.visualization import *

# global local vars
TARGET_DATASET_FILENAME = CHESTXRAY_TRAIN_TARGET_TFRECORD_PATH
TARGET_DATASET_PATH = CHESTXRAY_DATASET_PATH

USE_XE = False

if __name__ == "__main__":
    model = GANModel()
    discriminator = make_discriminator_model()

    # get the dataset
    train_dataset = read_dataset(TRAIN_TARGET_TFRECORD_PATH, DATASET_PATH, use_augmentation=USE_AUGMENTATION,
                                 use_patient_data=USE_PATIENT_DATA,
                                 use_feature_loss=True,
                                 secondary_filename=TARGET_DATASET_FILENAME,
                                 secondary_dataset_path=TARGET_DATASET_PATH, use_preprocess_img=True)
    val_dataset = read_dataset(VALID_TARGET_TFRECORD_PATH, DATASET_PATH, use_patient_data=USE_PATIENT_DATA,
                               use_feature_loss=False, use_preprocess_img=True)
    test_dataset = read_dataset(TEST_TARGET_TFRECORD_PATH, DATASET_PATH, use_patient_data=USE_PATIENT_DATA,
                                use_feature_loss=False, use_preprocess_img=True)

    # losses, optimizer, metrics
    _XEloss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.AUTO)

    # optimizer
    _optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, amsgrad=True)
    _optimizer_disc = tf.keras.optimizers.Adam(LEARNING_RATE, amsgrad=True)

    # _metric = AUC(name="auc", multi_label=True, num_classes=NUM_CLASSES)  # give recall for metric it is more accurate
    _metric = tf.keras.metrics.AUC(name="auc")  # give recall for metric it is more accurate
    _callbacks = get_callbacks()

    # build CallbackList
    _callbackList = configure_callbacks(_callbacks,
                                        model,
                                        do_validation=True,
                                        epochs=MAX_EPOCHS,
                                        mode=tf.estimator.ModeKeys.EVAL,
                                        verbose=0)

    # retrieve checkpoint
    init_epoch = 0
    if LOAD_WEIGHT_BOOL:
        target_model_weight, init_epoch = get_max_acc_weight(MODELCKP_PATH)
        if target_model_weight:  # if weight is Found
            model.load_weights(target_model_weight)
        else:
            print("[Load weight] No weight is found")

    # set all the parameters
    model_params = {
        "optimizer": _optimizer,
        "loss": {"predictions": _XEloss},
        "metrics": {"predictions": _metric}
    }

    model.compile(**model_params)  # compile model

    fit_params = {
        "epochs": MAX_EPOCHS,
        "validation_data": val_dataset,
        "initial_epoch": init_epoch,
        # "steps_per_epoch":2,
        "callbacks": _callbacks,
        "verbose": 1
    }

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    # save checkpoints
    checkpoint_dir = './checkpoints/disc'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        optimizer=_optimizer,
        optimizer_disc=_optimizer_disc,
        discriminator_optimizer=_optimizer_disc,
        discriminator=discriminator)


    class TrainWorker:
        def __init__(self, metric, lambda_weight=0.01, lambda_adv=0.001):
            self.metric = metric
            self.lambda_adv = lambda_adv
            self.lambda_weight = lambda_weight

        def calc_weight_loss(self, name):
            _weights_1 = model.get_layer(name).weights
            _weights_1 = tf.reshape(_weights_1[0], [-1])
            # _weights_1 = tf.squeeze(_weights_1[0])

            _weights_2 = model.get_layer(name + "_target").weights
            _weights_2 = tf.reshape(_weights_2[0], [-1])
            # _weights_2 = tf.squeeze(_weights_2[0])

            weight_loss = tf.math.abs(tf.keras.losses.cosine_similarity(_weights_1, _weights_2))
            # weight_loss = tf.keras.losses.cosine_similarity(_weights_1, _weights_2) + 1.

            return weight_loss

        # Notice the use of `tf.function`
        # This annotation causes the function to be "compiled".
        @tf.function
        def gan_train_step(self, source_image_batch, source_label_batch, target_image_batch):
            with tf.GradientTape(persistent=True) as g:
                source_predictions = model(source_image_batch, training=True)
                target_predictions = model(target_image_batch, training=True,
                                           dont_stop_gradient_shared=GAN_TRAIN_SHARED_FEATURES)

                # input the predicted feature to the discriminator
                source_disc_output = tf.stop_gradient(discriminator(source_predictions[1], training=True))
                target_disc_output = discriminator(target_predictions[2], training=True)

                # calculate xe loss
                source_xe_loss = _XEloss(source_label_batch, source_predictions[0])

                # calculate weights loss
                weight_loss = self.calc_weight_loss("block14_sepconv1")
                weight_loss += self.calc_weight_loss("block14_sepconv2")
                # weight_loss = tf.reduce_mean(weight_loss)

                # calculate gen loss, disc loss
                _one_matrix = tf.ones_like(target_disc_output)
                _zero_matrix = tf.zeros_like(target_disc_output)
                # _adap_weight = tf.stop_gradient(1. - tf.keras.losses.cosine_similarity(target_predictions[2], target_predictions[3]))
                # tf.reduce_mean((self.lambda_local * _adap_weight + self.eps_adv) *

                gen_loss = tf.reduce_mean(cross_entropy(_one_matrix, target_disc_output))

                disc_loss = tf.reduce_mean(cross_entropy(_one_matrix, source_disc_output) +
                                           cross_entropy(_zero_matrix, target_disc_output))

                total_loss = source_xe_loss + self.lambda_adv * gen_loss + self.lambda_weight * weight_loss
                # total_loss = source_xe_loss + self.lambda_weight * weight_loss

            gradients_of_model = g.gradient(total_loss, model.trainable_variables)
            gradients_of_discriminator = g.gradient(disc_loss, discriminator.trainable_variables)

            del g  # delete the persistent gradientTape

            _optimizer_disc.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            _optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))

            # calculate metrics
            self.metric.update_state(source_label_batch, source_predictions[0])

            return source_xe_loss, gen_loss, disc_loss, weight_loss, 0.

        @tf.function
        def xe_train_step(self, source_image_batch, source_label_batch, target_image_batch):
            with tf.GradientTape(persistent=True) as g:
                source_predictions = model(source_image_batch, training=True)

                # calculate xe loss
                xe_loss = _XEloss(source_label_batch, source_predictions[0])

                total_loss = xe_loss

            gradients_of_model = g.gradient(total_loss, model.trainable_variables)

            del g  # delete the persistent gradientTape

            _optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))

            # calculate metrics
            self.metric.update_state(source_label_batch, source_predictions[0])

            return xe_loss, 0., 0., 0., 0.


    # initiate worker
    trainWorker = TrainWorker(_metric, lambda_weight=LAMBDA_WEI, lambda_adv=LAMBDA_ADV)

    ## find initial epoch and load the weights too
    init_epoch = 0
    if LOAD_WEIGHT_BOOL:
        target_model_weight, init_epoch = get_max_acc_weight(MODELCKP_PATH)
        if target_model_weight:  # if weight is Found
            model.load_weights(target_model_weight)
        else:
            print("[Load weight] No weight is found")

    # load disc and optimizer checkpoints
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # training loop
    _callbackList.on_train_begin()

    # var for save checkpoint
    _global_auc = 0.
    num_losses = 6
    losses = [tf.keras.metrics.Mean() for _ in range(num_losses)]
    for epoch in range(init_epoch, fit_params["epochs"]):
        print("Epoch %d/%d" % (epoch + 1, fit_params["epochs"]))
        _callbackList.on_epoch_begin(epoch)  # on epoch start

        # reset losses mean
        [loss.reset_states() for loss in losses]

        g = trainWorker.gan_train_step if not USE_XE else trainWorker.xe_train_step
        with tqdm(total=math.ceil(TRAIN_N / BATCH_SIZE),
                  postfix=[
                      dict(xe_loss=np.inf, gen_loss=np.inf, disc_loss=np.inf, weight_loss=np.inf, _adap_weight=np.inf,
                           AUC=0.)]) as t:
            for i_batch, (source_image_batch, (source_label_batch, target_image_batch)) in enumerate(
                    train_dataset):
                _batch_size = tf.shape(source_image_batch)[0].numpy()
                _callbackList.on_batch_begin(i_batch, {"size": _batch_size})  # on batch begin

                _losses = g(source_image_batch, source_label_batch, target_image_batch)
                _auc = trainWorker.metric.result().numpy()

                # update loss
                [losses[i].update_state(_losses[i]) for i in range(num_losses - 1)]
                losses[num_losses - 1].update_state(_auc)

                # update tqdm
                t.postfix[0]["xe_loss"] = losses[0].result().numpy()
                t.postfix[0]["gen_loss"] = losses[1].result().numpy()
                t.postfix[0]["disc_loss"] = losses[2].result().numpy()
                t.postfix[0]["weight_loss"] = losses[3].result().numpy()
                t.postfix[0]["_adap_weight"] = losses[4].result().numpy()
                t.postfix[0]["AUC"] = losses[5].result().numpy()
                t.update()

                _callbackList.on_batch_end(i_batch, {"loss": losses[0].result()})  # on batch end

        # epoch_end
        print()
        print("Validating...")
        results = model.evaluate(val_dataset, callbacks=_callbacks)
        _callbackList.on_epoch_end(epoch, {"loss": losses[0].result(),
                                           "gen_loss": losses[1].result(),
                                           "disc_loss": losses[2].result(),
                                           "auc": _auc,
                                           "val_loss": results[0],
                                           "val_auc": results[2],
                                           "lr": model.optimizer.lr})  # on epoch end

        # reset states
        trainWorker.metric.reset_states()

        # save checkpoint
        if USE_GAN:
            if _auc > _global_auc:
                _global_auc = _auc
                checkpoint.save(file_prefix=checkpoint_prefix)

    _callbackList.on_train_end()

    # Evaluate the model on the test data using `evaluate`
    results = model.evaluate(test_dataset)
    print('test loss, test f1, test auc:', results)
