"""
Train normal model with binary XE as loss function
"""
from tensorflow.python.keras.callbacks import configure_callbacks
from tqdm import tqdm

from _callbacks import get_callbacks
from datasets.cheXpert_dataset import read_dataset
from models.discriminator import make_discriminator_model
from models.multi_label import *
from utils._auc import AUC
from utils.visualization import *

# global local vars
TARGET_DATASET_FILENAME = CHESTXRAY_TRAIN_TARGET_TFRECORD_PATH
TARGET_DATASET_PATH = CHESTXRAY_DATASET_PATH

if __name__ == "__main__":
    model = model_binaryXE_mid(USE_PATIENT_DATA, USE_WN)

    if USE_GAN:
        discriminator = make_discriminator_model()

    # get the dataset
    train_dataset = read_dataset(TRAIN_TARGET_TFRECORD_PATH, DATASET_PATH, use_augmentation=USE_AUGMENTATION,
                                 use_patient_data=USE_PATIENT_DATA,
                                 use_feature_loss=True,
                                 secondary_filename=TARGET_DATASET_FILENAME,
                                 secondary_dataset_path=TARGET_DATASET_PATH)
    val_dataset = read_dataset(VALID_TARGET_TFRECORD_PATH, DATASET_PATH, use_patient_data=USE_PATIENT_DATA,
                               use_feature_loss=False)
    test_dataset = read_dataset(TEST_TARGET_TFRECORD_PATH, DATASET_PATH, use_patient_data=USE_PATIENT_DATA,
                                use_feature_loss=False)

    # losses, optimizer, metrics
    _XEloss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.AUTO)

    # optimizer
    _optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, amsgrad=True, beta_1=0.5, beta_2=0.9)  # TODO
    _optimizer_disc = tf.keras.optimizers.Adam(LEARNING_RATE, amsgrad=True, beta_1=0.5, beta_2=0.9)

    _metric = AUC(name="auc", multi_label=True, num_classes=NUM_CLASSES)  # give recall for metric it is more accurate
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
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def discriminator_loss(source_output, target_output):
        real_loss = cross_entropy(tf.ones_like(target_output), target_output)
        fake_loss = cross_entropy(tf.zeros_like(source_output), source_output)
        total_loss = real_loss + fake_loss
        return total_loss


    def generator_loss(source_output, target_output):
        return cross_entropy(tf.ones_like(source_output), source_output) + cross_entropy(tf.zeros_like(target_output),
                                                                                         target_output)

    if USE_GAN:
        # save checkpoints
        checkpoint_dir = './checkpoints/disc'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=_optimizer,
                                         discriminator_optimizer=_optimizer_disc,
                                         discriminator=discriminator)


    class TrainWorker:
        def __init__(self, metric, reg_lambda=10., n_critic=5):
            self.metric = metric
            self.reg_lambda = reg_lambda  # this is for improved W-Gan
            self.n_critic = n_critic

            # updating var
            self.t_critic = 0

        # Notice the use of `tf.function`
        # This annotation causes the function to be "compiled".
        @tf.function
        def gan_train_step(self, source_image_batch, source_label_batch, target_image_batch, bs):
            _eps = tf.random.uniform((bs, 1))  # normal random
            with tf.GradientTape(persistent=True) as g:
                source_predictions = model(source_image_batch, training=True)
                target_predictions = model(target_image_batch, training=True)

                # input the predicted feature to the discriminator
                source_output = tf.reduce_mean(discriminator(source_predictions[1], training=True))
                target_output = tf.reduce_mean(discriminator(target_predictions[1], training=True))

                # x_tricap from the paper
                x_tricap = _eps * target_predictions[1] + (1 - _eps) * source_predictions[1]
                g.watch(x_tricap)

                # d grad for regularizer
                _d_grad = g.gradient(discriminator(x_tricap), x_tricap)
                _d_slopes = tf.sqrt(tf.reduce_sum(
                    tf.square(_d_grad), axis=1))

                disc_loss_unreg = target_output - source_output
                disc_loss = disc_loss_unreg + self.reg_lambda * tf.reduce_mean((_d_slopes - 1.) ** 2)

                # # calculate losses
                # xe_loss = _XEloss(source_label_batch, source_predictions[0])
                # gen_loss = generator_loss(source_output, target_output)
                # xe_gen_loss = xe_loss + gen_loss  # TODO
                # disc_loss = discriminator_loss(source_output, target_output)

                # calculate losses
                xe_loss = _XEloss(source_label_batch, source_predictions[0])
                gen_loss = -disc_loss_unreg

                if self.t_critic < (self.n_critic - 1):
                    xe_gen_loss = xe_loss
                    self.t_critic += 1
                else:
                    xe_gen_loss = xe_loss + gen_loss
                    self.t_critic = 0

            # gradients_of_xe = gen_tape.gradient(xe_loss, model.trainable_variables)
            # gradients_of_generator = gen_tape.gradient(gen_loss, model.trainable_variables)
            gradients_of_xe_gen = g.gradient(xe_gen_loss, model.trainable_variables)
            gradients_of_discriminator = g.gradient(disc_loss, discriminator.trainable_variables)

            _optimizer_disc.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            _optimizer.apply_gradients(zip(gradients_of_xe_gen, model.trainable_variables))

            # calculate metrics
            self.metric.update_state(source_label_batch, source_predictions[0])

            del g  # delete the persisten gradientTape
            return xe_loss, gen_loss, disc_loss

        @tf.function
        def mmd_train_step(self, source_image_batch, source_label_batch, target_image_batch, bs):
            with tf.GradientTape(persistent=True) as g:
                source_predictions = model(source_image_batch, training=True)
                target_predictions = model(target_image_batch, training=True)

                # notation based on MMD formula
                phi_xs = tf.reduce_mean(source_predictions[1], axis=0)
                phi_xt = tf.reduce_mean(target_predictions[1], axis=0)

                # calculate losses
                xe_loss = _XEloss(source_label_batch, source_predictions[0])
                dksq = tf.linalg.norm(phi_xs - phi_xt) ** 2

                total_loss = xe_loss + dksq

            # gradients_of_xe = gen_tape.gradient(xe_loss, model.trainable_variables)
            # gradients_of_generator = gen_tape.gradient(gen_loss, model.trainable_variables)
            gradients_of_xe_gen = g.gradient(total_loss, model.trainable_variables)

            _optimizer.apply_gradients(zip(gradients_of_xe_gen, model.trainable_variables))

            # calculate metrics
            self.metric.update_state(source_label_batch, source_predictions[0])

            del g  # delete the persisten gradientTape
            return xe_loss, dksq, 0.


    # initiate worker
    trainWorker = TrainWorker(_metric, reg_lambda=REG_LAMBDA, n_critic=N_CRITIC)

    ## find initial epoch and load the weights too
    init_epoch = 0
    if LOAD_WEIGHT_BOOL:
        target_model_weight, init_epoch = get_max_acc_weight(MODELCKP_PATH)
        if target_model_weight:  # if weight is Found
            model.load_weights(target_model_weight)
        else:
            print("[Load weight] No weight is found")

    # load disc and optimizer checkpoints
    if USE_GAN:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # training loop
    _callbackList.on_train_begin()

    # var for save checkpoint
    _global_auc = 0.

    for epoch in range(init_epoch, fit_params["epochs"]):
        print("Epoch %d/%d" % (epoch + 1, fit_params["epochs"]))
        _callbackList.on_epoch_begin(epoch)  # on epoch start
        with tqdm(total=math.ceil(TRAIN_N / BATCH_SIZE),
                  postfix=[dict(xe_loss=np.inf, gen_loss=np.inf, disc_loss=np.inf, AUC=0.)]) as t:
            for i_batch, (source_image_batch, (source_label_batch, target_image_batch)) in enumerate(
                    train_dataset):
                _batch_size = tf.shape(source_image_batch)[0].numpy()
                _callbackList.on_batch_begin(i_batch, {"size": _batch_size})  # on batch begin

                g = trainWorker.gan_train_step if USE_GAN else trainWorker.mmd_train_step
                xe_loss, gen_loss, disc_loss = g(source_image_batch, source_label_batch,
                                                 target_image_batch, _batch_size)
                _auc = trainWorker.metric.result().numpy()

                # update tqdm
                t.postfix[0]["xe_loss"] = xe_loss.numpy()
                t.postfix[0]["gen_loss"] = gen_loss.numpy()
                t.postfix[0]["disc_loss"] = disc_loss.numpy()
                t.postfix[0]["AUC"] = _auc
                t.update()

                _callbackList.on_batch_end(i_batch, {"loss": xe_loss})  # on batch end

        # epoch_end
        print()
        print("Validating...")
        results = model.evaluate(val_dataset, callbacks=_callbacks)
        _callbackList.on_epoch_end(epoch, {"loss": xe_loss.numpy(), "auc": _auc, "val_loss": results[0],
                                           "val_auc": results[2], "lr": model.optimizer.lr})  # on epoch end

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
