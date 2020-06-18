"""
Lists of callbacks for training
"""

from common_definitions import *
from datasets.cheXpert_dataset import read_dataset, read_image_and_preprocess
from utils.utils import *
from utils.visualization import *
from models.multi_label import *
import skimage.color
from utils.cylical_learning_rate import CyclicLR
from utils._auc import AUC

def get_callbacks(model=None):
    clr = CyclicLR(base_lr=CLR_BASELR, max_lr=CLR_MAXLR,
                   step_size=CLR_PATIENCE * ceil(TRAIN_N / BATCH_SIZE), mode='triangular')
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

    tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGDIR,
                                                     histogram_freq=1,
                                                     write_grads=True,
                                                     write_graph=False,
                                                     write_images=False)

    # Define the per-epoch callback.
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)

    _callbacks = [clr if USE_CLR else lrate, tensorboard_cbk, model_ckp]  # callbacks list
    # _callbacks = [tensorboard_cbk, model_ckp, early_stopping]  # callbacks list

    if USE_EARLY_STOPPING:
        _callbacks.append(early_stopping)

    return _callbacks
