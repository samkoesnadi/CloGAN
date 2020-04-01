from common_definitions import *
from datasets.common import *
from utils.utils import _np_to_binary

if __name__ == "__main__":
    train_dataset = read_dataset(TRAIN_TARGET_TFRECORD_PATH, DATASET_PATH, use_augmentation=False)

    _classes_measure = np.zeros(2**NUM_CLASSES, dtype=np.uint8)
    for (train_img, label) in tqdm(train_dataset):
        _index_target = np.apply_along_axis(_np_to_binary, 1, label)
        for _i in _index_target:
            _classes_measure[_i] += 1

    print("total empty classes:", (_classes_measure == 0).sum())
    print("max amount", _classes_measure.max())
    print("min amount", _classes_measure.min())
    # plt.bar(np.arange(0, 2**NUM_CLASSES), _classes_measure, width=1)
    # plt.savefig("test.png")