from sklearn.manifold import Isomap, TSNE
from common_definitions import *
import time
from utils.visualization import *
from datasets.cheXpert_dataset import read_dataset
from models.multi_label import model_binaryXE_mid
from models.multi_class import model_MC_SVM
from utils.utils import _np_to_binary

PRINT_PREDICTION = False

if __name__ == "__main__":
    if USE_SVM:
        model = model_MC_SVM(with_feature=True)
    else:
        model = model_binaryXE_mid(use_patient_data=USE_PATIENT_DATA)

    if LOAD_WEIGHT_BOOL:
        target_model_weight, _ = get_max_acc_weight(MODELCKP_PATH)
        if target_model_weight:  # if weight is Found
            model.load_weights(target_model_weight)
        else:
            print("[Load weight] No weight is found")

    test_dataset = read_dataset(
        CHEXPERT_TEST_TARGET_TFRECORD_PATH if EVAL_CHEXPERT else CHESTXRAY_TEST_TARGET_TFRECORD_PATH,
        CHEXPERT_DATASET_PATH if EVAL_CHEXPERT else CHESTXRAY_DATASET_PATH, evaluation_mode=True, use_patient_data=USE_PATIENT_DATA)

    _test_n = CHEXPERT_TEST_N if EVAL_CHEXPERT else CHESTXRAY_TEST_N

    _color_label = []
    _feature_nps = []
    for i_test, (input, label) in tqdm(enumerate(test_dataset)):
        predictions = model.predict(input)

        label = (predictions[0][:, TRAIN_FIVE_CATS_INDEX] >= 0.3).astype(np.float32) if PRINT_PREDICTION else label.numpy()
        labels = np.array(list(map(_np_to_binary, label)))
        feature_vectors = predictions[1]

        # filter zeros
        _i_zeros = np.argwhere(labels != 0)[:, 0]
        labels = labels[_i_zeros]
        feature_vectors = feature_vectors[_i_zeros]

        _color_label.extend(list(labels))
        _feature_nps.extend(feature_vectors)

    # convert to np arrays
    _color_label = np.array(_color_label)
    _feature_nps = np.array(_feature_nps)

    start_time = time.time()
    embedding = TSNE(n_components=2, init='pca', random_state=0, verbose=True)
    X_embedded = embedding.fit_transform(_feature_nps)
    print("time spent for manifold learning:", time.time() - start_time)

    # sketch it
    if EVAL_CHEXPERT:
        _scatter_plt = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=_color_label, cmap=plt.cm.Spectral)
    else:
        _scatter_plt = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=_color_label, cmap=plt.cm.Spectral, s=5)

    plt.colorbar(_scatter_plt)
    plt.axis('tight')

    get_and_mkdir("report/results/manifold_learning.png")
    plt.savefig("report/results/manifold_learning.png", bbox_inches="tight")
