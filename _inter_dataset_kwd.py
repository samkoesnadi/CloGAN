from numpy.random import default_rng
from utils.welford import Welford
from datasets.cheXpert_dataset import read_dataset
from utils.visualization import *
from models.multi_label import *
from sklearn.decomposition import PCA
from utils.kwd import *

# swap np with cp
USE_CUPY = True
try:
    import cupy as np
except ImportError as e:
    USE_CUPY = False

N_SAMPLES = 60
FEATURES_N = 64


if __name__ == "__main__":
    features_nps_1 = np.load(FEATURES_NP_FILE_1 + "_dimred.npy")
    features_nps_2 = np.load(FEATURES_NP_FILE_2 + "_dimred.npy")

    # run kernel_wasserstein_distance
    rng = default_rng()
    sample_numbers = np.array(rng.choice(TRAIN_N, size=N_SAMPLES, replace=False))
    features_nps_1 = np.array(features_nps_1)  # because of cupy conversion
    features_nps_2 = np.array(features_nps_2)  # because of cupy conversion
    welford_ = Welford()

    with tqdm(total=N_SAMPLES, desc="MAIN LOOP",
              postfix=[int(0), dict(value=0)]) as t:
        for i in sample_numbers:
            for j in tqdm(range(CHESTXRAY_TRAIN_N), desc="iter for TRAIN_N"):
                if i == j: continue
                welford_(kernel_wasserstein_distance(features_nps_1[i], features_nps_2[j], USE_CUPY))
            t.set_postfix(i_=i, value=welford_)
            t.update()

    print(welford_)
    print("k:", welford_.k)