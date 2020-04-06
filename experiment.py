from common_definitions import *
import scipy.stats

if __name__ == "__main__":
    a = cp.random.normal(0, 1, 2048)
    b = cp.random.normal(1, 2., 2048)
    import time
    start_time = time.time()
    x = scipy.stats.wasserstein_distance(b, a)
    y = np.mean(b) - np.mean(a)
    print("time spent:", time.time() - start_time)
    print(x)
    print(y)