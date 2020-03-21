from utils.kwd import *

a = np.random.normal(0, 1, 1000)
b = np.random.normal(0.5, 1, 1000)

print(kernel_wasserstein_distance(a, b))