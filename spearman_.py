# calculate the spearman's correlation between two variables
from numpy.random import rand
from numpy.random import seed
from scipy.stats import spearmanr
# seed random number generator
seed(1)
# prepare data
chestxray_chexpert = [0.82, 0.81, 0.86, 0.8, 0.84]  # train_dataset ... _ ... test_dataset
chexpert_chexpert = [0.82, 0.9, 0.88, 0.82, 0.92]
chestxray_chestxray = [0.89, 0.82, 0.73, 0.74, 0.81]
chexpert_chestxray = [.87, .8, .71, .74, .81]

# calculate spearman's correlation
coef, p = spearmanr(chestxray_chestxray, chexpert_chestxray)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('Samples are correlated (reject H0) p=%.3f' % p)