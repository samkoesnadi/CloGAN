import cv2
import numpy as np
import tensorflow as tf
def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs.shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    gaussian = np.random.normal(size=(row, col, 1), scale=1.).astype(np.float32)
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
    gaussian_img = cv2.addWeighted(X_imgs, 0.75, 0.25 * gaussian, 0.25, 0)
    return gaussian_img
GN = tf.keras.layers.GaussianNoise(1.)
def gauss_noise(x):
    return (x + tf.random.normal(x.shape, stddev=tf.random.uniform([], minval=0, maxval=0.1))).numpy()
image = cv2.imread('sample/00002032_006.png').astype(np.float32) # Only for grayscale image
noise_img = gauss_noise(image/255)
cv2.imwrite('sp_noise.jpg', noise_img*255)