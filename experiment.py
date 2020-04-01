import cv2
import numpy as np
import tensorflow as tf
import scipy.spatial

if __name__ == "__main__":
    a = np.array([1,0,0])
    b = np.array([1,1,0])
    c = np.array([0,1,1])

    print(scipy.spatial.distance.cosine(a,b))
    print(scipy.spatial.distance.cosine(a,c))
    print((1 - (a*b).sum()/np.sqrt((a**2).sum() * (b**2).sum())))