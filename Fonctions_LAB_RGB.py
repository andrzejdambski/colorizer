
# RGB batch -> Lab batch
import numpy as np
from skimage import color

def rgb_to_lab_batch(X):
    """
    Convertit un batch RGB (N, H, W, 3) en Lab.
    X doit être en float32 et normalisé entre 0 et 1.

    Retourne un array (N, H, W, 3) en Lab.
    """
    if X.dtype != np.float32 and X.dtype != np.float64:
        raise ValueError("X doit être en float32 ou float64 et normalisé entre 0 et 1.")

    X_lab = np.zeros_like(X)
    for i in range(len(X)):
        X_lab[i] = color.rgb2lab(X[i])
    return X_lab

# Lab batch -> RGB batch
def lab_to_rgb_batch(X_lab):
    """
    Convertit un batch Lab (N, H, W, 3) en RGB normalisé [0,1].

    Retourne un array (N, H, W, 3) en float32.
    """
    X_rgb = np.zeros_like(X_lab)
    for i in range(len(X_lab)):
        X_rgb[i] = color.lab2rgb(X_lab[i])
    return X_rgb.astype(np.float32)
