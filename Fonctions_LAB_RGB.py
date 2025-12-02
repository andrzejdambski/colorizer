
# RGB batch -> Lab batch
import numpy as np
from skimage import color

def rgb_to_lab_batch(X):
    """
    prends un array rgb; ressort un array lab en float 32
    """
    X_lab = color.rgb2lab(X)
    return X_lab

# Lab batch -> RGB batch
def lab_to_rgb_batch(X_lab):
    """
    Convertit un batch Lab (N, H, W, 3) en RGB
    Retourne un array (N, H, W, 3) en float32.
    """
    X_rgb = color.lab2rgb(X_lab)
    return X_rgb
