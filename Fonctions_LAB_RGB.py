
# RGB batch -> Lab batch
from skimage import color

def rgb_to_lab_batch(X):
    """
    Convertit un batch RGB (N, H, W, 3) en Lab.
    """
    X_lab = color.rgb2lab(X)
    return X_lab

# Si on veut le contraire
# Lab batch -> RGB batch
def lab_to_rgb_batch(X_lab):
    """
    Convertit un batch Lab (N, H, W, 3) en RGB normalisé [0,1].
    """
    X_rgb = color.lab2rgb(X_lab)
    return X_rgb
