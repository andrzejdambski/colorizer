from skimage import color
import numpy as np

def rgb_to_lab(X):
    """
    prends une liste d'array rgb; ressort un array lab
    """
    X_lab = color.rgb2lab(X)
    return X_lab

def extraire_L_et_A_B(img_lab):
    """
    A ce stade, la liste comprends des arrays LAB.
    Prend une liste, en resort 2, et pour chaque element de la liste le
    L est isole du A et B.
    """
    L = img_lab[...,0]
    A_B = img_lab[...,1:]

    return L.astype(np.float64), A_B.astype(np.float64)
