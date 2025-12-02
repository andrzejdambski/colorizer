# # Pour une seule image (entrée = image RGB NumPy, float32, valeurs [0,1])
# import numpy as np
# from skimage.color import rgb2lab

# def extraire_L_et_original(img_lab):
#     """
#     img_rgb : image RGB en numpy, shape (H, W, 3), normalisée entre [0,1]

#     Retourne :
#        L : image (H, W, 1) canal L normalisé sur [0,1] plutôt sur -1 et 1
#        original : image RGB identique à l'entrée
#     """

#     if img_lab.ndim != 3 or img_lab.shape[-1] != 3:
#         raise ValueError("img_rgb doit être de forme (H, W, 3).")


#     # Extraire L
#     L = img_lab[...,0]       # shape (H, W, 1)

#     # Normaliser entre [0,1] mettre plutôt entre -1 et 1
#     # L = L / 100.0

#     return L.astype(np.float32), img_lab.astype(np.float32)


import numpy as np
from skimage.color import rgb2lab

def extraire_L_et_original_batch(X_rgb):
    """
    X_rgb : batch d'images RGB, shape (N, H, W, 3), en float32.

    Retourne :
      - L : (N, H, W, 1)  canal L normalisé sur [0,1]
      - X_rgb : (N, H, W, 3)  images originales (inchangées)
    """
    if X_rgb.ndim != 4 or X_rgb.shape[-1] != 3:
        raise ValueError("X_rgb doit être de forme (N, H, W, 3).")

    # RGB -> Lab
    X_lab = rgb2lab(X_rgb)          # L dans [0,100]

    # On extrait L et on garde la dimension canal
    L = X_lab[..., 0:1]             # (N, H, W, 1)

    # Normalisation L ∈ [0,1]
    L = L / 100.0

    return L.astype(np.float32), X_rgb.astype(np.float32)
