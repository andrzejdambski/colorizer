import tensorflow as tf

# X c'est L et  y c'est AB

def normalize_L_AB(X, y):
    """
    Normalise L et AB pour le projet Colorizer.

    X : canal L (H, W, 1), valeurs dans [0,100]
    y : canaux AB (H, W, 2), valeurs dans [-128, 127]

    Retour :
    - X ∈ [0,1]
    - y ∈ [-1,1]
    """

    X = tf.cast(X, tf.float32) / 50  - 1    # L → [-1,1]
    y = tf.cast(y, tf.float32) / 128.0      # AB → [-1,1]

    return X, y

"""
On utilise de cette manière dans notre dataset :
dataset = dataset.map(normalize_L_AB)
"""

# Fonction inverse on veut retransformer pour afficher les couleurs
def denormalize_L_AB_np(L, AB):
    L = L * 100.0
    AB = AB * 128.0
    return L, AB
