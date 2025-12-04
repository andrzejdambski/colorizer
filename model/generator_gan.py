# model/generator_gan.py
"""
Générateur GAN pour le projet COLORIZER.

- Entrée  : L (64, 64, 1), normalisé dans [-1, 1]
- Sortie  : AB (64, 64, 2), normalisé dans [-1, 1]
- Architecture : U-Net (Pix2Pix-like)
- Utilisation :
      from model.generator_gan import build_generator_64x64
      gen = build_generator_64x64()
"""

import tensorflow as tf
from tensorflow.keras import layers


IMG_HEIGHT = 64
IMG_WIDTH = 64
IN_CHANNELS = 1    # L
OUT_CHANNELS = 2   # AB


# -------------------------------------------------------------------
# BLOCKS
# -------------------------------------------------------------------

def downsample(filters, size, apply_batchnorm=True):
    """
    Bloc encoder :
      Conv2D(stride=2) -> (BatchNorm) -> LeakyReLU
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(
            filters,
            kernel_size=size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    """
    Bloc decoder :
      Conv2DTranspose(stride=2) -> BatchNorm -> (Dropout) -> ReLU
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(
            filters,
            kernel_size=size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result


# -------------------------------------------------------------------
# GENERATOR
# -------------------------------------------------------------------

def build_generator_64x64():
    """
    Générateur U-Net pour la colorisation.

    Entrée :  (64, 64, 1)   -> L normalisé [-1,1]
    Sortie :  (64, 64, 2)   -> AB dans [-1,1]

    Retourne :
        tf.keras.Model
    """

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS), name="L_input")

    # ----- ENCODER -----
    down1 = downsample(64, 4, apply_batchnorm=False)(inputs)  # (32x32)
    down2 = downsample(128, 4)(down1)                         # (16x16)
    down3 = downsample(256, 4)(down2)                         # (8x8)
    down4 = downsample(512, 4)(down3)                         # (4x4)

    # ----- DECODER -----
    up1 = upsample(256, 4)(down4)                             # (8x8)
    up1 = layers.Concatenate()([up1, down3])

    up2 = upsample(128, 4)(up1)                               # (16x16)
    up2 = layers.Concatenate()([up2, down2])

    up3 = upsample(64, 4)(up2)                                # (32x32)
    up3 = layers.Concatenate()([up3, down1])

    up4 = upsample(32, 4)(up3)                                # (64x64)

    # ----- SORTIE -----
    initializer = tf.random_normal_initializer(0., 0.02)

    outputs = layers.Conv2D(
        OUT_CHANNELS,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",   # très important ! Normalisation [-1,1]
        name="AB_output",
    )(up4)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="generator_64x64")
    return model



# --------------------------------------------------------------------
# TEST RAPIDE EN AUTONOME
# --------------------------------------------------------------------

if __name__ == "__main__":
    print("🔧 Test du générateur...")
    gen = build_generator_64x64()
    gen.summary()

    dummy = tf.random.uniform((1, 64, 64, 1), minval=-1.0, maxval=1.0)
    out = gen(dummy)
    print("Sortie :", out.shape)
