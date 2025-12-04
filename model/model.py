# model/model.py
"""
Composants GAN pour le projet COLORIZER.

Ce fichier définit :
- build_discriminator_64x64 : discriminateur PatchGAN
- create_optimizers         : Adam pour G et D
- train_step                : une étape d'entraînement GAN

Le générateur est dans model/generator_gan.py et n'est PAS importé ici
pour éviter les imports circulaires. On passe le générateur en argument
à train_step.
"""

import tensorflow as tf
from tensorflow.keras import layers

IMG_HEIGHT = 64
IMG_WIDTH = 64
L_CHANNELS = 1
AB_CHANNELS = 2
LAMBDA_L1 = 100.0

# Binary cross-entropy pour des logits
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)



# --------------------------------------------------------------------
# DISCRIMINATEUR PATCHGAN 64x64
# --------------------------------------------------------------------
def build_discriminator_64x64() -> tf.keras.Model:
    """
    Discriminateur PatchGAN :
      - inputs: L (64,64,1) et AB (64,64,2)
      - concat -> (64,64,3)
      - output: carte de logits (~6x6x1)
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, L_CHANNELS), name="disc_L")
    tar = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, AB_CHANNELS), name="disc_AB")

    x = layers.Concatenate()([inp, tar])  # (64,64,3)

    # 64x64 -> 32x32
    down1 = layers.Conv2D(
        64, 4, strides=2, padding="same",
        kernel_initializer=initializer, use_bias=False
    )(x)
    down1 = layers.LeakyReLU(0.2)(down1)

    # 32x32 -> 16x16
    down2 = layers.Conv2D(
        128, 4, strides=2, padding="same",
        kernel_initializer=initializer, use_bias=False
    )(down1)
    down2 = layers.BatchNormalization()(down2)
    down2 = layers.LeakyReLU(0.2)(down2)

    # 16x16 -> 8x8
    down3 = layers.Conv2D(
        256, 4, strides=2, padding="same",
        kernel_initializer=initializer, use_bias=False
    )(down2)
    down3 = layers.BatchNormalization()(down3)
    down3 = layers.LeakyReLU(0.2)(down3)

    # 8x8 -> 7x7 (après ZeroPadding2D 10x10 puis conv 4x4 stride 1)
    zero_pad1 = layers.ZeroPadding2D()(down3)  # 10x10
    conv = layers.Conv2D(
        512, 4, strides=1,
        kernel_initializer=initializer, use_bias=False
    )(zero_pad1)                                # 7x7
    conv = layers.BatchNormalization()(conv)
    conv = layers.LeakyReLU(0.2)(conv)

    # 7x7 -> ~6x6 après ZeroPadding2D 9x9 puis conv 4x4
    zero_pad2 = layers.ZeroPadding2D()(conv)   # 9x9
    outputs = layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer
        # pas d'activation -> logits
    )(zero_pad2)                                # ~6x6x1

    return tf.keras.Model(inputs=[inp, tar], outputs=outputs, name="discriminator_64x64")


# -------------------------------------------------------------------
# PERTES
# -------------------------------------------------------------------
def discriminator_loss(real_logits, fake_logits):
    """Perte D : veut real→1, fake→0."""
    real_loss = loss_obj(tf.ones_like(real_logits), real_logits)
    fake_loss = loss_obj(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss


def generator_gan_loss(fake_logits):
    """Perte GAN de G : veut fake→1."""
    return loss_obj(tf.ones_like(fake_logits), fake_logits)


def generator_total_loss(fake_logits, real_AB, fake_AB, lambda_l1: float = LAMBDA_L1):
    """
    Perte totale du générateur = GAN + lambda * L1
    """
    gan = generator_gan_loss(fake_logits)
    l1 = tf.reduce_mean(tf.abs(real_AB - fake_AB))
    total = gan + lambda_l1 * l1
    return total, gan, l1


# -------------------------------------------------------------------
# OPTIMISEURS
# -------------------------------------------------------------------
def create_optimizers(lr: float = 2e-4, beta_1: float = 0.5):
    """Adam pour G et D (comme Pix2Pix)."""
    gen_opt = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
    disc_opt = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
    return gen_opt, disc_opt


# -------------------------------------------------------------------
# UNE ÉTAPE D'ENTRAÎNEMENT GAN
# -------------------------------------------------------------------
@tf.function
def train_step(generator,
               discriminator,
               gen_opt,
               disc_opt,
               L_batch,
               AB_batch):
    """
    Une étape d'entraînement GAN :
      - met à jour D
      - met à jour G

    L_batch : (batch,64,64,1)  L normalisé [-1,1]
    AB_batch: (batch,64,64,2)  AB normalisé [-1,1]
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        AB_fake = generator(L_batch, training=True)

        real_logits = discriminator([L_batch, AB_batch], training=True)
        fake_logits = discriminator([L_batch, AB_fake], training=True)

        gen_total, gen_gan, gen_l1 = generator_total_loss(
            fake_logits, AB_batch, AB_fake
        )
        disc_loss = discriminator_loss(real_logits, fake_logits)

    gen_grads = gen_tape.gradient(gen_total, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return {
        "gen_total": gen_total,
        "gen_gan": gen_gan,
        "gen_l1": gen_l1,
        "disc_loss": disc_loss,
    }
