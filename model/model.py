import os
import math
import time
import sys
import csv

import numpy as np
from skimage import color

import tensorflow as tf
from keras import Sequential, Input, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from google.cloud import storage


# -----------------------------------------------------------------------------
#  Blocks de base
# -----------------------------------------------------------------------------
def downsample(filters, kernel_size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = Sequential()
    result.add(
        layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())
    return result


def upsample(filters, kernel_size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = Sequential()
    result.add(
        layers.Conv2DTranspose(
            filters,
            kernel_size=kernel_size,
            strides=(2, 2),
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


# -----------------------------------------------------------------------------
#  Generator U-Net (L 256×256×1 → AB 256×256×2)
# -----------------------------------------------------------------------------
def Generator(image_size=256):
    inputs = Input(shape=(image_size, image_size, 1))

    # down
    down_stack = [downsample(image_size, 3, apply_batchnorm=False)]
    n_filters = image_size * 2
    for _ in range(int(math.log2(image_size) - 3)):
        if n_filters >= 512:
            down_stack.append(downsample(512, 3))
        else:
            down_stack.append(downsample(n_filters, 3))
            n_filters *= 2

    # up
    n_filters = n_filters // 2
    up_stack = []
    for _ in range(int(math.log2(image_size) - 2)):
        if n_filters >= 64:
            up_stack.append(upsample(64, 3, apply_dropout=True))
        else:
            up_stack.append(upsample(n_filters, 3, apply_dropout=True))
            n_filters //= 2

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = layers.Conv2DTranspose(
        2,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",  # AB normalisés dans [-1, 1]
    )

    x = inputs
    skips = []

    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# -----------------------------------------------------------------------------
#  Losses
# -----------------------------------------------------------------------------
def mae(gen_output, target):
    return tf.reduce_mean(tf.abs(gen_output - target))


LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def Discriminator(image_size):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = tf.keras.layers.Input(
        shape=[image_size, image_size, 1], name="input_image"
    )
    tar = tf.keras.layers.Input(
        shape=[image_size, image_size, 2], name="target_image"
    )

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 3, False)(x)
    down2 = downsample(128, 3)(down1)
    down3 = downsample(256, 3)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(
        256, 3, strides=1, kernel_initializer=initializer, use_bias=False
    )(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(
        1, 3, strides=1, kernel_initializer=initializer
    )(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# def lab_to_rgb_batch(L_batch, AB_batch, max_images=4):
#     """
#     Convertit un batch (L normalisé, AB normalisés) en batch RGB pour TensorBoard.
#     L_batch : (N, H, W, 1)  dans [-1, 1]
#     AB_batch : (N, H, W, 2) dans [-1, 1]
#     Retour : (N, H, W, 3) float32 dans [0, 1]
#     """
#     N = min(max_images, L_batch.shape[0])

#     L = L_batch[:N, :, :, 0]          # (N, H, W)
#     AB = AB_batch[:N]                # (N, H, W, 2)

#     # dénormalisation L ∈ [0, 100]
#     L_rescaled = (L + 1.0) * 50.0

#     # dénormalisation AB ∈ [-128, 128]
#     AB_rescaled = AB * 128.0

#     rgbs = []
#     for i in range(N):
#         lab = np.stack(
#             [
#                 L_rescaled[i],
#                 AB_rescaled[i, :, :, 0],
#                 AB_rescaled[i, :, :, 1],
#             ],
#             axis=-1,
#         )  # (H, W, 3)

#         rgb = color.lab2rgb(lab)  # float dans [0, 1]
#         rgbs.append(rgb.astype("float32"))

#     return np.stack(rgbs, axis=0)  # (N, H, W, 3)
def lab_to_rgb_batch(L_batch, AB_batch, max_images=4):
    """
    Convertit un batch (L normalisé, AB normalisés) en batch RGB pour TensorBoard.

    L_batch : (N, H, W)      ou (N, H, W, 1) dans [-1, 1]
    AB_batch : (N, H, W, 2)  dans [-1, 1]

    Retour : (N, H, W, 3) float32 dans [0, 1]
    """
    # nombre d'images à convertir
    N = min(max_images, L_batch.shape[0])

    # Gérer 3D (N,H,W) et 4D (N,H,W,1)
    if L_batch.ndim == 4:
        L = L_batch[:N, :, :, 0]          # (N, H, W)
    elif L_batch.ndim == 3:
        L = L_batch[:N, :, :]             # (N, H, W)
    else:
        raise ValueError(f"L_batch doit être 3D ou 4D, reçu shape={L_batch.shape}")

    AB = AB_batch[:N, ...]               # (N, H, W, 2)

    # dénormalisation L ∈ [0, 100]
    L_rescaled = (L + 1.0) * 50.0

    # dénormalisation AB ∈ [-128, 128]
    AB_rescaled = AB * 128.0

    rgbs = []
    for i in range(N):
        lab = np.stack(
            [
                L_rescaled[i],
                AB_rescaled[i, :, :, 0],
                AB_rescaled[i, :, :, 1],
            ],
            axis=-1,
        )  # (H, W, 3)

        rgb = color.lab2rgb(lab)  # float dans [0, 1]
        rgbs.append(rgb.astype("float32"))

    return np.stack(rgbs, axis=0)  # (N, H, W, 3)


# -----------------------------------------------------------------------------
#  Boucle d'entraînement GAN
# -----------------------------------------------------------------------------
def train_step(
    input_image,
    target,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    step=None,
):
    """
    Un pas d'entraînement GAN.
    Pas de @tf.function ici pour garder les prints lisibles et éviter les soucis.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator(
            [input_image, gen_output], training=True
        )

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    return {
        "gen_total_loss": gen_total_loss,
        "gen_gan_loss": gen_gan_loss,
        "gen_l1_loss": gen_l1_loss,
        "disc_loss": disc_loss,
    }


def fit(
    train_ds,
    epochs,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    checkpoint=None,
    checkpoint_prefix=None,
    log_dir=None,
    sample_batch=None,
):
    """
    Entraînement GAN avec :
      - barre de progression
      - log CSV
      - log TensorBoard (losses + images générées)
    """

    # -------------------------
    # CSV log
    # -------------------------
    log_path = "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "batch", "gen_total_loss", "gen_gan_loss", "gen_l1_loss", "disc_loss"]
        )

    # -------------------------
    # TensorBoard writer
    # -------------------------
    writer = tf.summary.create_file_writer(log_dir) if log_dir is not None else None
    global_step = 0

    # nombre de batches total (si dataset batché)
    try:
        total_batches = len(train_ds)
    except TypeError:
        total_batches = None

    bar_width = 40

    for epoch in range(epochs):
        start = time.time()
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")

        for step, (input_image, target) in enumerate(train_ds):
            # 1. train step
            losses = train_step(
                input_image,
                target,
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                step,
            )

            # 2. CSV logging
            with open(log_path, "a", newline="") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow(
                    [
                        epoch + 1,
                        step + 1,
                        float(losses["gen_total_loss"]),
                        float(losses["gen_gan_loss"]),
                        float(losses["gen_l1_loss"]),
                        float(losses["disc_loss"]),
                    ]
                )

            # 3. TensorBoard scalars
            if writer is not None:
                with writer.as_default():
                    tf.summary.scalar("gen_total_loss", losses["gen_total_loss"], step=global_step)
                    tf.summary.scalar("gen_gan_loss", losses["gen_gan_loss"], step=global_step)
                    tf.summary.scalar("gen_l1_loss", losses["gen_l1_loss"], step=global_step)
                    tf.summary.scalar("disc_loss", losses["disc_loss"], step=global_step)
                global_step += 1

            # 4. barre de progression
            if total_batches is not None:
                progress = (step + 1) / total_batches
                filled = int(progress * bar_width)
                bar = "█" * filled + "-" * (bar_width - filled)

                sys.stdout.write(
                    f"\r[{bar}]  {step+1}/{total_batches}  "
                    f"gen_loss: {losses['gen_total_loss']:.4f}  "
                    f"disc_loss: {losses['disc_loss']:.4f}"
                )
                sys.stdout.flush()

        print(f"\nEpoch {epoch + 1} done in {time.time() - start:.2f}s")

        # 5. Images dans TensorBoard en fin d'epoch
        if writer is not None and sample_batch is not None:
            sample_L, _ = sample_batch
            gen_AB = generator(sample_L, training=False).numpy()
            L_np = sample_L.numpy()
            rgb_batch = lab_to_rgb_batch(L_np, gen_AB, max_images=4)

            with writer.as_default():
                tf.summary.image("generated_images", rgb_batch, step=epoch + 1)

        # 6. checkpoints
        if (
            checkpoint is not None
            and checkpoint_prefix is not None
            and (epoch + 1) % 10 == 0
        ):
            checkpoint.save(file_prefix=checkpoint_prefix)
            print("💾 Checkpoint saved")



# -----------------------------------------------------------------------------
#  Sauvegarde (local + optionnel GCS)
# -----------------------------------------------------------------------------
def save_model(model, model_filename="model_trained.keras", bucket_name=None):
    """
    Sauvegarde le modèle localement, et éventuellement sur GCS si bucket_name est fourni.
    """
    # sauvegarde locale
    os.makedirs(os.path.dirname(model_filename) or ".", exist_ok=True)
    model.save(model_filename)
    print(f"✅ Model saved locally at {model_filename}")

    # upload GCS optionnel
    if bucket_name:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"models/{os.path.basename(model_filename)}")
        blob.upload_from_filename(model_filename)
        print(
            f"✅ Model uploaded to GCS bucket '{bucket_name}' as '{blob.name}'"
        )


# -----------------------------------------------------------------------------
#  Entraînement "classique" Keras (non GAN) – laissé pour info
# -----------------------------------------------------------------------------
def train(generator, train_ds, val_ds, epochs=10):
    es = EarlyStopping(patience=5, verbose=0, restore_best_weights=True)
    chk = ModelCheckpoint(
        filepath="./checkpoints/checkpoint.model.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    history = generator.fit(
        train_ds, epochs=epochs, verbose=1, callbacks=[chk], validation_data=val_ds
    )
    return history
