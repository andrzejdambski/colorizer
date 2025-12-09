import os
import tensorflow as tf
import datetime

from preproc.preproc import get_list_of_paths, preprocess
from model.model import Generator, Discriminator, mae, fit, save_model


tf.config.set_visible_devices([], 'GPU')
print("Devices visibles :", tf.config.list_physical_devices())

# -----------------------------
# Paramètres
# -----------------------------
DATA_DIR = "./raw_data/catsdata"
BATCH_SIZE = 32
IMAGE_SIZE = 256
EPOCHS = 20  # augmenter à 10, 20… si besoins

# -----------------------------
# Dataset
# -----------------------------
jpg_paths, _ = get_list_of_paths(DATA_DIR)
print(f"🖼️  {len(jpg_paths)} images trouvées dans {DATA_DIR}")

ds = tf.data.Dataset.from_tensor_slices(jpg_paths)
ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print("✅ Dataset prêt (LAB normalisé 256×256)")

split_ratio = 0.2
test_len = int(len(ds) * split_ratio)
test_ds = ds.take(test_len)
train_ds = ds.skip(test_len)
print(f"✅ Split train/test fait (≈ {int((1-split_ratio)*100)}% / {int(split_ratio*100)}%)")

# -----------------------------
# Modèles + optimizers
# -----------------------------
generator = Generator(IMAGE_SIZE)
generator.compile(loss=mae, optimizer="adam")
print("✅ Generator créé")

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator = Discriminator(image_size=IMAGE_SIZE)
print("✅ Discriminator créé")

checkpoint_dir = "./training_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

# -----------------------------
# TensorBoard log dir + batch de samples
# -----------------------------
log_dir = f"logs/gan_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
sample_batch = next(iter(test_ds))  # (L_batch, AB_batch)

# -----------------------------
# Entraînement GAN
# -----------------------------

fit(
    train_ds,
    EPOCHS,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    checkpoint,
    checkpoint_prefix,
    log_dir=log_dir,
    sample_batch=sample_batch,
)

# -----------------------------
# Sauvegarde du modèle pour l'API FastAPI
# -----------------------------
MODEL_PATH = "model_trained.keras"
save_model(generator, MODEL_PATH, bucket_name=None)  # local uniquement
