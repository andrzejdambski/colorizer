import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
from skimage import color
import tensorflow as tf
from pathlib import Path


def get_list_of_paths(path_to_data):
    """
    Returns a tuple, two lists. The first is a list with file names of jpg files
    The other is a list with file names of .cat files
    You need to have cats folder in your directory
    """

    p = Path(path_to_data)
    jpgs = sorted(list(p.glob("**/*.jpg")))
    cats = sorted(list(p.glob("**/*.cat")))

    # Convert PosixPath objects to strings
    jpgs = [str(p) for p in jpgs]
    cats = [str(c) for c in cats]

    return jpgs, cats


def zoom_on_cat_face(jpg_path, cat_path, image_size=64, data_augmentation=False):
    """
    Takes the file paths for a .jpg and the .cat associated,
    scales, rotates and crops the image to a IMAGE SIZE of a centered cat's head.
    Returns: np array
    """
    # Convert tf.string tensors to Python strings if needed
    if isinstance(jpg_path, tf.Tensor):
        jpg_path = jpg_path.numpy().decode()
    if isinstance(cat_path, tf.Tensor):
        cat_path = cat_path.numpy().decode()

    # --- Load image as PIL (safe in eager) ---
    img_pil = Image.open(jpg_path).convert("RGB")

    # --- Load .cat file as string ---
    with open(cat_path, "r") as f:
        cat_cat = f.read()

    # --- Parse eyes coordinates ---
    cat_leye = [int(x) for x in cat_cat.split()[1:3]]
    cat_reye = [int(x) for x in cat_cat.split()[3:5]]

    # output position of the left eye
    leye_loc = [(18 / 64) * image_size, (30 / 64) * image_size]

    # --- scale ---
    distance_eyes = math.sqrt(
        (cat_reye[1] - cat_leye[1]) ** 2 + (cat_reye[0] - cat_leye[0]) ** 2
    )
    distance_output = image_size - 2 * leye_loc[0]
    if distance_output == 0:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    scale = distance_eyes / distance_output
    a = scale
    b = 0
    c = ((1 - scale) * img_pil.size[0]) / 2
    d = 0
    e = scale
    f = ((1 - scale) * img_pil.size[1]) / 2
    img_scaled = img_pil.transform(
        img_pil.size, Image.AFFINE, (a, b, c, d, e, f), Image.Resampling.BILINEAR
    )

    # --- rotation ---
    if (cat_reye[0] - cat_leye[0]) == 0:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    sign = 1 if cat_reye[1] > cat_leye[1] else -1
    rotation = math.atan((cat_reye[1] - cat_leye[1]) / (cat_reye[0] - cat_leye[0])) * (
        180 / math.pi
    )
    new_leye_center = (
        img_pil.size[0] / 2 - 1 / scale * (img_pil.size[0] / 2 - cat_leye[0]),
        img_pil.size[1] / 2 - 1 / scale * (img_pil.size[1] / 2 - cat_leye[1]),
    )
    img_rotated = img_scaled.rotate(
        sign * rotation, center=new_leye_center, resample=Image.Resampling.BILINEAR
    )

    # --- crop ---
    cat_c = img_rotated.crop(
        (
            new_leye_center[0] - leye_loc[0],
            new_leye_center[1] - leye_loc[1],
            new_leye_center[0] + (image_size - leye_loc[0]),
            new_leye_center[1] + (image_size - leye_loc[1]),
        )
    )

    return np.array(cat_c, dtype=np.uint8)


def visualising_an_image(cat_number, path_to_data):
    """
    cat_number est un numero de chat, changer le pour visualiser un chat diff
    """
    jpg, cat = get_list_of_paths(path_to_data)
    cat_image = Path(jpg[cat_number])
    file_cat = Path(cat[cat_number])
    image = zoom_on_cat_face(cat_image, file_cat, 128)
    return plt.imshow(image)


def rgb_to_lab(rgb):
    """
    convertit de rgb en lab. Stateless transformation
    """
    return color.rgb2lab(rgb).astype(np.float32)


def rgb_to_lab_tf(rgb):
    """
    adapte fonction rgb_to_lab a un tensor.
    """
    a = tf.numpy_function(func=rgb_to_lab, inp=[rgb], Tout=tf.float32)
    return a


def zoom_on_cat_face_tf(jpg_path, cat_path):
    """
    adapte la fonction zoom_on_cat_face a un tensor
    """
    a = tf.py_function(func=zoom_on_cat_face, inp=[jpg_path, cat_path], Tout=tf.uint8)
    return a


def preprocess(jpg_path, cat_path):
    """
    depuis un path, zoom sur la tete du chat
    Change le format au LAB
    Change le format de la donnee pour qu'elle aie la bonne taille
    Isole le L du AB
    Normalise le L et le AB
    Retourne le L et le AB
    """
    img_np = zoom_on_cat_face_tf(jpg_path, cat_path)

    img_lab = rgb_to_lab_tf(img_np)

    img_lab.set_shape((64, 64, 3))

    L = img_lab[..., 0]
    AB = img_lab[..., 1:]

    L_norm = L / 50 - 1
    AB_norm = AB / 128

    return L_norm, AB_norm


def get_dataset(batch_number, path_to_data, shuffle=False, seed=42):
    """
    Builds the batched tf.data pipeline for all samples.
    Optionally shuffles the file list before mapping.
    """
    liste_de_jpgs, liste_de_cat = get_list_of_paths(path_to_data)
    ds = tf.data.Dataset.from_tensor_slices((liste_de_jpgs, liste_de_cat))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(liste_de_jpgs),
            seed=seed,
            reshuffle_each_iteration=False,
        )
    dataset = (
        ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_number)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def get_train_test_datasets(batch_number, path_to_data, test_ratio=0.2, seed=42):
    """
    Returns (train_ds, test_ds) with a deterministic split on the file list.
    Splits on the raw file list (not the batched dataset) to keep item counts aligned.
    """
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    liste_de_jpgs, liste_de_cat = get_list_of_paths(path_to_data)
    ds = tf.data.Dataset.from_tensor_slices((liste_de_jpgs, liste_de_cat))
    ds = ds.shuffle(
        buffer_size=len(liste_de_jpgs),
        seed=seed,
        reshuffle_each_iteration=False,
    )

    test_len = int(len(liste_de_jpgs) * test_ratio)
    test_ds = (
        ds.take(test_len)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_number)
        .prefetch(tf.data.AUTOTUNE)
    )
    train_ds = (
        ds.skip(test_len)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_number)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds
