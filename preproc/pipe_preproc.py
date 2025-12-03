# importation des librairies
import numpy as np
import tensorflow as tf
import LAB as lab

# Import file
data = np.load(
    "/Users/ninadoinjashvili/code/andrzejdambski/colorizer/raw_data/cat_full_tf.npy"
)

# RGB to LAB
lab_data = lab.rgb_to_lab(data)

# Isolation des L et A_B et création de 2 datasets
L_dataset = lab_data[:, :, :, 0]
AB_dataset = lab_data[:, :, :, 1:]

# Datasets numpy conversion to TF ==> deux datasets TF
L_tf = tf.data.Dataset.from_tensor_slices(L_dataset)
AB_tf = tf.data.Dataset.from_tensor_slices(AB_dataset)

# Normalisation des L et A_B dans deux datasets normalisés
L_normalized = L_tf.map(lambda x: x / 50.0 - 1)
AB_normalized = AB_tf.map(lambda x: (x / 128.0))

# Préparation à l'entrainement en 32 tenseurs
batch_size = 32
L_normalized = L_normalized.batch(batch_size)
AB_normalized = AB_normalized.batch(batch_size)

# Conversions des datasets TF en arrays numpy et enregistrer
L_array = np.concatenate([batch.numpy() for batch in L_normalized], axis=0)
AB_array = np.concatenate([batch.numpy() for batch in AB_normalized], axis=0)

# Sauvegarder les fichiers dans le dossier raw_data
np.save("../../raw_data/L_normalized.npy", L_array)
np.save("../../raw_data/AB_normalized.npy", AB_array)

# Affichage des shapes des deux datasets npy enregistrés
print(L_array.shape, AB_array.shape)
