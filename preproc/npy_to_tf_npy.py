import tensorflow as tf
import numpy as np

#Charger le fichier numpy et le convertir en tableau numpy de type float32
numpy_array = np.load('cats_64x64_full.npy')
data_np = np.asarray(numpy_array, np.float32)

#Convertir le tableau numpy en tenseur TensorFlow
tensor_resultat = tf.convert_to_tensor(numpy_array)

#Sauvegarder le tenseur TensorFlow en tant que fichier numpy
resultats_np = tensor_resultat.numpy()
np.save('tf_dataset_cat_64x64_full.npy', resultats_np)

#Afficher la forme du tenseur résultant
print(tensor_resultat.shape)
