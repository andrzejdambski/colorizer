import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import os
import datetime
import requests
from preproc.api import getting_file_names
from preproc.preproc import get_dataset, get_list_of_paths, preprocess
import time
from keras import Sequential,Input,layers
from keras.callbacks import EarlyStopping
from model.model import Generator, mae, train
import math

# url = 'gs://colorizer/'
url = 'gs://nina-cats-data/'

l_jpg = getting_file_names()
# l_cat = getting_file_names('cat')

l_jpg = [url+jpg for jpg in l_jpg]
# l_cat = [url+cat for cat in l_cat]

ds = tf.data.Dataset.from_tensor_slices((l_jpg))
ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)
print(next(iter(ds)))

split_ratio = 0.2
test_len = int(len(ds)*split_ratio)
test_ds = ds.take(test_len)
train_ds = ds.skip(test_len)

image_size = 256
generator = Generator(image_size)

generator.compile(loss=mae,optimizer='adam')

history = train(generator,train_ds,test_ds)

save_dir = '/opt/colorizer/'
save_dir_1 = '/model_trained/'
path = os.path.join(save_dir_1, f"model_final.keras")
history.save(path)
print(f"✅ Modèle sauvegardé : {path}")