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
from model.model import Generator, Discriminator, mae, generator_loss, train, train_step, fit, save_model
import math
from keras.models import save_model, load_model
from google.cloud import storage

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 
# "/Users/andrzej/.config/gcloud/application_default_credentials.json"

# url = 'gs://colorizer/'
# url = 'gs://catsdata/'
BUCKET_NAME = 'catsdata'
url = './raw_data/catsdata/'

l_jpg = getting_file_names()
# l_cat = getting_file_names('cat')

l_jpg = [url+jpg for jpg in l_jpg]
# l_cat = [url+cat for cat in l_cat]

ds = tf.data.Dataset.from_tensor_slices((l_jpg))
ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)
print(next(iter(ds)))
print(f'\n ✅ mapable dataset created')


split_ratio = 0.2
test_len = int(len(ds)*split_ratio)
test_ds = ds.take(test_len)
train_ds = ds.skip(test_len)
print(f'\n ✅ train test split done')


image_size = 256
generator = Generator(image_size)
print(f'\n ✅ generator done')

generator.compile(loss=mae,optimizer='adam')

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator = Discriminator(image_size=image_size)
print(f'\n ✅ discriminator done')

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

epochs = 10
fit(train_ds,epochs,discriminator,generator_optimizer,discriminator_optimizer,checkpoint,checkpoint_prefix)

model_filename = 'model_trained.keras'    
save_model(generator,model_filename,BUCKET_NAME)
# # history = train(generator,train_ds,test_ds,epochs=epochs)

# save_dir = '/opt/colorizer/'
# save_dir_1 = '/model_trained/'
# path = os.path.join(save_dir_1, f"model_final.keras")
# history.save(path)
# print(f"✅ Modèle sauvegardé : {path}")
