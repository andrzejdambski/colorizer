import tensorflow as tf
import numpy as np
from keras import Sequential,Input,layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math


def downsample(filter,kernel_size,apply_batchnorm=True):

    intitializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential()
    result.add(layers.Conv2D(filter,kernel_size=kernel_size,strides=(2,2),padding='same',\
        kernel_initializer=intitializer,use_bias=False))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())

    return result

def upsample(filter,kernel_size,apply_dropout=False):

    intitializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential()
    result.add(layers.Conv2DTranspose(filter,kernel_size=kernel_size,strides=(2,2),padding='same',\
        kernel_initializer=intitializer,use_bias=False))

    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result

def Generator(image_size=64):

    inputs = Input(shape=(image_size,image_size,1))

    down_stack = [downsample(image_size,3,apply_batchnorm=False)]
    n_filters = image_size*2
    for n in range(int(math.log2(image_size)-3)):
        if n_filters >= 512:
            down_stack.append(downsample(512,3))
        else:
            down_stack.append(downsample(n_filters,3))
            n_filters*=2

    n_filters=n_filters/2
    up_stack = []
    for n in range(int(math.log2(image_size)-2)):
        if n_filters >= 64:
            up_stack.append(upsample(64,3,apply_dropout=True))
        else:
            up_stack.append(upsample(n_filters,3,apply_dropout=True))
            n_filters=n_filters/2

    intitializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(2,kernel_size=3,strides=(2,2),padding='same',\
        kernel_initializer=intitializer,activation='tanh')

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


generator = Generator()

def mae(gen_output,target):

    l1_loss = tf.reduce_mean(tf.abs(gen_output-target))
    return l1_loss

def train(generator, train_ds, val_ds):
    
    es = EarlyStopping(patience=5, verbose=0, restore_best_weights=True)
    chk = ModelCheckpoint(filepath = './checkpoints/checkpoint.model.keras',monitor='val_loss',\
        save_best_only=True,mode='min')
    history = generator.fit(train_ds, epochs=10,verbose=1,callbacks=[chk], validation_data=val_ds)
    
    return history
