import tensorflow as tf
import numpy as np
from keras import Sequential,Input,layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math
import time


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


LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[64, 64, 2], name='input_image')
  tar = tf.keras.layers.Input(shape=[64, 64, 2], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 3, False)(x)  # (batch_size, 128, 128, 64) -> 32,32,64
  down2 = downsample(128, 3)(down1)  # (batch_size, 64, 64, 128) -> 16,16,64
  down3 = downsample(256, 3)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256) -> 18,18,64
  conv = tf.keras.layers.Conv2D(256, 3, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512) -> 16,16,64

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 3, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1) -> 16,16,64

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
    
def fit(train_ds, test_ds, steps):
    
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

    #   generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)


    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
      

def train(generator, train_ds, val_ds):
    
    es = EarlyStopping(patience=5, verbose=0, restore_best_weights=True)
    chk = ModelCheckpoint(filepath = './checkpoints/checkpoint.model.keras',monitor='val_loss',\
        save_best_only=True,mode='min')
    history = generator.fit(train_ds, epochs=10,verbose=1,callbacks=[chk], validation_data=val_ds)
    
    return history
