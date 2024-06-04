#!/usr/bin/env python

# To use a convnet on an arbitrarily-shaped tensor needs careful
#  attention to padding to make the sizes come out correctly.
# This is a test script - applying convolution layers to
#  a test tensor to see what sizes the outputs are, so I can
#  tweak the padding params to get it right.
# I ought to be able to work out the conv layer effects from
#  the documentation, but I haven't managed it.

import tensorflow as tf

# exec(open("interactive_test.mk4.py").read(), globals())

# Tensor of desired size to start with
imt = tf.zeros(shape=([1, 244, 180, 1]))

print("Starting!")

print()
print()
print("mk2")
imt = tf.zeros(shape=([1, 244, 180, 1]))
tf.print(tf.shape(imt))
step = tf.keras.layers.Conv2D(         filters=5,      kernel_size=3,      strides=(1, 1),     padding="same",     activation="elu",   )(imt)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=5,      kernel_size=3,      strides=(1, 1),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=10,     kernel_size=3,      strides=(2, 2),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=10,     kernel_size=3,      strides=(1, 1),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=10,     kernel_size=3,      strides=(1, 1),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=15,     kernel_size=3,      strides=(2, 2),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=15,     kernel_size=3,      strides=(1, 1),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=15,     kernel_size=3,      strides=(1, 1),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=20,     kernel_size=3,      strides=(2, 2),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=20,     kernel_size=3,      strides=(1, 1),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=20,     kernel_size=3,      strides=(1, 1),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=40,     kernel_size=3,      strides=(2, 2),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=40,     kernel_size=3,      strides=(1, 1),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=40,     kernel_size=3,      strides=(1, 1),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(         filters=80,     kernel_size=3,      strides=(2, 2),     padding="same",     activation="elu",   )(step)
tf.print(tf.shape(step))
# step = tf.keras.layers.Conv2D(         filters=40,     kernel_size=3,      strides=(1, 1),     padding="same",     activation="elu",   )(step)
# tf.print(tf.shape(step))
# step = tf.keras.layers.Conv2D(         filters=40,     kernel_size=3,      strides=(1, 1),     padding="same",     activation="elu",   )(step)
# tf.print(tf.shape(step))
# step = tf.keras.layers.Conv2D(         filters=80,     kernel_size=3,      strides=(2, 2),     padding="same",     activation="elu",   )(step)
# tf.print(tf.shape(step))
print("bottom")

tf.print(tf.shape(step))
# step = tf.keras.layers.Conv2DTranspose(  filters=40,     kernel_size=3,      strides=2,      padding="same",     output_padding=(1, 1),      activation="elu",   )(step)
# tf.print(tf.shape(step))
# step = tf.keras.layers.Conv2D(           filters=40,     kernel_size=3,      strides=(1, 1), padding="same",                                 activation="elu",   )(step)
# tf.print(tf.shape(step))
# step = tf.keras.layers.Conv2D(           filters=40,     kernel_size=3,      strides=(1, 1), padding="same",                                 activation="elu",   )(step)
# tf.print(tf.shape(step))
step = tf.keras.layers.Conv2DTranspose(  filters=40,     kernel_size=3,      strides=2,      padding="same",     output_padding=(1, 1),      activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(           filters=40,     kernel_size=3,      strides=(1, 1), padding="same",                                 activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(           filters=40,     kernel_size=3,      strides=(1, 1), padding="same",                                 activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2DTranspose(  filters=20,     kernel_size=3,      strides=2,      padding="same",     output_padding=(0, 0),      activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(           filters=20,     kernel_size=3,      strides=(1, 1), padding="same",                                 activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(           filters=20,     kernel_size=3,      strides=(1, 1), padding="same",                                 activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2DTranspose(  filters=15,     kernel_size=3,      strides=2,      padding="same",     output_padding=(0, 0),      activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(           filters=15,     kernel_size=3,      strides=(1, 1), padding="same",                                 activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(           filters=15,     kernel_size=3,      strides=(1, 1), padding="same",                                 activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2DTranspose(  filters=10,     kernel_size=3,      strides=2,      padding="same",     output_padding=(1, 1),      activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(           filters=10,     kernel_size=3,      strides=(1, 1), padding="same",                                 activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(           filters=10,     kernel_size=3,      strides=(1, 1), padding="same",                                 activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2DTranspose(  filters=5,      kernel_size=3,      strides=2,      padding="same", output_padding=(1, 1),                              )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(           filters=5,      kernel_size=3,      strides=(1, 1), padding="same",                                 activation="elu",   )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(           filters=1,      kernel_size=3,      strides=(1, 1), padding="same",                                 activation="elu",   )(step)
tf.print(tf.shape(step))




# exec(open('./interactive_test.mk4.py').read())
