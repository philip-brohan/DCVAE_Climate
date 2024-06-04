#!/usr/bin/env python

# To use a convnet on an arbitrarily-shaped tensor needs careful
#  attention to padding to make the sizes come out correctly.
# This is a test script - applying convolution layers to
#  a test tensor to see what sizes the outputs are, so I can
#  tweak the padding params to get it right.
# I ought to be able to work out the conv layer effects from
#  the documentation, but I haven't managed it.

import tensorflow as tf

# Tensor of desired size to start with
imt = tf.zeros(shape=([1, 244, 180, 1]))

print(" ")
tf.print(tf.shape(imt))

step = tf.keras.layers.Conv2D(
    filters=5,
    kernel_size=3,
    strides=(2, 2),
    padding="same",
)(imt)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(
    filters=10,
    kernel_size=3,
    strides=(2, 2),
    padding="same",
)(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(
    filters=10,
    kernel_size=3,
    strides=(2, 2),
    padding="same",
)(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2D(
    filters=20,
    kernel_size=3,
    strides=(2, 2),
    padding="same",
)(step)
# tf.print(tf.shape(step))
# step = tf.keras.layers.Conv2D(
#     filters=40,
#     kernel_size=3,
#     strides=(2, 2),
#     padding="same",
# )(step)
# tf.print(tf.shape(step))
# step = tf.keras.layers.Conv2D(
#     filters=40,
#     kernel_size=3,
#     strides=(2, 2),
#     padding="same",
# )(step)

# tf.print(tf.shape(step))
# step = tf.keras.layers.Conv2DTranspose(
#     filters=40, kernel_size=3, strides=2, padding="same", output_padding=(1, 1)
# )(step)
# tf.print(tf.shape(step))
# step = tf.keras.layers.Conv2DTranspose(
#     filters=20, kernel_size=3, strides=2, padding="same", output_padding=(1, 1)
# )(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2DTranspose(
    filters=10, kernel_size=3, strides=2, padding="same", output_padding=(0, 0)
)(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2DTranspose(
    filters=10, kernel_size=3, strides=2, padding="same", output_padding=(0, 0)
)(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2DTranspose(
    filters=5, kernel_size=3, strides=2, padding="same", output_padding=(1, 1)
)(step)
tf.print(tf.shape(step))
step = tf.keras.layers.Conv2DTranspose(
    filters=1, kernel_size=3, strides=2, padding="same", output_padding=(1, 1)
)(step)
tf.print(tf.shape(step))


# exec(open('./interactive_test.rlevA.py').read())
