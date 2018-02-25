#---------------------------------------------------------------
#   
#   A collection of network blocks
#
#---------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# Basic convolutional layer
def f_E(input_layer, dt, shape, scope):
    """
    Explicit Euler block with two convolutional layers then a residual shortcut.
    dt: Step size. (float)
    shape: NSize of square kernel to use (int), umber of filters (int). (list)
    """

    size, filters = shape
    # Compute Deterministic function
    fdd = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)
    
    fd = tf.layers.conv2d(
        inputs=fdd,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)

    return tf.add( input_layer, tf.add( tf.scalar_mul(dt,fd)))



# Residual Stochastic Convolutional Layer
# This corresponds to Strong Explicit Euler-Maruyama
def Stf_EM(input_layer, dt, shape, scope):
    """
    Explicit Euler Maruyama block with two  non-interactign branches, stochastic and deterministic\
    each has 2 convolutional layers then a residual shortcut.
    dt: Step size. (float)
    shape: NSize of square kernel to use (int), umber of filters (int). (list)
    """

    size, filters = shape
    # Initialize Diagonal noise
    dz = tf.random_normal(tf.shape(input_layer))
    root_dt = tf.sqrt(dt)

    # Compute Stochastic function 
    # Need two layers for better approximation
    fss = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)

    fs = tf.layers.conv2d(
        inputs=fss,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)

    # Compute Deterministic function
    fdd = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)
    
    fd = tf.layers.conv2d(
        inputs=fdd,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)

    return tf.add( input_layer, tf.add( tf.scalar_mul(dt,fd) , tf.scalar_mul( root_dt, tf.multiply(fs,dz) ) ) )



# Dictionary of available blocks
BLOCKS = {'f_E':f_E,
           'Stf_EM': Stf_EM}
