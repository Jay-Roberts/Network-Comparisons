#---------------------------------------------------------------
#   
#   A collection of network blocks
#
#---------------------------------------------------------------

import tensorflow as tf

# Vanilla convolutional layer
def van(input_layer, dt, shape, scope):
    """
    Basic convolutional layer used for comparison purposes.
    dt: *NOT USED* Step size. (float)
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

    return fd

# Residual convolutional layer
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
        activation=None)
    

    bn = tf.layers.conv2d(
        fdd,
        filters,
        size,
        padding='same',
        activation=tf.nn.relu

    )

    fd = tf.layers.conv2d(
        inputs=bn,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)

    return tf.add( input_layer,  tf.scalar_mul(dt,fd))



# Residual Stochastic Convolutional Layer
# This corresponds to Strong Explicit Euler-Maruyama
def Sf_EM(input_layer, dt, shape, scope,name):
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
        activation=tf.nn.relu,
        reuse=tf.AUTO_REUSE,
        name=name)

    fs = tf.layers.conv2d(
        inputs=fss,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu,
        reuse=tf.AUTO_REUSE,
        name=name)

    # Compute Deterministic function
    fdd = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu,
        reuse=tf.AUTO_REUSE,
        name=name)
    
    fd = tf.layers.conv2d(
        inputs=fdd,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu,
        reuse=tf.AUTO_REUSE,
        name=name)

    return tf.add( input_layer, tf.add( tf.scalar_mul(dt,fd) , tf.scalar_mul( root_dt, tf.multiply(fs,dz) ) ) )

# Residual Stochastic Convolutional Layer
# This corresponds to Weak Explicit Euler-Maruyama
def Wf_EM(input_layer, dt, shape, scope='Deep',name='weak_stochastic'):
    """
    Explicit Euler Maruyama block with two  non-interactign branches, stochastic and deterministic\
    each has 2 convolutional layers then a residual shortcut.
    dt: Step size. (float)
    shape: NSize of square kernel to use (int), umber of filters (int). (list)
    """

    size, filters = shape
    #filters=16
    #size=shape
    # Initialize Diagonal noise
    dz = tf.random_normal(tf.shape(input_layer))
    dz = tf.sign(dz)
    dz += 1
    dz = dz*0.5

    root_dt = tf.sqrt(dt)

    # Compute Stochastic function 
    # Need two layers for better approximation
    fss = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=None,
        reuse=tf.AUTO_REUSE,
        name=name)

    bns = tf.layers.conv2d(
        fss,
        filters,
        size,
        padding='same',
        activation=tf.nn.relu

    )

    fs = tf.layers.conv2d(
        inputs=bns,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu,
        reuse=tf.AUTO_REUSE,
        name=name)

    # Compute Deterministic function
    fdd = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=None,
        reuse=tf.AUTO_REUSE,
        name=name)

    bnd = tf.layers.conv2d(
        fdd,
        filters,
        size,
        padding='same',
        activation=tf.nn.relu

    )
    
    fd = tf.layers.conv2d(
        inputs=bnd,
        filters=filters,
        kernel_size=size, # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu,
        reuse=tf.AUTO_REUSE,
        name=name)

    return tf.add( input_layer, tf.add( tf.scalar_mul(dt,fd) , tf.scalar_mul( root_dt, tf.multiply(fs,dz) ) ) )


# Dictionary of available blocks
BLOCKS = {
    'van': van,
    'f_E':f_E,
    'Sf_EM': Sf_EM,
    'Wf_EM': Wf_EM
        }
