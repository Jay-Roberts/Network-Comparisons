#---------------------------------------------------------------
#   
#   A collection of network blocks and other helper functions
#
#---------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


#-----------------------------------------------
#           Network Blocks
#-----------------------------------------------

# Basic convolutional layer
def basic_layer(input_layer, dt, scope):
    """
    Explicit Euler block with two branchs, stochastic and determinsitic,\
    both of which are two convolution layers.
    """

    # Compute Deterministic function
    fdd = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[5, 5], # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)
    
    fd = tf.layers.conv2d(
        inputs=fdd,
        filters=16,
        kernel_size=[5, 5], # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)

    return tf.add( input_layer, tf.add( tf.scalar_mul(dt,fd)))



# Residual Stochastic Convolutional Layer
# This corresponds to Strong Explicit Euler-Maruyama
def sNN_layer(input_layer, dt, scope):
    """
    Strong Explicit Euler-Maruyama block with two branchs, stochastic and determinsitic,\
    both of which are two convolution layers.
    """

    # Initialize Diagonal noise
    dz = tf.random_normal(tf.shape(input_layer))
    root_dt = tf.sqrt(dt)

    # Compute Stochastic function 
    # Need two layers for better approximation
    fss = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[5, 5], # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)

    fs = tf.layers.conv2d(
        inputs=fss,
        filters=16,
        kernel_size=[5, 5], # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)

    # Compute Deterministic function
    fdd = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[5, 5], # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)
    
    fd = tf.layers.conv2d(
        inputs=fdd,
        filters=16,
        kernel_size=[5, 5], # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)

    return tf.add( input_layer, tf.add( tf.scalar_mul(dt,fd) , tf.scalar_mul( root_dt, tf.multiply(fs,dz) ) ) )






#-----------------------------------------------
#           Input Functions
#-----------------------------------------------

# Helper function to get game tfrecords
def get_name(tag):
    
    game,name,file_dir = tag
    result = []
    path = file_dir+'/%s/%s'%(game,name)
    game_folders = os.listdir(path)
    
    for x in game_folders:
        result.append(path+'/'+x)
    
    return result

# joins lists of lists
def join_list(l):
    result = []
    for x in l:
        result+=x
    return result

# Reads the tf.Example files
def screen_shot_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    print('HERE', serialized_example)

    # The tfrecord to has an image and its label
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image'], tf.float32)

    # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
    image = tf.cast(image, tf.float32) / 255 - 0.5

    # Real small now
    image =  tf.reshape(image, [28, 28,3])

    # Get rid of label for now
    label = tf.cast(features['label'], tf.int32)
    return {"x":image,"y": label}


# The input function
def img_label_tfrecord_input_fn(name,file_dir = ['TFRecords'],
                    num_epochs = None,
                    shuffle = True,
                    batch_size = 100):
    """
    An input function expected tfrecord data with a feature for both image and label.
    """
    # Get games
    file_dir = file_dir[0]
    game_IDs = os.listdir(file_dir)
    num_games = len(game_IDs)
    files = [file_dir]*num_games
    
    # Get tfrecords
    if name == 'train':
        games = zip(game_IDs,['train']*num_games,files)
    if name == 'test':
        games = zip(game_IDs,['test']*num_games,files)
    if name == 'val':
        games = zip(game_IDs,['val']*num_games,files)

    filenames = list(map(get_name,games))
    filenames = join_list(filenames)
    

    # Import image data
    dataset = tf.data.TFRecordDataset(filenames)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size = batch_size)

    # Map the parser over dataset, and batch results by up to batch_size
    # Shuffling and batching can be slow
    # get more resources
    num_slaves = mp.cpu_count()
    dataset = dataset.map(my_parser,num_parallel_calls=num_slaves)
    
    # Buffer the batch size of data
    dataset = dataset.prefetch(batch_size)

    # Batch it and make iterator
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    
    features = iterator.get_next()

    return features

INPUT_FNS = {'img_label':img_label_tfrecord_input_fn}
