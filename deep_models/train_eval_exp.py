#---------------------------------------------------------------
#   
#   Functions for training, evaluating, and exporting models
#
#---------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import glob
import os
import tensorflow as tf
import multiprocessing as mp

#-----------------------------------------------
#           Input Functions
#-----------------------------------------------

# Helper function to get game tfrecords
def get_name(tag):
    """
    Get the tfrecords from the file_dir. Files are saved as file_dir/game/name_x.tfrecords
    tag: (game,name,file_dir) from screen_shot_input_fn. 
    """
    game,name,file_dir = tag
    path = '/'.join([file_dir,game,name])
    print(path)
    
    records = glob.glob(path+'*.tfrecords')    
    print('recs: ',records)
    return records

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

    # No longer returning label
    label = tf.cast(features['label'], tf.int32)
    return {"x":image,"y":label}


# The input function
def screen_shot_input_fn(name,file_dir = ['TFRecords'],
                    num_epochs = None,
                    shuffle = True,
                    batch_size = 100):
    """
    The input function for training, testing, and evaluation.
    
    file_dir: Directory where the tfrecords are stored. Default is ['TFRecords']. (list)
    name: What mode is the model in. Must be from {'train','test','eval'}. (str)
    num_epochs: Number of epochs. (int)
    shuffle: Whether to shuffle input. Default True. (bool)
    batch_size: Batch size of input. Also the buffer size. Default 100. (int)
    """
    # Get games
    file_dir = file_dir[0]
        
    game_IDs = os.listdir(file_dir)
    print(game_IDs)
    num_games = len(game_IDs)
    files = [file_dir]*num_games
    names = [name]*num_games

    # Get tfrecords
    games = zip(game_IDs,names,files)
    filenames = list(map(get_name,games))
    filenames = join_list(filenames)
    print('files found: ' ,filenames)

    # Import image data
    dataset = tf.data.TFRecordDataset(filenames)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size = batch_size)

    # Map the parser over dataset, and batch results by up to batch_size
    # Shuffling and batching can be slow
    # get more resources
    num_slaves = mp.cpu_count()
    dataset = dataset.map(screen_shot_parser,num_parallel_calls=num_slaves)
    
    # Buffer the batch size of data
    dataset = dataset.prefetch(batch_size)

    # Batch it and make iterator
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    
    features = iterator.get_next()

    return features

# Dictionary of available input functions
INPUT_FNS= {'screen_shots':screen_shot_input_fn}