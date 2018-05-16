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
    Inputs:
        tag: (game,name,file_dir) from screen_shot_input_fn. (str,str,str)
    Retruns:
        List of tfrecord files matching the game and name.
    """
    game,name,file_dir = tag
    path = '/'.join([file_dir,game,name])
    
    records = glob.glob(path+'*.tfrecords')    
    return records

# joins lists of lists
def join_list(l):
    """
    Joins lists.
    """
    result = []
    for x in l:
        result+=x
    return result

# Reads the tf.Example files
def screen_shot_parser(serialized_example,resolution):
    """Parses a single tf.Example into image and label tensors.
    Inputs:
        serialized_example: A tfrecord with features "image" and "label". (tfrecord)
        resoltuion: tuple resolution of image. (tuple)
    Returns:
        Dictionary with keys "x" and "y" for the image as a normalized tf.Tensor 
            and the label as tf.int32 respectively.
    """

    # The tfrecord to has an image and its label
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image'], tf.uint8)

    # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
    image = tf.cast(image, tf.float32) / 255 - 0.5
    

    # Real small now
    # resolution must be 3x32x32
    image =  tf.reshape(image, [list(resolution)[2],list(resolution)[0],list(resolution)[1]])
    image = tf.transpose(image,perm=[1,2,0])

    # No longer returning label
    label = tf.cast(features['label'], tf.int32)
    return {"x":image,"y":label}


# The input function
def screen_shot_input_fn(name,resolution,
                    file_dir = ['TFRecords'],
                    num_epochs = None,
                    shuffle = True,
                    batch_size = 100):
    """
    The input function for training, testing, and evaluation.
    Inputs:
        file_dir: Directory where the tfrecords are stored. Default is ['TFRecords']. (list)
        name: What mode is the model in. Must be from {'train','test','eval'}. (str)
        num_epochs: Number of repeats for dataset. (int)
        shuffle: Whether to shuffle input. Default True. (bool)
        batch_size: Batch size of input. Also the buffer size. Default 100. (int)
    """
    # Get games
    file_dir = file_dir[0]
        
    game_IDs = os.listdir(file_dir)
    num_games = len(game_IDs)
    files = [file_dir]*num_games
    names = [name]*num_games

    # Get tfrecords
    games = zip(game_IDs,names,files)
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
    dataset = dataset.map(lambda x: screen_shot_parser(x,resolution),num_parallel_calls=num_slaves)
    
    # Buffer the batch size of data
    dataset = dataset.prefetch(batch_size)

    # Batch it and make iterator
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(count=num_epochs)
    iterator = dataset.make_one_shot_iterator()
    
    features = iterator.get_next()

    return features

def cifar_input_fn(name,resolution,
                    file_dir = ['TFRecords'],
                    num_epochs = None,
                    shuffle = True,
                    batch_size = 100):
    """
    The input function for training, testing, and evaluation.
    Inputs:
        file_dir: Directory where the tfrecords are stored. Default is ['TFRecords']. (list)
        name: What mode is the model in. Must be from {'train','test','eval'}. (str)
        num_epochs: Number of repeats for dataset. (int)
        shuffle: Whether to shuffle input. Default True. (bool)
        batch_size: Batch size of input. Also the buffer size. Default 100. (int)
    """
    # Get games
    #file_dir = file_dir
    
    filenames = os.path.join(os.curdir , os.path.join(file_dir[0] , name + '.tfrecords'))
    #print("=================== filenames, ",filenames)
    # Import image data
    dataset = tf.data.TFRecordDataset(filenames)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size = batch_size)

    # Map the parser over dataset, and batch results by up to batch_size
    # Shuffling and batching can be slow
    # get more resources
    num_slaves = mp.cpu_count()
    print("=================resolution", resolution)
    dataset = dataset.map(lambda x: screen_shot_parser(x,resolution),num_parallel_calls=num_slaves)
    
    # Buffer the batch size of data
    dataset = dataset.prefetch(batch_size)

    # Batch it and make iterator
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(count=num_epochs)
    iterator = dataset.make_one_shot_iterator()
    
    features = iterator.get_next()
    print("=============features shape", features)

    return features


# Wrap up resolutions
def screen_shot_input_fn_28x28(name,file_dir = ['TFRecords'],
                    num_epochs = None,
                    shuffle = True,
                    batch_size = 100):
    return screen_shot_input_fn(name,(28,28,3),file_dir = file_dir,
                    num_epochs = None,
                    shuffle = shuffle,
                    batch_size = 100)

def screen_shot_input_fn_224x224(name,file_dir = ['TFRecords'],
                    num_epochs = None,
                    shuffle = True,
                    batch_size = 100):
    return screen_shot_input_fn(name,(224,224,3),file_dir = file_dir,
                    num_epochs = None,
                    shuffle = shuffle,
                    batch_size = 100)

def cifar(name,file_dir = ['cifar-10-data'],
                    num_epochs = None,
                    shuffle = True,
                    batch_size = 100):
    return cifar_input_fn(name,(32,32,3),file_dir = file_dir,
                    num_epochs = None,
                    shuffle = shuffle,
                    batch_size = 100)

# Dictionary of available input functions
INPUT_FNS= {(28,28,3):screen_shot_input_fn_28x28,
            (224,224,3):screen_shot_input_fn_224x224,
            (28,28,1):'mnist',
            (32,32,3):cifar}

# Training routines

def train_and_eval( data_dir,model_fn,model_dir,input_shape,
                        exp_dir,
                        train_steps=None,
                        train_epochs=None,
                        train_batch=100,
                        eval_steps=None,
                        eval_epochs=None,
                        eval_batch=100
                        ):
    """
    Train, evaluate, and export a saved model. For training and eval either steps or epochs \
    must be set or mode will run forever.
    Inputs:
        data_dir: Where the data is stored. Must be accessible by input_fn. 
                    For mnist set to None. (str)
        model_fn: The model function to use to construct estimator.
        input_shape: Image resolution to use in model. (tuple)
        exp_dir: Directory to export checkpoints and saved model. (str)
        train_steps: Number of steps to train for. If None train_epochs must be. (int)
        train_epochs: Number of epochs to run through training data. If None train_steps must be. (int)
        train_batch: Batch size to use for training. Default 100. (int)
        eval_steps: Number of steps to eval for. If None eval_epochs must be. (int)
        eval_epochs: Number of epochs to run through evaling data. If None eval_steps must be. (int)
        eval_batch: Batch size to use for evaling. Default 100. (int)
    Returns:
        None
    
    """
    # Get input function
    input_fn = INPUT_FNS[input_shape]


    # Construct the classifier
    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=model_dir)
    

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=500)



    # Make training and evaluation input functions
    # MNIST input function downloads the data here
    if list(input_shape) == [28,28,1]:
        # Load training and eval data
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images  # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images  # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        train_input = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data, "y": train_labels},
            batch_size=train_batch,
            num_epochs=train_epochs,
            shuffle=True)
        
        eval_input = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data, "y": eval_labels},
            batch_size=eval_batch,
            num_epochs=eval_epochs,
            shuffle=False)
        
        # MNIST doesn't have labels as input so its
        # feature spec is different
        feature_spec = {"x":tf.placeholder(dtype=tf.float32,shape = [28,28,1]),
                        "y": tf.placeholder(dtype=np.int32,shape = [1])}

    # Handle non mnist data
    else:
        # Get the train input function
        train_input = lambda: input_fn('train',
            file_dir = [data_dir],
            num_epochs=train_epochs,
            batch_size=train_batch)

        # Don't shuffle evaluation data
        eval_input = lambda: input_fn('eval', 
            file_dir=[data_dir],
            batch_size=eval_batch,
            num_epochs=eval_epochs,
            shuffle=False)
        
        # Make the feature spec for exporting
        feature_spec = {"x":tf.placeholder(dtype=tf.float32,shape = input_shape),
                        "y":tf.placeholder(dtype = tf.int32,shape = [1])}

    #Train the model
    classifier.train(train_input,
                    steps=train_steps,
                    )

    # Evaluate the model
    eval_name = 'eval'
    classifier.evaluate(eval_input,
                        steps = eval_steps)
    
    # Export the model
    classifier.export_savedmodel(exp_dir,
                        tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
                        )