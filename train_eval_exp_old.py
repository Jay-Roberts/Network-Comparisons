from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

import argparse
import os

import model_fns as mf
import support_fns as spf

def run_experiment(hparams):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    model_fn = mf.Model_Functions[hparams.model_fn]

    model_dir = '/'.join([hparams.save_dir,hparams.model_fn])
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn, 
        model_dir=model_dir)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=500)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=hparams.train_batch_size,
        num_epochs=hparams.num_epochs,
        shuffle=True)
    
    # Run Training
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=hparams.train_steps)
        #hooks=[logging_hook])


    # Evaluate the model and print results
    # Save where to put and look for checkpoints
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1, # what is this for?
        shuffle=False)
    
    # Run evaluation
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn,
    steps=hparams.eval_steps)
    print(eval_results)

    # Export the saved model
    # Dictionary of features.
    feature_spec = {"x":tf.placeholder(dtype=tf.float32,shape = [28,28,1])}
    
    # Where to save it

    mnist_classifier.export_savedmodel(model_dir,
                                tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
                                )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments

    # Type of model
    parser.add_argument(
        '--model-fn',
        help='The model in trainer file to run task on',
        type=str,
        choices=['Test','SRNN_2','SRNN_10','SRNN_20'],
        default='Test'
    )

    # Where to save variables
    parser.add_argument(
        '--save-dir',
        help='Where to save experiment information',
        type=str,
        default='Testit'
    )
    
    # Training and eval parameters
    parser.add_argument(
        '--num-epochs',
        help="""\
        Maximum number of training data epochs on which to train.
        If both --max-steps and --num-epochs are specified,
        the training job will run for --max-steps or --num-epochs,
        whichever occurs first. If unspecified will run for --max-steps.\
        """,
        type=int,
    )
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=40
    )

    parser.add_argument(
        '--train-steps',
        help="""
        Steps to run the training job for. If --num-epochs is not specified,
        this must be. Otherwise the training job will run indefinitely.\
        """,
        #type=int,
        #default=20
    )
    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint.',
        default=100,
        type=int
    )


    # Argument to turn on all logging
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )
    

    args = parser.parse_args()


    # Make save dir if needed
    if args.save_dir:
        print('Saving in save-dir: %s'%(args.save_dir))
        if not os.path.isdir(args.save_dir):
            print('Creating save-dir: %s '%(args.save_dir))
            os.mkdir(args.save_dir)
        


    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    # Run the training job
    hparams=hparam.HParams(**args.__dict__)
    run_experiment(hparams)