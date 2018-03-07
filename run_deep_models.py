# Run deep_models experiments

from deep_models import models
import numpy as np
import tensorflow as tf 
import os
import parser
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments

    # Model Specs
    parser.add_argument(
        '--block',
        help='Block to use in deep layer',
        type=str,
        required=True,
        choices=['van',
        'f_E',
        'Stf_EM'])
    
    parser.add_argument(
        '--depth',
        help='Depth of model',
        type=int,
        required=True)
    
    parser.add_argument(
        '--resolution',
        help='reolution of images in tfrecords files',
        type=int,
        required=True,
        nargs=2
    )

    parser.add_argument(
        '--dt',
        help='step size for deep layers',
        type=float,
        default=0.1)
    

    parser.add_argument(
        '--file-dir',
        help='GCS or local paths to tf record data. Must be organized data/gameID/{test,train,val}',
        nargs='+',
        required=True
    )

    parser.add_argument(
        '--stoch-runs',
        help='Number of simulations to run montecarlo for stochastic models',
        type=int,
        default=1
    )


    # Training arguments
    parser.add_argument(
        '--model-dir',
        help='GCS location to write checkpoints and export models',
        required=True,
    )
    # Training parameters
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
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=40
    )
    # Experiment arguments
    parser.add_argument(
        '--train-steps',
        help="""\
        Steps to run the training job for. If --num-epochs is not specified,
        this must be. Otherwise the training job will run indefinitely.\
        """,
        type=int
    )
    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
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
    #print(args)
    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    # Slows things down. Turn off for large models
    tf.logging.set_verbosity('INFO')

    # Get derived model inputs.
    #
    
    # Input function from the resolution
    input_fn = 'screen_shots_'+str(args.resolution[0])
    input_shape =tuple( args.resolution + [3])

    # Find number of classes in TFRecords directory
    classes = len(os.listdir(args.file_dir[0]))


    # Create the class
    # Set model parameters
    model_param = {'block':'f_E',
                'depth':1,
                'dt':0.1,
                'conv_spec':[5,16],
                'learning_rate':0.01,
                'activation':tf.nn.relu}
    test_screen_shot_model = models.DeepModel('van',1, 
                                                        input_shape = (28,28,3),
                                                        conv_spec = [5,16],
                                                        num_classes=5,
                                                        dt=0.01,
                                                        learning_rate=.001,
                                                        activation=tf.nn.relu 
                                                        )
    
    #test_screen_shot_model.train_and_eval(args.file_dir[0],'traintest',
    #                    train_steps=2,
    #                    eval_steps=2)
    model_path = 'dtest/28x28x3_5/van1/traintest/1520456939'
    test_screen_shot_model.predict('/home/jay/Network-Comparisons/otest_images/',model_path=model_path)


