from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

#-----------------------------------------------
#           Define blocks here
#-----------------------------------------------


# Residual Stochastic Convolutional Layer
# This corresponds to Explicit Euler-Maruyama
def sNN_layer(input_layer, dt, scope):

    # Initialize Diagonal noise
    dz = tf.random_normal(tf.shape(input_layer))
    root_dt = tf.sqrt(dt)

    # Compute Stochastic function at previous step
    fs = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[5, 5], # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)

    # Compute Deterministic function
    fd = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[5, 5], # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)

    return tf.add( input_layer, tf.add( tf.scalar_mul(dt,fd) , tf.scalar_mul( root_dt, tf.multiply(fs,dz) ) ) )



#-----------------------------------------------
#           Define model fns here
#-----------------------------------------------



# A test function modified from the tf tutorial:
# https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/layers/cnn_mnist.py

def test_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)
    

    #---------------------ALL MODELS NEED THIS PORTION BELOW-----------------------------#
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    
    # Here we modify the prediction routine to save exported Estimators after evaluation
    if mode == tf.estimator.ModeKeys.PREDICT:
        print('Infer')
        pred_out = tf.estimator.export.PredictOutput(predictions)
        exp_outs = {'predict_outputs':pred_out}

        return tf.estimator.EstimatorSpec(mode=mode,
                                         predictions=predictions,
                                         export_outputs=exp_outs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        print('Training')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    print('Evaluating')
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



#---------------------------------------------------------------------------------------------------
#   A Stochastic Residual Neural Network 
#   with diagonal noise (SRN)
#   2 snn blocks
#   
#---------------------------------------------------------------------------------------------------

def SRNN_model_fn(features, labels, mode):
    """
    name: SRNN_2
    """
    print('MODE:',mode)
    #print('Feature type: ', features.shape)
    # Input layer
    #input_layer = tf.reshape(features['image'], [-1,28,28,3],name='input_layer')
    #labels = features['label']

    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv0 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[5, 5], # Make small to allow for more layers
        padding="same",
        activation=tf.nn.relu)

    # Step size - set for stability
    dt = 0.1

    Residual = tf.contrib.layers.repeat(conv0, 2, sNN_layer, dt, scope='')
  
    
    # Remove pool to preserve size
    #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    # Make sure size matches residual
    Residual_flat = tf.reshape(Residual, [-1, 28 * 28 * 16])
    dense = tf.layers.dense(inputs=Residual_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    # Units is number of games
    logits = tf.layers.dense(inputs=dropout, units=10)

 #---------------------ALL MODELS NEED THIS PORTION BELOW-----------------------------#
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    
    # Here we modify the prediction routine to save exported Estimators after evaluation
    if mode == tf.estimator.ModeKeys.PREDICT:
        print('Infer')
        pred_out = tf.estimator.export.PredictOutput(predictions)
        exp_outs = {'predict_outputs':pred_out}

        return tf.estimator.EstimatorSpec(mode=mode,
                                         predictions=predictions,
                                         export_outputs=exp_outs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        print('Training')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    print('Evaluating')
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




# Add any new model functions here with a unique identifier
Model_Functions = {
    'Test': test_fn,
    'SRNN_2': SRNN_model_fn
}