from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import support_fns as sfn

tf.logging.set_verbosity(tf.logging.INFO)

#-----------------------------------------------
#           Define model fns here
#-----------------------------------------------



# A test function modified from the tf tutorial
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

def SRNN_2_model_fn(features, labels, mode):
    """
    A 20 layer Stochastic Residual network. Interpreted as propogating up to time T = 1.
    name: SRNN_2 \
    Input size: 28x28x1 \
    dt = 0.5
    
    """
    print('MODE:',mode)

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
    dt = 0.5

    Residual = tf.contrib.layers.repeat(conv0, 2, sfn.sNN_layer, dt, scope='')

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




def SRNN_10_model_fn(features, labels, mode):
    """
    A 10 layer Stochastic Residual network. Interpreted as propogating up to time T = 1.
    name: SRNN_10 \
    Input size: 28x28x1 \
    dt = 0.1  
    """
    print('MODE:',mode)

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

    Residual = tf.contrib.layers.repeat(conv0, 10, sfn.sNN_layer, dt, scope='')
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


def SRNN_20_model_fn(features, labels, mode):
    """
    A 20 layer Stochastic Residual network. Interpreted as propogating up to time T = 1.
    Name: SRNN_20 \
    Input size: 28x28x1 \
    dt = 0.05
    """
    print('MODE:',mode)

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
    dt = 0.05

    Residual = tf.contrib.layers.repeat(conv0, 20, sfn.sNN_layer, dt, scope='')

    # Dense Layer
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
    'SRNN_2': SRNN_2_model_fn,
    'SRNN_10': SRNN_10_model_fn,
    'SRNN_20': SRNN_20_model_fn
}