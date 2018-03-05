

def model_fn(features,labels=None,labels=None,mode=None,model_spec):
    """block, depth = 1,
                             input_shape = (28,28,3),
                             num_classes = 10,
                             conv_spec = [5,16],"""
    
    keys = ['block','depth','input_shape','num_classes','conv_spec','dt','activation',]
    block, depth, input_shape, num_classes, conv_spec,dt,activation = [model_spec.get(key) for key in keys]
    
    print('MODE:',mode)
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1]+list(input_shape))

    # MNIST is fed labels directly
    # Need to pick out features for the training of other models
    if not input_fn == 'mnist':
        labels = features["y"]
        
    
    # Initial convolution layer
    kernel_size, filters = conv_spec

    conv0 = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size, # Make small to allow for more layers
        padding="same",
        activation=activation)
    
    
    # Deep portion
    # Step size - set for stability
    h = dt

    Deep = tf.contrib.layers.repeat(conv0,
                                        depth,
                                        block_fn, 
                                        h,
                                        conv_spec, scope='')

    # Dense Layer
    # Make sure size matches Deep
    Deep_size = input_shape[0]*input_shape[1]*conv_spec[1]
    Deep_flat = tf.reshape(Deep, [-1, Deep_size])

    dense = tf.layers.dense(inputs=Deep_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    last = dropout
    # Logits Layer
    # Units is number of games
    logits = tf.layers.dense(inputs=last, units=num_classes)

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

