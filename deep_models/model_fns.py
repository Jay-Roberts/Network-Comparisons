import tensorflow as tf 

# Deterministic CNN 
def model_fn(exp_spec,features=None,labels=None,mode=None):
    """
    Creates a model function according to experiment specifications exp_spec.
    Inputs:
        exp_spec: Model specifications as dictionary.
        {   'depth': depth of model (int),
            'activation': activation function (function),
            'block': block function (function),
            'dt': step size for deep layer (float),
            'input_shape': input_shape (tuple),
            'conv_spec': [filter size, number of filters] (list),
            'classes': number of classes (int),
            'learning_rate': learning rate for training (float)
            'mnist': whether data is mnist. (bool),
            'cifar': whether data is cifar. (bool)
                        }
    """
    
    keys = [
        'block',
        'depth',
        'input_shape',
        'num_classes',
        'conv_spec',
        'final_units',
        'dt',
        'activation',
        'learning_rate'
        ]
    inputs = [exp_spec.get(key) for key in keys]

    if 'mnist' in exp_spec.keys():
        input_shape = (28,28,1)

    if 'cifar' in exp_spec.keys():
        input_shape = (32,32,3)
        #input_shape = (16,16,3)
    
    
    block, depth, input_shape = exp_spec['block'], exp_spec['depth'], exp_spec['input_shape']
    num_classes, conv_spec,dt = exp_spec['classes'], exp_spec['conv_spec'], exp_spec['dt']
    activation,learning_rate, final_units = exp_spec['activation'], exp_spec['learning_rate'], exp_spec['final_units']
    
    print('=======================CLASSES===========================',num_classes)
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1]+list(input_shape))
    labels = features["y"]

    # Initial convolution layer
    kernel_size, filters = conv_spec

    conv0 = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size, # Make small to allow for more layers
        #kernel_regularizer=tf.nn.l2_loss,
        padding="same",
        activation=activation)
    
    # Deep portion
    # Step size - set for stability
    h = dt
    block_fn = block
    Deep = tf.contrib.layers.repeat(conv0,
                                        depth,
                                        block_fn,
                                        dt,
                                        conv_spec)

    # Dense Layer
    # Make sure size matches Deep
    #print('The resolution is ', input_shape[0], ' ', input_shape[1], ' ', conv_spec[1])
    Deep_size = input_shape[0]*input_shape[1]*conv_spec[1]
    Deep_flat = tf.reshape(Deep, [-1, Deep_size])

    # Ideally units is 1024, but using small value to allow for GPU Training on my GTX 970. Experiment here for the P100's
    #dense = tf.layers.dense(inputs=Deep_flat, units=final_units, activation=tf.nn.relu, kernel_regularizer=tf.nn.l2_loss)
    dropout = tf.layers.dropout( inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    last = dropout
    # Logits Layer
    # Units is number of games
    #logits = tf.layers.dense(inputs=last, units=num_classes, kernel_regularizer=tf.nn.l2_loss,)

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
    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.losses.add_loss(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))
    regularization_coefficient = h
    loss = tf.add(tf.losses.get_total_loss(add_regularization_losses=False),
                    regularization_coefficient*tf.losses.get_regularization_loss() )
    
    # Keep track of loss
    tf.summary.scalar("loss",loss)
    summary_hook = tf.train.SummarySaverHook(
            save_steps = 100,
            output_dir='./TBTraining',
            summary_op=tf.summary.merge_all())

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        print('Training')
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode=mode,
                                         loss=loss, 
                                         train_op=train_op, 
                                         training_hooks=[summary_hook])

    if mode == tf.estimator.ModeKeys.EVAL:
        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        print('Evaluating')
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Strong Stochastic CNN

def weak_stoch_model_fn(exp_spec,features=None,labels=None,mode=None):
    """
    Creates a stochastic model function according to experiment specifications exp_spec. Approximates
    various moments for inference. 
    Inputs:
        exp_spec: Model specifications as dictionary.
        {   'depth': depth of model (int),
            'activation': activation function (function),
            'block': block function (function),
            'dt': step size for deep layer (float),
            'input_shape': input_shape (tuple),
            'conv_spec': [filter size, number of filters] (list),
            'classes': number of classes (int),
            'learning_rate': learning rate for training (float)
            'stoch_passes': number of passes for stochastic block
                        }
    """
    
    keys = [
        'block',
        'depth',
        'input_shape',
        'num_classes',
        'conv_spec',
        'final_units',
        'dt',
        'activation',
        'learning_rate'
        ]
    inputs = [exp_spec.get(key) for key in keys]

    if 'mnist' in exp_spec.keys():
        input_shape = (28,28,1)

    if 'cifar' in exp_spec.keys():
        input_shape = (32,32,3)
        #input_shape = (16,16,3)

    block, depth, input_shape = exp_spec['block'], exp_spec['depth'], exp_spec['input_shape']
    num_classes, conv_spec,dt = exp_spec['classes'], exp_spec['conv_spec'], exp_spec['dt']
    activation,learning_rate, passes = exp_spec['activation'], exp_spec['learning_rate'], exp_spec['stoch_passes']
    final_units = exp_spec['final_units']
    
    
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1]+list(input_shape))

    labels = features["y"]   
    #batch_size = features['batch'].get_shape()[1]
    #batch_size = features['batch']
    #print("===========Batch size", batch_size)
    
    # Initial convolution layer
    kernel_size, filters = conv_spec

    conv0 = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size, # Make small to allow for more layers
        #kernel_regularizer=tf.nn.l2_loss,
        padding="same",
        activation=activation,
        name='InitConv')
    
    #
    # Make deep repeater from scratch to control scope
    #
    def deep_ones(x,d,blk,hh,conv,scope='',name='deep_twos'):
        deep = x
        for i in range(d):
            # Blocks have reuse = tf.AUTO_REUSE this allows for looping
            # with same kernel.
            iname = 'x'.join([name,str(i)])

            deep = blk(deep, hh,conv,scope=scope,name=iname)
        return deep

    #
    # Stochastic simulation phase
    #    

    # Run the process once to initialize kernels and get guess
    h = dt
    Deep = deep_ones(conv0,depth,block,h,conv_spec,name='StochasticPasses')
    
    # Dense Layer
    # Make sure size matches Deep
    Deep_size = input_shape[0]*input_shape[1]*conv_spec[1]
    Deep_flat = tf.reshape(Deep, [-1, Deep_size])

    # Combine the convolution features
    dense = tf.layers.dense(inputs=Deep_flat,
                            units=final_units,
                            #kernel_regularizer=tf.nn.l2_loss,
                            activation=tf.nn.relu,
                            name='inner_dense'
                            )
    
    # Try a basic nonsense output
    # Compress into one output
    #logits_guess = tf.zeros(shape=[batch_size,num_classes])
    # Initialize one layer of network

    
    logits_guess = tf.layers.dense(inputs = dense, 
                                units = num_classes, 
                                #kernel_regularizer=tf.nn.l2_loss,
                                #scope='dense_two',
                                name='inner_guess',
                                reuse=tf.AUTO_REUSE)
    
    

    for run in range(passes):
        
        Deep = tf.stop_gradient(Deep,name='StopIT_Deep')
        Deep_flat = tf.stop_gradient(Deep_flat, name='StopID_Flat')
        dense = tf.stop_gradient(dense,name='StopIT_dense')
        


        Deep = deep_ones(conv0,depth,block,h,conv_spec,name='StochasticPasses')

        # Dense Layer
        # Make sure size matches Deep
        Deep_size = input_shape[0]*input_shape[1]*conv_spec[1]
        Deep_flat = tf.reshape(Deep, [-1, Deep_size])

        # Combine the convolution features
        dense = tf.layers.dense(inputs=Deep_flat,
                                units=final_units,
                                activation=tf.nn.relu,
                                #kernel_regularizer=tf.nn.l2_loss,
                                #scope='dense_two',
                                name='inner_dense',
                                reuse=tf.AUTO_REUSE)
        

        logits_guess1 = tf.layers.dense(inputs = dense, 
                                units = num_classes, 
                                #kernel_regularizer=tf.nn.l2_loss,
                                #scope='dense_two',
                                name='inner_guess',
                                reuse=tf.AUTO_REUSE)
        
        # Try an overly simple output
        logits_guess1 = tf.stop_gradient(logits_guess1,name='StopIT_logits')

        # Compress into one output
        logits_guess = tf.add(logits_guess,logits_guess1)

        #if run>0:
        #    Deep = tf.stop_gradient(Deep,name='StopIT_Deep')
        #    Deep_flat = tf.stop_gradient(Deep_flat, name='StopID_Flat')
        #    dense = tf.stop_gradient(dense,name='StopIT_dense')
        #    logits_guess1 = tf.stop_gradient(logits_guess1,name='StopIT_logits')

        

    # Make it an average
    normalizer = 1/float(passes-1)
    logits = tf.scalar_mul(normalizer,logits_guess)





    #loop_out = tf.reshape(guess,[-1],name="loop_out")
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, output_type = tf.int32),
        "probabilities": tf.nn.softmax(logits, name="softmax")
    }
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    #loss = tf.losses.mean_squared_error(labels,loop_out)
    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.losses.add_loss(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))
    regularization_coefficient = h
    loss = tf.add(tf.losses.get_total_loss(add_regularization_losses=False),
                    regularization_coefficient*tf.losses.get_regularization_loss() )
    #acc = tf.metrics.accuracy(
    #            labels=labels, predictions=predictions["classes"], name="acc_summary")
    correct_prediction = tf.equal(labels, predictions["classes"])
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("loss",loss)
    tf.summary.scalar("accuracy",acc)
    

    summary_hook = tf.train.SummarySaverHook(
            save_steps = 500,
            output_dir='./TBTraining',
            summary_op=tf.summary.merge_all())


    # Here we modify the prediction routine to save exported Estimators after evaluation
    if mode == tf.estimator.ModeKeys.PREDICT:
        print('Infer')
        pred_out = tf.estimator.export.PredictOutput(predictions)
        exp_outs = {'predict_outputs':pred_out}

        return tf.estimator.EstimatorSpec(mode=mode,
                                        predictions=predictions,
                                        export_outputs=exp_outs)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        print('Training')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        
        
        return tf.estimator.EstimatorSpec(mode=mode,
                                         loss=loss, 
                                         train_op=train_op, 
                                         training_hooks=[summary_hook])

    if mode == tf.estimator.ModeKeys.EVAL:
        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        print('Evaluating')
        return tf.estimator.EstimatorSpec(mode=mode, 
                                        loss=loss, 
                                        eval_metric_ops=eval_metric_ops, 
                                        evaluation_hooks=[summary_hook])

