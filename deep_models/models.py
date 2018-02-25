import tensorflow as tf
from deep_models import train_eval_exp
from deep_models import blocks
import numpy as np
import pandas as pd 

class ExpModel:
    def __init__(self,block,depth,input_fn,
                    model_dir='Models',
                    input_shape=(28,28,3),
                    num_classes=10,
                    conv = [5,16],
                    dt=.1,
                    activation=tf.nn.relu):
        """
        Creates a Stochastic, or not, residual network with depth-number of block type layers.
        block: Choice of block to repeat.
                Must be from {'Stf_EM','f_E'}. (str)
        depth: Number of repeats of block. (int)
        input_fn: Must be 'mnist' or key from INPUT_FNS dictionary. (str)
        model_dir: (Optional) Directory to store model outputs. Default 'Models' (str)
        color: (Optional) Whether data has color channel. Default False (bool)
        classes: (Optional) Number of classes the data has. Default 10. (int)
        input_shape: (Optional) Resolution of input image. Default (28,28). (tup)
        filters: (Optional) Number of filters for initial convolution. Default 16 (int)
        activation: (Optional) Activation function of initial and final layer of network. 
                    Default is relu. (function)
        dt: (Optional) Step size for blocks. Default 0.1 (float)
        conv: Size of square kernel to use (int), Number of filters (int). Default [5,16] (list)
        """

        # Save attributes
        self.model_dir = model_dir  

        # input_fn based on data type attribute
        if block[0]=='S':
            self.stoch = True
        else:
            self.stoch = False
        
        if input_fn == 'mnist':
            self.input_fn = input_fn
            self.input_shape = (28,28,1)
        else:
            self.input_fn = train_eval_exp.INPUT_FNS[input_fn]
            self.input_shape = input_shape

        # Deep layer attributes
        self.depth = depth
        self.act = activation   
        self.block_fn = blocks.BLOCKS[block]    # Block type
        self.conv_shape = conv  # Convolution shape
        self.dt = dt    # Step size

        # First and last layer attributes

        self.classes = num_classes  # Number of names to categorize
        

        #----------------------------------------
        #       MODEL FUNCTION
        #----------------------------------------

        def mk_model_fn(features,labels=None,mode=None):
            print('MODE:',mode)
            # Input Layer
            # Reshape X to 4-D tensor: [batch_size, width, height, channels]
            input_layer = tf.reshape(features["x"], [-1]+list(self.input_shape))

            # MNIST is fed labels directly
            # Need to pick out features for the training of other models
            if not self.input_fn == 'mnist':
                labels = features["y"]
                
            
            # Initial convolution layer
            kernel_size, filters = self.conv_shape

            conv0 = tf.layers.conv2d(
                inputs=input_layer,
                filters=filters,
                kernel_size=kernel_size, # Make small to allow for more layers
                padding="same",
                activation=self.act)
            
            
            # Deep portion
            # Step size - set for stability
            h = self.dt

            Deep = tf.contrib.layers.repeat(conv0,
                                                self.depth,
                                                self.block_fn, 
                                                h,
                                                self.conv_shape, scope='')

            # Dense Layer
            # Make sure size matches Deep
            Deep_size = self.input_shape[0]*self.input_shape[1]*self.conv_shape[1]
            Deep_flat = tf.reshape(Deep, [-1, Deep_size])

            dense = tf.layers.dense(inputs=Deep_flat, units=1024, activation=tf.nn.relu)

            # No dropout on stochastic networks since the stochastic block
            # should help regularize already
            if self.stoch:
                last = dense
            else:
                dropout = tf.layers.dropout(
                    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
                last = dropout
            # Logits Layer
            # Units is number of games
            logits = tf.layers.dense(inputs=last, units=self.classes)
        
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

        # Add model function to model
        self.model_fn = mk_model_fn

        #----------------------------------------
        #       TRAINING, EVAL, AND EXPORT
        #----------------------------------------

        def mk_train_and_eval( data_dir, exp_dir,
                        batch_size=100,
                        train_epochs=None,
                        train_steps=None,
                        eval_steps=None,
                        eval_epochs=None):
            """
            Parameters:
                Train and evaluate the model using Estimators. Eval batch size is set to 1.
                data_dir: Where the data is stored. Must be accessible by input_fn. 
                            For mnist set to None. (str)
                exp_dir: Directory to export checkpoints and saved model. (str)
                batch_size: Training batch size. Default 100. (int)
                train_epochs: Number of epochs to train for. If None must set train_steps.
                            Default None. (int)
                train_steps: Max number of training steps. If none will run until out of inputs.
                            Default None (int)
                eval_epochs: Number of evaluation epochs. Default None (int)
            
            """
            # Construct the classifier
            classifier = tf.estimator.Estimator( model_fn=self.model_fn,
                                                model_dir=self.model_dir)
            

            # Set up logging for predictions
            # Log the values in the "Softmax" tensor with label "probabilities"
            tensors_to_log = {"probabilities": "softmax_tensor"}
            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=500)

            # Make training and evaluation input functions
            # MNIST input function downloads the data here
            if self.input_fn == 'mnist':
                # Load training and eval data
                mnist = tf.contrib.learn.datasets.load_dataset("mnist")
                train_data = mnist.train.images  # Returns np.array
                train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
                eval_data = mnist.test.images  # Returns np.array
                eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

                train_input = tf.estimator.inputs.numpy_input_fn(
                    x={"x": train_data},
                    y=train_labels,
                    batch_size=batch_size,
                    num_epochs=train_epochs,
                    shuffle=True)
                
                eval_input = tf.estimator.inputs.numpy_input_fn(
                    x={"x": eval_data},
                    y=eval_labels,
                    num_epochs=eval_epochs,
                    shuffle=False)
                
                # MNIST doesn't have labels as input so its
                # feature spec is different
                feature_spec = {"x":tf.placeholder(dtype=tf.float32,shape = [28,28,1])}

            # Handle non mnist data
            else:
                # Get the train input function
                train_input = lambda: self.input_fn('train',file_dir = [data_dir],
                    num_epochs=train_epochs,
                    batch_size=batch_size)

                # Don't shuffle evaluation data
                eval_input = lambda: self.input_fn('eval', file_dir=[data_dir],
                    batch_size=1,
                    shuffle=False)
                
                # Make the feature spec for exporting
                feature_spec = {"x":tf.placeholder(dtype=tf.float32,shape = self.input_shape),
                                "y":tf.placeholder(dtype = tf.int32,shape = [1])}

            #Train the model
            classifier.train(train_input,
                            steps=train_steps,
                            )

            # Evaluate the model
            eval_name = '/'.join([model_dir,input_fn,'_eval'])
            classifier.evaluate(eval_input,
                                steps = eval_steps,
                                name=eval_name)
            
            # Export the model
            exp_name = '/'.join([model_dir,exp_dir])
            
            # Add the directory to the model
            self.exp_dir = exp_name

            classifier.export_savedmodel(exp_name,
                                tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
                                )

        # add the training and eval as an attribute
        self.train_and_eval = mk_train_and_eval

        #----------------------------------------
        #       PREDICTION
        #----------------------------------------


        def mk_prediction(data_dir, model_dir,
                            labels_file = 'labels_key.csv',
                            col_names = ['NAME','LABEL'],
                            ):
            """
            Predict classes from raw image files. 
            data_dir: Directory containing images to classify. (str)
            model_dir: Directory containing model's .pd file. (str)
            labels_file: Name of the csv file with the labels to category mapping. (str)
            col_names: Name of the columns from the csv labels_file. 'NAME' should be actual\
            class name. 'LABEL' is the integer label used in the model.(list)
            """

            wrk_dir = os.getcwd()
            graph_path = '/'.join([wrk_dir,model_dir])
            print('Graph path: %s'%(graph_path))

            # Get image directory and image path names
            images = os.listdir(data_dir)
            images = ['/'.join([data_dir,img]) for img in images]

            num_imgs = len(images)

            # Make labels key for mnist
            if self.input_fn == 'mnist':
                labels_key = {'NAME': range(10), 'LABEL': range(10)}

            else:
                # Get the labels key
                labels_key = pd.read_csv('labels_key.csv')

                # Translate the columns
                translate = {col_names[0]: 'NAME', col_names[1]: 'LABEL'}
                labels_key = labels_key.rename(columns=translate)
                

            names = list(labels_key['CLASS'])

            
            # Get prediction results
            results =predict_imgs(images,graph_path,res=self.input_shape)

            # Format them to be a DataFrame
            results = list(zip(results['names'],results['confidences'],results['inferences']))

            for rix in range(len(results)):
                name, conf, infer = results[rix]
                # Switch inference from label to gameID
                infer = [names[infer[0]]]
                results[rix] = name+conf+infer

            columns = ['image']
            columns+= names
            columns+= ['inference']

            results_df = pd.DataFrame(results, columns=columns)
            # Round the probabilities 
            results_df[names] = results_df[names].apply(lambda x: pd.Series.round(x,4))

            # Save the results
            results_df.to_csv(args.save+'.csv')

        self.predict = mk_prediction





    def describe(self):
            return (self.block,self.depth,self.input_shape)
    

    