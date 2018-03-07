import tensorflow as tf
import csv
import pickle
import os
import blocks

class ExpModel:
    def __init__(self,block,depth,input_fn,
                    model_dir='Models',
                    input_shape=(28,28,3),
                    num_classes=10,
                    conv = [5,16],
                    dt=0.1,
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


        # The model directory is:
        # block/depth/model_dir
        save_path = '/'.join(['models',block,str(depth)+'_layer',model_dir])
        att_path = '/'.join([save_path,'ATTRIBUTES'])
        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            old_atr = None
        else:
            print('Previous model found')
                        # The attributes are pickeled in a dictionary
            with open(att_path,'rb') as attr_file:
                old_atr = pickle.load(attr_file)
            
        # input_fn based on data type
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
        self.model_dir = save_path          

        
        # Make a dictionary to hold attribute info
        # save this to prevent overwritting in same model_dir
        ATTRIBUTES = {'input_fn': self.input_fn,
                        'depth': self.depth,
                        'activation': self.act,
                        'block': self.block_fn,
                        'dt': self.dt,
                        'input_shpae': self.input_shape,
                        'conv_shape': self.conv_shape,
                        'classes': self.classes}
        
        if old_atr:
            assert ATTRIBUTES == old_atr, "Existing model parameters do not match current model.\
                                        Remove existing model or rename new model_dir."
        else:
            with open(att_path,'wb') as attr_file:
                pickle.dump(ATTRIBUTES, attr_file)

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
        
        # Add the directory to the model
        self.exp_dir = '/'.join([self.model_dir,'saved_models'])

        #----------------------------------------
        #       TRAINING, EVAL, AND EXPORT
        #----------------------------------------

        def mk_train_and_eval( data_dir,
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
            eval_name = 'eval'
            classifier.evaluate(eval_input,
                                steps = eval_steps)
            
            # Export the model
            classifier.export_savedmodel(self.exp_dir,
                                tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
                                )

        # add the training and eval as an attribute
        self.train_and_eval = mk_train_and_eval

        #----------------------------------------
        #       PREDICTION
        #----------------------------------------


        def mk_prediction(save_dir,
                            data_dir='test_images',
                            labels_file = 'labels_key.csv',
                            col_names = ['NAME','LABEL'],
                            out_name = 'predicitons'
                            ):
            """
            Predict classes from raw image files. 
            save_dir: Directory containing model's .pd file. (str)
            data_dir: Directory containing images to classify. (str)
            labels_file: Name of the csv file with the labels to category mapping. (str)
            col_names: Name of the columns from the csv labels_file. 'NAME' should be actual\
            class name. 'LABEL' is the integer label used in the model.(list)
            """

            wrk_dir = os.getcwd()
            graph_path = '/'.join([wrk_dir,self.exp_dir,save_dir])
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
                labels_key = pd.read_csv(labels_file)

                # Translate the columns
                translate = {col_names[0]: 'NAME', col_names[1]: 'LABEL'}
                labels_key = labels_key.rename(columns=translate)
                

            names = list(labels_key['NAME'])

            
            # Get prediction results
            results =predict.predict_imgs(images,graph_path,res=self.input_shape)

            # Format them to be a DataFrame
            results = list(zip(results['names'],results['confidences'],results['inferences']))
    
            for rix in range(len(results)):
                name, conf, infer = results[rix]
                # Switch inference from label
                infer = [names[infer[0]]]
                results[rix] = name+conf+infer

            columns = ['image']
            columns+= names
            columns+= ['inference']

            results_df = pd.DataFrame(results, columns=columns)
            # Round the probabilities 
            results_df[names] = results_df[names].apply(lambda x: pd.Series.round(x,4))

            # Save the results
            results_df.to_csv(out_name+'.csv')
            print(results_df.head())

        self.predict = mk_prediction    


class DeepModel:
    def __init__(self,block,depth,
                    model_dir='dtest',
                    input_shape=(28,28,3),
                    num_classes=10,
                    conv_spec = [5,16],
                    dt=.1,
                    learning_rate=.001,
                    activation=tf.nn.relu,
                    mnist=False):
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

        self.mnsit = mnist
        # The model directory is:
        # block/depth/model_dir
        input_shape_path = 'x'.join([str(n) for n in input_shape])
        data_path = '_'.join([input_shape_path,str(num_classes)])
        save_path = '/'.join([model_dir,data_path, block+str(depth)])
        self.model_dir = save_path
        self.exp_dir = save_path

        att_path = '/'.join([save_path,'ATTRIBUTES.P'])
        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            old_atr = None
        elif os.path.isfile(att_path):
            print('Previous model found')
                        # The attributes are pickeled in a dictionary
            with open(att_path,'rb') as attr_file:
                old_atr = pickle.load(attr_file)
        else:
            old_atr = None
        
        # input_fn based on data type
        if block[0]=='S':
            self.stoch = True
        else:
            self.stoch = False
        

        # Make a dictionary to hold attribute info
        # save this to prevent overwritting in same model_dir
        print(block)
        block_fn = blocks.BLOCKS[block]
        ATTRIBUTES = {  'depth': depth,
                        'activation': activation,
                        'block': block_fn,
                        'dt': dt,
                        'input_shape': input_shape,
                        'conv_spec': conv_spec,
                        'classes': num_classes,
                        'learning_rate': learning_rate
                        }
        
        self.model_specs = ATTRIBUTES
        
        if old_atr:
            assert ATTRIBUTES == old_atr, "Existing model parameters do not match current model.\
                                        Remove existing model or rename new model_dir."
        else:
            with open(att_path,'wb') as attr_file:
                pickle.dump(ATTRIBUTES, attr_file)
        
        # Make the model function
        import model_fns
        self.model_fn = lambda features, labels, mode: model_fns.model_fn(self.model_specs,
                                                                        features=features,
                                                                        labels=labels,
                                                                        mode=mode)
    
    # Define the training and evaluation routine
    def train_and_eval(self,data_dir,exp_dir,
                        train_steps=None,
                        train_epochs=None,
                        train_batch=100,
                        eval_steps=None,
                        eval_epochs=None,
                        eval_batch=100):
        
        import train_eval_exp
        
        in_shp  = self.model_specs['input_shape']
        export_dir = '/'.join([self.model_dir,exp_dir])
        
        # Update export directory for training
        self.exp_dir = export_dir

        train_eval_exp.train_and_eval(data_dir, self.model_fn,self.model_dir,in_shp,export_dir,
                        train_steps=train_steps,
                        train_batch=train_batch,
                        train_epochs=train_epochs,
                        eval_steps=eval_steps,
                        eval_batch=eval_batch,
                        eval_epochs=eval_epochs)
    
    def predict(self,images_dir,model_path=None,
            labels_key='labels_key.csv',
            out_name='predictions'):

        # Get pb dir
        if model_path:
            save_dir = model_path
            print('SAVE DIR: ', save_dir)
        else:
            # Find most recent
            old_models = os.listdir(self.exp_dir)
            old_models = [self.exp_dir + '/'+x for x in old_models]

            mod_times = [os.path.getmtime(x) for x in old_models]
            most_recent_ix = mod_times.index( max(mod_times))

            save_dir = old_models[most_recent_ix]

        # Make labels key for mnist
        if self.mnsit:
            labels_dict = {'NAME': range(10), 'LABEL': range(10)}
        
        else:
            with open(labels_key,mode='rb') as csvfile:
                reader = csv.reader(csvfile)
                
                # Skip past header
                reader.next()
                labels_dict = {'NAME':[],'LABEL':[]}

                for row in reader:
                    name, label = row
                    labels_dict['NAME'].append(name)
                    labels_dict['LABEL'].append(label)
        
        res = self.model_specs['input_shape']

        from predict import mk_prediction
        mk_prediction(save_dir,labels_dict, res,data_dir=images_dir,out_name =out_name)


