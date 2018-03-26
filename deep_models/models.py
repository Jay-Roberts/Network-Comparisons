import tensorflow as tf
import csv
import pickle
import os
from deep_models import blocks

class DeepModel:
    def __init__(self,block,depth,
                    model_dir='d_models',
                    input_shape=(28,28,3),
                    num_classes=10,
                    conv_spec = [5,16],
                    final_units = 1024,
                    dt=.1,
                    learning_rate=.001,
                    activation=tf.nn.relu,
                    stoch_passes=None,
                    mnist=False):
        """
        Class to create, train, evaluate, save, and make inferences from deep network models.
        Inputs:
            block: Choice of block to repeat.
                    Must be from {'Stf_EM','f_E','van','Wf_EM'}. (str)
            depth: Number of repeats of block. (int)
            model_dir: (Optional) Directory to store model outputs. 
                Default 'd_models' (str)
            input_shape: (Optional) Resolution of input image. 
                Default (28,28). (tup)
            num_classes: (Optional) Number of classes the data has. 
                Default 10. (int)
            conv_spec: Size of square kernel to use (int), Number of filters (int). 
                Default [5,16] (list)
            dt: (Optional) Step size for blocks. 
                Default 0.1 (float)
            learning_rate: (Optional) The learning rate for gradient decent during training.
                Deault .001 (float)
            activation: (Optional) Activation function of initial and final layer of network. 
                Default is relu. (function)
            mnist: (Optional) Whether to load data from mnist or not. 
                Default is False. (bool)        
        Returns: DeepModel class with the following attributes
            
            exp_dir: Base directory for exporting .pb files (str)
            mnist: Whether to load data from mnist or not. (bool)
            model_dir: Base directory to save model files. (str)
            model_spec: Wraps up inputs of __init__ into dictionary. (dict)

            model_fn: the model function that defines the computational graph for tf.Estimator.
                   
        """

        self.mnsit = mnist
        
        # Saved in model_dir/resolution_classes/block depth
        input_shape_path = 'x'.join([str(n) for n in input_shape])

        data_path = '_'.join([input_shape_path,str(num_classes)])
        save_path = '/'.join([model_dir,data_path, block+str(depth)])

        # Attribute to model for later
        self.model_dir = save_path
        self.exp_dir = save_path

        # Save path for attributes dicitionary
        att_path = '/'.join([save_path,'ATTRIBUTES.P'])
        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            old_atr = None

        elif os.path.isfile(att_path):
            # The attributes are pickled in a dictionary

            with open(att_path,'rb') as attr_file:
                old_atr = pickle.load(attr_file)
        else:
            old_atr = None
        
        
        

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
                        'learning_rate': learning_rate,
                        'final_units': final_units
                        }
        # model_fn based on data type
        if stoch_passes != 0:
            # If stochastic add passes to attributes
            ATTRIBUTES['stoch_passes'] = stoch_passes
        
        if mnist:
            ATTRIBUTES['mnist'] = True
        
        self.model_specs = ATTRIBUTES
        
        # Check for old model compatability
        if old_atr:
            assert ATTRIBUTES == old_atr, "Existing model parameters do not match current model.\
                                        Remove existing model or rename new model_dir."
        else:
            with open(att_path,'wb') as attr_file:
                pickle.dump(ATTRIBUTES, attr_file)
        
        # Import appropriate model function
        if stoch_passes:
            # Pick out weak model
            if block[0] == 'W':
                from .model_fns import weak_stoch_model_fn as model_fn
            else:
                print('Strong models are not currently supported. Using weak model instead')
                from .model_fns import weak_stoch_model_fn as model_fn
        else:
            from .model_fns import model_fn as model_fn
        
        # Make model function
        self.model_fn = lambda features, labels, mode: model_fn(self.model_specs,
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
        
        
        in_shp  = self.model_specs['input_shape']
        export_dir = '/'.join([self.model_dir,exp_dir])
        
        # Update export directory for training
        self.exp_dir = export_dir

        from .train_eval_exp import train_and_eval
        train_and_eval(data_dir, self.model_fn,self.model_dir,in_shp,export_dir,
                        train_steps=train_steps,
                        train_batch=train_batch,
                        train_epochs=train_epochs,
                        eval_steps=eval_steps,
                        eval_batch=eval_batch,
                        eval_epochs=eval_epochs)
    
    def predict(self,images_dir,model_path=None,
            labels_key='labels_key.csv',
            out_name='predictions'):
        """
        Predict classes from raw image files. 
        Inputs:
            save_dir: Relative directory containing model's .pd file. (str)
            labels_dict: Dictionary of model labels to actual classes. (dict)
            res: Resolution of input to model. (tuple)
            data_dir: Relative directory containing images to classify. 
                Default is 'test_images'. (str)
            out_name: Name of csv containing predictions. 
                Default is 'predictions'. (str)
        Returns:
            Creates out_name.csv with predictions from the model of testimages.
        """

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

        from .predict import mk_prediction
        mk_prediction(save_dir,labels_dict, res,data_dir=images_dir,out_name =out_name)


