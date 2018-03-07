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
                    dt=.1,
                    learning_rate=.001,
                    activation=tf.nn.relu,
                    mnist=False):
        """
        Class to create, train, evaluate, save, and make inferences from deep network models.
        Inputs:
            block: Choice of block to repeat.
                    Must be from {'Stf_EM','f_E'}. (str)
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


