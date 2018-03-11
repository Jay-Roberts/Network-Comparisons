# Network-Comparisons

A small wrapper package to easily test different deep architecture block types. A basic run script is in *test_run.sh*. 

## Set Up

Clone the repository to get the *deep_models* package. Currently it must be in the local directory you wish to import it in. 

```git
    git clone https://github.com/Jay-Roberts/Network-Comparisons.git
```
Testing on the mnist model uses tensorflows built in mnist dataset. First create the class.
    
```python

    from deep_models import models

    test_screen_shot_model = models.DeepModel('van',2, 
                                input_shape = (28,28,1),
                                conv_spec = [5,16],
                                num_classes=10,
                                mnist=True)
 ```
This will create a deep network with an initial convolution layer of 16 filters with size 5x5 and then repeat through the block type 'van' (vanilla cnn block). The input shape and number of classes it expects is consistent with MNIST data and we told it to load the MNIST dataset.

To train the model.

```python
    test_screen_shot_model.train_and_eval('datadir','traintest',
                                        train_steps=2000,
                                        eval_steps=2000)
```

'datadir' is where the data is stored (unnecessary for mnist) and 'traintest' is a subdirectory to hold models.

To predict from the model.
```python
    model_path = test_screen_shot_model.exp_dir
    test_screen_shot_model.predict('testimages/')
```
If the _train _ and _ eval_ method has been run the model path can be found in self.exp_dir, if not then you must specify the path the the directory containing the .pb save file. 

## API

**Class: DeepModel** :defines the DeepModels class with a defined block type, depth, activation function. The first and last layer are fixed as convolution and logits with softmax respectively but we plan to allow these to be customized in the future.


```python
    deep_models.DeepModel(block,depth,
                    model_dir='d_models',
                    input_shape=(28,28,3),
                    num_classes=10,
                    conv_spec = [5,16],
                    dt=.1,
                    learning_rate=.001,
                    activation=tf.nn.relu,
                    stoch_passes=None,
                    mnist=False):
```

* Inputs:
    * block:  Choice of block to repeat. Must be a key from *blocks*.BLOCKS. (str)
    * depth: Number of repeats of block. (int)
    * model_dir: (Optional) Directory to store model outputs. 
    Default 'd_models' (str)
    * input_shape: (Optional) Resolution of input image. 
    Default (28,28). (tup)
    * num_classes: (Optional) Number of classes the data has. 
    Default 10. (int)
    * conv_spec: Size of square kernel to use (int), Number of filters (int). 
    Default [5,16] (list)
    * dt: (Optional) Step size for blocks. 
    Default 0.1 (float)
    * learning_rate: (Optional) The learning rate for gradient decent during training.
    Deault .001 (float)
    * activation: (Optional) Activation function of initial and final layer of network. 
    Default is relu. (function)
    * stoch_passes: Experimental leave as None.
    * mnist: (Optional) Whether to load data from mnist or not. 
    Default is False. (bool)        

Returns: DeepModel class with following 

**Properties:**

* exp_dir: Base directory for exporting .pb files (str)
* model_dir: Base directory to save model files. (str)
* model_spec: Wraps up inputs of __init__ into dictionary. (dict)

* model_fn: the model function that defines the computational graph for tf.Estimator.
    * Consists of an initial convolution layer with _conv _ spec_  number of filters and kernel size. Then a _depth_ deep layer made of repeated _block_s . Then a final hidden dense layer with 1024 units, this will be modifiable in the future. Output is a logits layer with _num _ classes_ units.

**Methods:**

**Train, Evaluate, Export:**
```python
    DeepModel.train_and_eval(self,data_dir,exp_dir,
                        train_steps=None,
                        train_epochs=None,
                        train_batch=100,
                        eval_steps=None,
                        eval_epochs=None,
                        eval_batch=100):
```
Train, evaluate, and export a saved model. 
* Inputs:
    * data_dir: Where the data is stored. Must be accessible by input_fn. 
                For mnist set to None. (str)
    * model_fn: The model function to use to construct estimator.
    * input_shape: Image resolution to use in model. (tuple)
    * exp_dir: Directory to export checkpoints and saved model. (str)
    * train_steps: Number of steps to train for. If None train_epochs must be. (int)
    * train_epochs: Number of epochs to run through training data. If None train_steps must be. (int)
    * train_batch: Batch size to use for training. Default 100. (int)
    * eval_steps: Number of steps to eval for. If None eval_epochs must be. (int)
    * eval_epochs: Number of epochs to run through evaling data. If None eval_steps must be. (int)
    * eval_batch: Batch size to use for evaling. Default 100. (int)

* Returns:
    None

**Prediction:**
    
```python
    DeepModel.predict(self,images_dir,
            model_path=None,
            labels_key='labels_key.csv',
            out_name='predictions'):
```
Predict classes from raw image files. 

* Inputs:
    * save_dir: Relative directory containing model's .pd file. (str)
    * labels_dict: Dictionary of model labels to actual classes. (dict)
    * res: Resolution of input to model. (tuple)
    * data_dir: Relative directory containing images to classify. 
        Default is 'test_images'. (str)
    * out_name: Name of csv containing predictions. 
        Default is 'predictions'. (str)
* Returns:
    * Creates out_name.csv with predictions from the model of testimages.
