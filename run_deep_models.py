# Run deep_models experiments

from deep_models import models
import numpy as np
import tensorflow as tf 

# Slows things down. Turn off for large models
tf.logging.set_verbosity('INFO')

#test = 'mnist'
test = 'scrn'
if test == 'mnist':
    # Test ExpModel on MNIST
    test_mnist_model = models.ExpModel('f_E',2,'mnist',model_dir='test_mnist',dt=.1)
    test_mnist_model.train_and_eval('mnist',train_steps=200,eval_steps=10)

else:
    # Test ExpModel on screen shot data
    test_screen_shot_model = models.ExpModel('f_E',3,'screen_shots_28',
                                            model_dir='test_scrn',
                                            input_shape = (28,28,3)
                                            dt=.1,num_classes=3)
    # Train Eval and Save
    test_screen_shot_model.train_and_eval('TFRecords_28x28',train_steps=2,eval_steps=10)
    # Predict
    test_screen_shot_model.predict(col_names=['GAMEID','LABEL'])
