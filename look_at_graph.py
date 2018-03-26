import tensorflow as tf 
import numpy as np 

model_dir = '/home/jay/Network-Comparisons/d_models/28x28x1_10_2/Wf_EM1/model.ckpt-5.meta'
model_stuff = '/home/jay/Network-Comparisons/d_models/28x28x1_10_2/Wf_EM1/model.ckpt-5'
# Placeholder for feeding the hungry hungry model
t_input = tf.placeholder(np.float32, 
                    shape=[None, 28, 28, 1], 
                    name='new_input') # define the input tensor

# Need to feed the input into the 'input' node of the graph. 
# This is done by specifying the input map.

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(model_dir, input_map={'Reshape':  t_input})

    new_saver.restore(sess, model_stuff)

    vars = tf.get_collection('variables')
    stuff = [n.name for n in tf.get_default_graph().as_graph_def().node]

    with open('vars.txt','w') as fp:
        for x in vars:
            fp.write(x+'\n')
    
    PRINTIT = False
    if PRINTIT:
        for x in vars: 
            print(x)
        for x in stuff:
            print(x)