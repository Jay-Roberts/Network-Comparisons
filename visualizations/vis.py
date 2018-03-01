import tf_cnnvis
import numpy as np 
import tensorflow as tf 

input_img = np.zeros([28,1,3])
input_img = np.expand_dims(input_img, axis = 0)

model_dir = '/home/jay/Network-Comparisons/models/3_28x28x3/test_scrn/model.ckpt-2.meta'
model_stuff = '/home/jay/Network-Comparisons/models/3_28x28x3/test_scrn/model.ckpt-2'

X = tf.placeholder(dtype=np.float32,name = 'x')
Y = tf.placeholder(dtype = tf.int32, name = 'label')

with tf.Session() as sess:
    hope = tf.train.import_meta_graph(model_dir)
    hope.restore(sess,model_stuff)
    tf_cnnvis.activation_visualization(None,value_feed_dict={X:input_img,Y:0},
                                        layers='c', 
                                        path_logdir='Log',
                                        path_outdir='Output')

sess.close()

