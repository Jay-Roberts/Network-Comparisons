import tf_cnnvis
import numpy as np 
import tensorflow as tf 



model_dir = '/home/jay/Network-Comparisons/models/3_28x28x3/test_scrn/model.ckpt-2.meta'
model_stuff = '/home/jay/Network-Comparisons/models/3_28x28x3/test_scrn/model.ckpt-2'

X = tf.placeholder(dtype=np.float32,shape = [1,28,28,3], name = 'xin')

with tf.Session() as sess:
    # A test input image
    input_img = np.zeros([28,28,3])
    input_img = np.expand_dims(input_img, axis = 0)

    # Placeholder for feeding the hungry hungry model
    t_input = tf.placeholder(np.float32, 
                        shape=[None, 28, 28, 3], 
                        name='new_input') # define the input tensor

    # Need to feed the input into the 'input' node of the graph. 
    # This is done by specifying the input map.
    new_saver = tf.train.import_meta_graph(model_dir, input_map={'input':  t_input})
    new_saver.restore(sess, model_stuff)
    
    vars = tf.get_collection('variables')
    stuff = [n.name for n in tf.get_default_graph().as_graph_def().node]

    TESTING = False
    if TESTING:
        for x in vars: 
            print(x)
        
        for x in stuff:
            print(x)
    tf_cnnvis.activation_visualization(sess,value_feed_dict={t_input:input_img},
                                        layers='r', 
                                        path_logdir='Log',
                                        path_outdir='Output')
    tf_cnnvis.deepdream_visualization(sess,
                            value_feed_dict={X:input_img},
                            layer='softmax_tensor',classes = [1],
                            path_logdir='DLog', path_outdir='DOutput')

sess.close()

