import tf_cnnvis
import numpy as np 
import tensorflow as tf 
from deep_models import predict

model_dir = '/home/jay/Network-Comparisons/28x28_4000/model.ckpt-4000.meta'
model_stuff = '/home/jay/Network-Comparisons/28x28_4000/model.ckpt-4000'

img_path = '/home/jay/Network-Comparisons/test_images/cat.jpeg'
with tf.Session() as sess:
    # A test input image
    input_img = np.zeros([28,28,3])

    input_img = predict.load_image(img_path,[28,28,3])
    input_img = np.expand_dims(input_img, axis = 0)

    # Placeholder for feeding the hungry hungry model
    t_input = tf.placeholder(np.float32, 
                        shape=[None, 28, 28, 3], 
                        name='new_input') # define the input tensor

    # Need to feed the input into the 'input' node of the graph. 
    # This is done by specifying the input map.
    new_saver = tf.train.import_meta_graph(model_dir, input_map={'Reshape':  t_input})
    #new_saver = tf.train.import_meta_graph(model_dir)
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
                                        path_logdir='actLog',
                                        path_outdir='Output')
    
    tf_cnnvis.deconv_visualization(sess,value_feed_dict={t_input:input_img},
                                        layers='r', 
                                        path_logdir='deconLog',
                                        path_outdir='Output')
    
    for i in range(1,8):
        if i == 1:
            layer = 'conv2d/Relu'
        else:
            layer = 'conv2d_%d/Relu'%(i)
        tf_cnnvis.deepdream_visualization(sess,
                                value_feed_dict={t_input:input_img},
                                layer=layer,classes = [0,1,2,3,4],
                                path_logdir='Log', path_outdir='Output')
    
sess.close()

