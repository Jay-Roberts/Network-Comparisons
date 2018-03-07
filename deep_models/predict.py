#---------------------------------------------------------------
#   
#   A collection of network blocks and other helper functions
#
#---------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os
import tensorflow as tf
import cv2
import csv

# Loads images
def load_image(addr,res):
    """
    Loads an image from a file as a float32 numpy array.It resizes to res and normalizes data.
    Inputs:
        addr: The file path to the image file. (str)
        res: Desired resolution including color channel. (list)
    Returns:
        Resized image as numpy array shape (res,3) with dtype np.float32 in range [-0.5,0.5]
        to fit with normalized input from training data.
    """

    # Check for color channels
    if res[-1] == 3:    
        img = cv2.imread(addr)
        # cv2 loads images as BGR, convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(addr,0)

    img = cv2.resize(img,tuple(list(res)[:-1]))
    img.astype(np.float32)
    # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
    img = img / 255 - 0.5    
    return img


# Feed images as np.arrays into the trained model
def predict_np(images,graph_path):
    """
    Takes a numpy image and uses the model from graph_path to predict what label it is. Returns a
    dictionary of infered games and the confidence probabilities.
    Inputs:
        images: Numpy arrays to be classified. Must fit the resolution for the graph. (list)
        graph_path: relative path to the trained model .pb file. (str)
    Returns:
        Dict with keys {inferences,confidences}.
            inferences: int, The softmax guess corresponding to the inferred label for the image.
            confidences: A list where each element is a list with the probability the image corresponds to a given label.
    """
    
    # Use the predictor loader to load from .pb files
    saved_model_predictor = tf.contrib.predictor.from_saved_model(export_dir=graph_path)
    
    num_imgs =images.shape[0]

    inferences, confidences = [0]*num_imgs,[0]*num_imgs
    for j in range(num_imgs):
        # Get image
        img = images[j,:,:,:]
        # Predictor HUNGRY for dictionary with one image
        input_dict = {'x': img}
        # Predict it
        output_dict =saved_model_predictor(input_dict)

        inferences[j], confidences[j]= output_dict["classes"], list(output_dict["probabilities"][0])
    
    # Unpack
    inferences = [inf[0] for inf in inferences]
    
    # Make return dictionary
    results = {'inferences':inferences,'confidences': confidences}
    return results
    

# Take a list of img file names and returns predictions
def predict_imgs(images,graph_path,res=(28,28,3)):
    """
    Takes list of img file names and returns dictionary of infrered label and confidence probabilites.
    Inputs:
        images: File names of image as str. (list)
        graph_path: Relative directory contaning the saved .pb model file. (str)
        res: Resolution of image required for the model to predict. (tuple)
    Returns:
        Dict with keys {names,inferences,confidences}.
            names: Name of file wihtout directory tree or file extension. (str)
            inferences: The softmax guess corresponding to the inferred label for the image. (int)
            confidences: Probabilities the image corresponds to a given label. (list)
    """

    num_imgs = len(images)
    # Get file names w/o dir
    names = [img_path.split('/')[-1] for img_path in images]
    # Get file names w/o extension
    names = [[name.split('.')[0]] for name in names]

    # Init input
    
    input_array = np.zeros((num_imgs,res[0],res[1],res[2]))

    # Fill input
    for img_ix in range(num_imgs):
        img_path = images[img_ix]

        img = load_image(img_path,res)
        if len(img.shape) == 2:
            img.shape = res
        
        input_array[img_ix,:,:,:] = img
    
    results = predict_np(input_array,graph_path)
    
    # Add names
    results['names'] = [name[0] for name in names]

    return results

def mk_prediction(save_dir,labels_dict,res,
                    data_dir='test_images',
                    out_name = 'predicitons'
                    ):
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

    labels_key = labels_dict
    #wrk_dir = os.getcwd()
    #graph_path = '/'.join([wrk_dir,save_dir])
    graph_path = save_dir
    print('Graph path: %s'%(graph_path))

    # Get image directory and image path names
    images = os.listdir(data_dir)
    images = ['/'.join([data_dir,img]) for img in images]

    num_imgs = len(images)
    names = list(labels_key['NAME'])

    # Get prediction results
    results = predict_imgs(images,graph_path,res=res)

    # Write the csv
    with open(out_name+'.csv','wb') as outfile:
        fieldnames = results.keys()
        writer = csv.writer(outfile)
        
        # Write the header
        writer.writerow(fieldnames)

        # Assumes each key has same number of elements
        for i in range(num_imgs):
            name, confidences, inferences  = results['names'][i], results['confidences'][i], results['inferences'][i]
            writer.writerow([name,confidences,inferences])
        

    
    #results_df = pd.DataFrame(results, columns=columns)
    # Round the probabilities 
    #results_df[names] = results_df[names].apply(lambda x: pd.Series.round(x,4))

    # Save the results
    #results_df.to_csv(out_name+'.csv')
    #print(results_df.head())
