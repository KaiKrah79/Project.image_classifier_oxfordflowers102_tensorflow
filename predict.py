import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image
import argparse


def get_input_args():
    """Setting up the argparse arguments for command line 
    --> https://docs.python.org/3/library/argparse.html
    
    Inputs: 
        None
    
    Outputs: 
        parser.parse_Arg()"""

    # Creating a parser for the commandline arguments
    parser = argparse.ArgumentParser(description = 'Command Line Application for predicting flowers based on the Oxford Flowers 102 dataset.')

    # Defining the command line arguments for the parser
    ## Argument for image_file to predict
    parser.add_argument('--image_file', type = str, default = 'test_images/cautleya_spicata.jpg', help = 'Imgage File, Default is test_images/cautleya_spicata.jpg')

    ## Argument for the trained model to use
    parser.add_argument('--saved_model', type = str, default = '1735894396_MobileNet.h5', help = 'Saved Keras Model, default Model is 1735894396_MobileNet.h5')

    ## Argument for the top-k classes to put out
    parser.add_argument('--topk', type = int, default = 1, help = 'Top k predicitons, default is 1.')

    ## Argument for the json file to load for label-mapping
    parser.add_argument('--json_file', type = str, default = 'label_map.json', help = 'JSON-File, default is label-map.json')

    # Returning the arguments
    return parser.parse_args()

def process_image(image):
    """Preprocesses an image and returns a numpy-array with shape (224, 224, 3)
    Input: 
        image - as a numpy-array
    Output: 
        image.numpy() - image as a numpy-array
    """

    #Converting the image to tensor, resize and normalize it
    image_size = 224
    image = tf.convert_to_tensor(image)#, dtype = tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255

    # Returning as a numpy-array
    return image.numpy()

def predict(image_path, model, top_k):
    """Predicts the class (or classes) and probabilities of an image using a trained model.
    Input:
        Path to an image file, a trained model, top k predicitons
    
    Output:
        probs, classes: topk probabilites and classes of the image-prediciton as numpy-array
    """

    # Image-Preprocessing
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    expanded_image = np.expand_dims(processed_image, axis = 0)

    # Prediction
    prediction = model.predict(expanded_image)
    
    # Getting the top_k classes and probabilities
    top_k_probs, top_k_classes = tf.nn.top_k(prediction, k = top_k)
    top_k_probs = list(top_k_probs.numpy()[0])
    top_k_classes = list(top_k_classes.numpy()[0])
    
    return top_k_probs, top_k_classes

# Main root
## Getting the input arguments for the prediction
in_arg = get_input_args()

## Getting the class value mapping with JSON-file
print(f'Loading the json-file {in_arg.json_file}')
with open(in_arg.json_file, 'r') as f:
    class_names = json.load(f)
num_classes = len (class_names)
print(f'JSON-File loaded with {num_classes} classes.')
print(f'----------------------')

## Loading the given model
print(f'Loading the model-file {in_arg.saved_model}.')
loaded_model = tf.keras.models.load_model(in_arg.saved_model, custom_objects = {'KerasLayer':hub.KerasLayer})
print(f'Model loaded, here ist the summary:')
print(loaded_model.summary())
print(f'----------------------')

## Predicting the given image
print(f'Predicting the given image.')
probs, classes = predict(in_arg.image_file, loaded_model, in_arg.topk)

##Changing the top k predicted classes to class_names
classes = [class_names[str(i)] for i in classes]
print(f'Image predicted.')
print(f'----------------------')

## Displaying the image and the models prediction
print(f'The image {in_arg.image_file} is predicted as:')
for i in range(in_arg.topk):
    print(f'{classes[i]:<30}{probs[i]*100:.2f} %')
print(f'----------------------')

# End of code