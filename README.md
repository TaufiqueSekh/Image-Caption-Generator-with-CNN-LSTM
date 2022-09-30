Q.What is Image Caption Generator?
Ans :Image caption generator is a task that involves computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English.

Image Caption Generator with CNN – About the Python based Project
The objective of our project is to learn the concepts of a CNN and LSTM model and build a working model of Image caption generator by implementing CNN with LSTM.

In this Python project, we will be implementing the caption generator using CNN (Convolutional Neural Networks) and LSTM (Long short term memory). The image features will be extracted from Xception which is a CNN model trained on the imagenet dataset and then we feed the features into the LSTM model which will be responsible for generating the image captions.

The Dataset of Python based Project
For the image caption generator, we will be using the Flickr_8K dataset. There are also other big datasets like Flickr_30K and MSCOCO dataset but it can take weeks just to train the network so we will be using a small Flickr8k dataset. The advantage of a huge dataset is that we can build better models.

A direct link to download the dataset (Size: 1GB).
Flicker8k_Dataset : https://drive.google.com/drive/folders/1pVfNAsuYEKga5QVSydH1NexKJv9z4HCE?usp=sharing

Flickr_8k_text : https://drive.google.com/drive/folders/1lzFWBy81cR0NaMFCEG3kDeGRF1BifPqX?usp=sharing
The Flickr_8k_text folder contains file Flickr8k.token which is the main file of our dataset that contains image name and their respective captions separated by newline(“\n”).


#Pre-requisites
This project requires good knowledge of Deep learning, Python, working on Jupyter notebooks, Keras library, Numpy, and Natural language processing.

Make sure you have installed all the following necessary libraries:

pip install tensorflow
keras
pillow
numpy
tqdm
jupyterlab
Image Caption Generator – Python based Project


Q. What is CNN?
Ans >Convolutional Neural networks are specialized deep neural networks which can process the data that has input shape like a 2D matrix. Images are easily represented as a 2D matrix and CNN is very useful in working with images.

CNN is basically used for image classifications and identifying if an image is a bird, a plane or Superman, etc.

working of Deep CNN - Python based project

It scans images from left to right and top to bottom to pull out important features from the image and combines the feature to classify images. It can handle the images that have been translated, rotated, scaled and changes in perspective.


What is LSTM?
LSTM stands for Long short term memory, they are a type of RNN (recurrent neural network) which is well suited for sequence prediction problems. Based on the previous text, we can predict what the next word will be. It has proven itself effective from the traditional RNN by overcoming the limitations of RNN which had short term memory. LSTM can carry out relevant information throughout the processing of inputs and with a forget gate, it discards non-relevant information.

This is what an LSTM cell looks like –

LSTM Cell Structure - simple python project

Image Caption Generator Model

So, to make our image caption generator model, we will be merging these architectures. It is also called a CNN-RNN model.

CNN is used for extracting features from the image. We will use the pre-trained model Xception.
LSTM will use the information from CNN to help generate a description of the image.
Model of Image Caption Generator - python based project

Downloaded from dataset:

Flicker8k_Dataset – Dataset folder which contains 8091 images.
Flickr_8k_text – Dataset folder which contains text files and captions of images.
The below files will be created by us while making the project.

Models – It will contain our trained models.
Descriptions.txt – This text file contains all image names and their captions after preprocessing.
Features.p – Pickle object that contains an image and their feature vector extracted from the Xception pre-trained CNN model.
Tokenizer.p – Contains tokens mapped with an index value.
Model.png – Visual representation of dimensions of our project.
Testing_caption_generator.py – Python file for generating a caption of any image.
Training_caption_generator.ipynb – Jupyter notebook in which we train and build our image caption generator.
You can download all the files from the link:

Image Caption Generator – Python Project Files

structure - python based project




Building the Python based Project
Let’s start by initializing the jupyter notebook server by typing jupyter lab in the console of your project folder. It will open up the interactive Python notebook where you can run your code. Create a Python3 notebook and name it training_caption_generator.ipynb

jupyter lab - python based project 

1. First, we import all the necessary packages
2. Getting and performing data cleaning
3. Extracting the feature vector from all images 
4. Loading dataset for Training the model
5. Tokenizing the vocabulary 
6. Create Data generator
7. Defining the CNN-RNN model
8. Training the model
9. Testing the model



