# Transfer-Learning-Using-Keras-For-Image-Recognition
In this project I have done transfer learning exercise to train
a model for image recognition on the chest X-ray dataset

I used the Keras data loader to read the train and test images
as shown here https://keras.io/api/preprocessing/image/.

I have developed two files train.py and test.py.

train.py takes two inputs: the input training directory
and a model file name to save the model to.

test.py take two inputs: the test directory
and a model file name to load the model.

The output of test.py is the test error of the data which is
the number of misclassifications divided by size of the test set. 
I achieved test accuracy of 92.1% in this project
