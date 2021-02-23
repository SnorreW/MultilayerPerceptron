#!/usr/bin/env python
# coding: utf-8
# Made in jupyter notebook

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from keras.utils import to_categorical

#This part where I implement the dataset is mostly the same for both machine learning models.
#Implementing the MNIST dataset.
#Splitting the dataset into a training dataset and a testing dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#These are the features.
#See how the x values look. The x values are the images. There is 60 000 training images and 10 000 testing images.
#It is a 28x28 pixel image and each pixel has a value from 0 to 255 that makes the color.
print(x_train.shape)
print(x_test.shape)

#See how the first image looks.
print(x_train[0])

#See how the y values looks. The y values are the labels. Each has a value from 0 to 9 that is linked to its corresponding image.
print(y_train)
print(y_test)

#Using matplotlib to plot the first image. It looks like 5.
plt.imshow(x_train[0])
plt.show()

#Reshaping the data so it can fit in the model. x_train.shape[0] = 60 000 which makes the rows.
#This part is a little different from the CNN. The MLP cannot take a 4D array so I turned it into a 2D array by multiplying 28 and 28.
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

#Normalizing the data. In this case, it is unnecessary to have a value from 0 to 255 as it is not efficient. Instead I can give it a value from 0 to 1.
x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.astype('float32')
x_test /= 255

#See how the first y_training value looks.
print(y_train[0])

#This changes the y value to an array of ones and zeroes. 
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Visualizing what the first y value looks like now.
print(y_train[0])

#The machine learning Model(Multi-layer perceptron classifier).
#hidden_layer_sizes means number of neurons and hidden layers.
#activation='logisitc' means logistisc sigmoid function.
#alpha=1e-4 means the penalty perameter or the regulization term. 1e-4 = 0.0001.
#solver='sgd' means the solver for weight optimization. sgd is stochastic gradient descent.
#tol=1e-4 means the tolerance for the optimization. 1e-4 = 0.0001.
#random_state is random number generation.
#learning_rate_init is the initial learning rate.
#verbose means if it should print out messages containing the progress.
mlp = MLPClassifier(hidden_layer_sizes=(30,30), activation='logistic', alpha=1e-4, solver='sgd', 
                    tol=1e-4, random_state=1, learning_rate_init=.1, verbose=True)
#Training
#The fit method fits the model to x_train and y_train.
mlp.fit(x_train, y_train)

#Testing
#It predicts the 10 000 images in x_test using the multi-layer perceptron classifer that was made.
predictions = mlp.predict(x_test)
#Prints out what it thinks the first 51 images are.
print(predictions[:50])
#Prints out what the correct answers are.
print(y_test[:50])

#Prints out accuracy from 0 to 1. 
print(accuracy_score(y_test, predictions))

