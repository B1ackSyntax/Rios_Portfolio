# Rios_Portfolio
Portfolio with Data Science examples

# Project 1:
# Convolutional Neural Network
## Classification using the MNIST Dataset
### Packages: `SkLearn`,`Pandas`, `Tensorflow`, `Keras`, `Seaborn`, `Numpy`

This example will use the **MNIST** handwritten digits dataset from **Keras** to explore *deep learning* with a **Convolutional Neural Network**, also known as a *convnet*. Convnets are common in computer-vision applications, such as recognizing objects in images and video, and in non-vision applications, such as *natural-language processing*. **MNIST** has 70,000 labeled digit image samples: 60,000 for *training* and 10,000 for *testing*. Each sample is a gray-scale 28 x 28 pixel image: 784 total features represented as a *NumPy* array. **MNIST's** labels are *integer values* in the range of 0 through 9, indicating the digit each image represents. This *convnet* will perform **probabilistic classification**. The model will output an *array* of 10 probabilities, indicating the likelihood that the digit belongs to a particular one of the classes 0 through 9. The class with the *highest* probability is the predicted value.

__[More info on CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)__

__[More info on Probalistic Classification](https://en.wikipedia.org/wiki/Probabilistic_classification)__

A **Keras** neural network will consist of the following:

* **Network** (*model*) - a sequence of *layers* containing the neurons used to learn from the samples. Each layer's neurons receive inputs, process them using an *activation function* and produce outputs. The data is fed into the network via an *input layer* that specifies the dimensions of the sample data. This is followed by *hidden layers* of neurons that implement the learning and an *output layer* that produces predictions. **Deep Learning** - The more layers that are *stacked*, the deeper the network is. 

* **Loss Function** - produces a measure of how well the network predicts the *target* values. Lower loss values indicate better predictions

* **Optimizer** - attempts to minimize the values produced by the loss function to tune the network to make better predictions

Steps for this project:

* Explore data
* Visualize data
* Prapare data
* Create Neural Network (testing, training)
* Evaluate
* Make predictions
* Review error
* Save model
