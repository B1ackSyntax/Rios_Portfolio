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


# Project 2
# Recurrent Neural Network
## Sentiment Analysis using Binary Classification
### Packages: `Tensorflow`, `Keras`, `SkLearn`

**Project 2** will use the **Keras's** bundled **IMDB** *Internet Movie Database* movie reviews dataset to perform a **binary classification** to determine if a review is negative or positve.

**RNN**s process sequences of data, such as the text of a sentence. **Recurrent** means the neural network contains *loops* that causes the output of a given layer to become the input to the same layer in the next *time step*. A *time step* is the next point in time in a time series, a *time step* would be the next word in a sequence of words. Looping in **RNN**s enables learning and remembering relationships among the data in the sequence. 

For example condsider the following:
* The movie is not good

* The actor is good

* The actor is great!

The first sentence is cleary negative. The second is positive but not as positve as the third sentence. The word *good* in the first sentence has its own positive sentiment, however, when it follows the word *not* which appears before *good* in this sequence, the sentiment becomes negative. **RNNs** take into account the relationship among the earlier and later parts of a sequence. Determining the meaning of text can involve many words to consider and an unknown number of words between them. This project will use a **LSTM** *Long Short-Term Memory* layer to make the network **recurrent** and optimize learning from sequences like the ones described above.

**More Information:**
* __[Overview of Recurrent Neural Networks](https://www.analyticsindiamag.com/overview-of-recurrent-neural-networks-and-their-applications/)__
* __[Applications](https://en.wikipedia.org/wiki/Recurrent_neural_network#Applications)__
* __[Binary Clasification](https://docs.aws.amazon.com/machine-learning/latest/dg/binary-classification.html)__

Steps for this project:
* Explore data
* Decode data
* Prepare data
* Split data
* Compile model
* Train model
* Evaluate error
