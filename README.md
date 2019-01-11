# Sentiment Classification

In this section, we classify a sentence as a positive or negative sentiment using a 3-layered Neural Network in Tensorflow.

We first create a lexicon (dictionary) of all the words in our positive and negative sentiment files. Then we create the feature sets and labels for the training and testing data. </br> The code is available in the file: Sentiment_featureset.py.

Then we create a neural network model and train and test the data using tensorflow. </br> The code is available in the file: Sentiment_classification.py.

The accuracy achieved is about 60%. The accuracy for this model is less because we are using the classic Neural Network model. If we use Recurrent Neural Network (RNN) for this problem, we will achieve a better accuracy as RNN gives better results for sequential data and our input feature here i.e. sentence is a sequence of words.

The data is available at: </br>
https://pythonprogramming.net/static/downloads/machine-learning-data/pos.txt - Positive sentiment
https://pythonprogramming.net/static/downloads/machine-learning-data/neg.txt - Negative sentiment
