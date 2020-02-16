# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *

class FFNN(nn.Module):
    """
    Defines the core neural network for doing multiclass classification over a single datapoint at a time. This consists
    of matrix multiplication, tanh nonlinearity, another matrix multiplication, and then
    a log softmax layer to give the ouputs. Log softmax is numerically more stable. If you take a softmax over
    [-100, 100], you will end up with [0, 1], which if you then take the log of (to compute log likelihood) will
    break.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, inp, hid, out, embeddings: WordEmbeddings):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(FFNN, self).__init__()
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.embed_vec_size = len(embeddings.vectors[0])
        self.embedding = nn.Embedding(len(embeddings.vectors), self.embed_vec_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings.vectors))
        self.embedding.weight.requires_grad_(False)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
    
    def avg_word_vecs(self, word_idx: List[int]):
        retval = torch.zeros(self.embed_vec_size)
        idxs = torch.tensor(word_idx, dtype=torch.long)
        embeddings = self.embedding(idxs)


    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        return self.log_softmax(self.W(self.g(self.V(x))))

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]):
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]):
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, ffnn: nn.Module, embeddings: WordEmbeddings):
        self.ffnn = ffnn
        self.embed = embeddings



def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    epochs = args.epochs
    lr = args.lr
    input_size = len(word_embeddings.vectors[0])
    num_classes = 2
    hidden_layer_size = 10
    ffnn = FFNN(input_size, hidden_layer_size, num_classes, word_embeddings)
    optimizer = optim.Adam(ffnn.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for ex in train_exs:
            avg_vec = avg_word_vecs(ex.words, word_embeddings)
            ffnn.zero_grad()
            probs = ffnn.forward(avg_vec)
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(ex.label,dtype=np.int64)), 1)
            loss = nn.CrossEntropyLoss()
            output = loss(probs, y_onehot)
            total_loss += output
            loss.backward()
            optimizer.step()
        print("[DAN] total loss after epoch {}: {}".format(epoch, total_loss))

            
    raise NotImplementedError

