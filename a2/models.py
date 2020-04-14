# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords

stopwrds = set(stopwords.words('english')) | set(string.punctuation)

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
        self.log_softmax = nn.LogSoftmax()
        self.embed_vec_size = len(embeddings.vectors[0])
        self.embedding = nn.Embedding(len(embeddings.vectors), self.embed_vec_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings.vectors))
        # self.embedding.weight.requires_grad_(False)

        # # Initialize weights according to a formula due to Xavier Glorot.
        # nn.init.xavier_uniform_(self.V.weight)
        # nn.init.xavier_normal_(self.W.weight)


    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        x = self.embedding(x)
        x = x.mean(1)

        # print("W", self.W.weight.size())
        # print("V", self.V.weight.size())
        # print("x", x.size())

        Vx = self.V(x)
        tanh_res = self.g(Vx)
        res = self.W(tanh_res)
        smax = self.log_softmax(res)
        
        return smax

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

    def predict(self, ex_words: List[str]):
        words = [word.lower() for word in ex_words]
        idxs, _ = get_indices([SentimentExample(ex_words, 0)], self.embed)
        probs = self.ffnn(idxs)
        probs = probs.squeeze(0)
        if probs[0] > probs[1]:
            return 0
        return 1

def create2D(m,n):
    """ Create a m x n dimension array"""
    ret = []
    for _ in range(m):
        ret.append([0] * n)
    return ret

def get_indices(exs: List[SentimentExample], embed: WordEmbeddings):
    """
    Get the indices of a batch of Sentiment examples EXS
    """
    indexer = embed.word_indexer
    max_len = float("-inf")
    for ex in exs:
        max_len = max(max_len, len(ex.words))
    idxs = create2D(len(exs), max_len)
    labels = []
    for i in range(len(exs)):
        ex_len = len(exs[i].words)
        for j in range(ex_len):
            idx = indexer.index_of(exs[i].words[j])
            idx = idx if idx != -1 else 1
            idxs[i][j] = idx
            # idxs[i][j] = idx if idx != -1 else indexer.index_of("UNK")
        for j in range(ex_len, max_len):
            idxs[i][j] = 0
            # idxs[i][j] = indexer.index_of("PAD")
        labels.append(exs[i].label)
    return torch.LongTensor(idxs),torch.LongTensor(labels)

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    epochs = args.num_epochs
    lr = args.lr
    input_size = len(word_embeddings.vectors[0])
    num_classes = 2
    batch_size = 64
    hidden_layer_size = args.hidden_size
    ffnn = FFNN(input_size, hidden_layer_size, num_classes, word_embeddings)
    optimizer = optim.Adam(ffnn.parameters(), lr=lr)
    ffnn.train()
    loss = nn.NLLLoss()

    for epoch in range(epochs):
        np.random.shuffle(train_exs)
        total_loss = 0.0
        for i in range(0, len(train_exs), batch_size):
            exs = train_exs[i : i+batch_size]
            idxs, labels = get_indices(exs, word_embeddings)
            ffnn.zero_grad()
            probs = ffnn.forward(idxs)
            print(probs)
            output = loss(probs, labels)
            total_loss += output.item()
            output.backward()
            optimizer.step()
        print("[DAN] total loss after epoch {}: {}".format(epoch+1, total_loss))
    return NeuralSentimentClassifier(ffnn, word_embeddings)

            

