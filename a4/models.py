# models.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import collections

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred 
    with more consonants than vowels, classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

def get_indices(s: str, vocab_index):
        return [vocab_index.index_of(char) for char in s]

class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, rnn, vocab_index):
        self.rnn = rnn
        self.indexer = vocab_index

    def predict(self, context):
        idxs = torch.tensor(get_indices(context, self.indexer))
        probs = self.rnn(idxs.unsqueeze(0))
        probs = probs.squeeze(0)
        relevant_prob = probs[19]
        if relevant_prob[0] > relevant_prob[1]:
            return 0
        return 1


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)

class RNNOverChars(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, output_dim):
        super(RNNOverChars, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.embeddings.weight.requires_grad_(True)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.W = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=2)

    """
        input x: this should be a tensor of indices corresponding to the index
        of each character. The embeddings should return a |x| x embed_size
        Tensor. We can unsqueeze or view this into an extra third dimensions
    """
    def forward(self, x):
        n = len(x)
        embeds = self.embeddings(x)
        output, _ = self.lstm(embeds)
        return self.log_softmax(self.W(output))

def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    n = len(train_vowel_exs)
    EMBED_DIM = 15
    HIDDEN_DIM = 5
    LR = 0.01
    EPOCHS = 30
    BATCH_SIZE = 32
    OUTPUT_DIM = 2
    rnn = RNNOverChars(EMBED_DIM, HIDDEN_DIM, len(vocab_index), OUTPUT_DIM)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=LR)
    np.random.shuffle(train_cons_exs)
    np.random.shuffle(train_vowel_exs)

    def train_batch(strs: List[str], vowel: bool):
        rnn.zero_grad()
        idxs = torch.tensor([get_indices(s, vocab_index) for s in strs], dtype=torch.long)
        probs = rnn(idxs)
        prob_splice = probs[:, 19:20,]
        loss = loss_function(prob_splice.view(len(strs), 2), torch.tensor([int(vowel)] * len(strs), dtype=torch.long))
        loss.backward()
        optimizer.step()
        return loss.item()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            total_loss += train_batch(train_cons_exs[i:i+BATCH_SIZE], False)
            total_loss += train_batch(train_vowel_exs[i:i+BATCH_SIZE], True)
        print("Loss after epoch {} = {}".format(epoch, total_loss))
    return RNNClassifier(rnn, vocab_index)


#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_log_prob_single(self, next_char, context):
        """
        Scores one character following the given context. That is, returns
        log P(next_char | context)
        The log should be base e
        :param next_char:
        :param context: a single character to score
        :return:
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return:
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, rnn, vocab_indexer):
        self.rnn = rnn
        self.indexer = vocab_indexer

    def get_log_prob_single(self, next_char, context):
        # context.append(next_char)
        idxs = torch.tensor(get_indices(context, self.indexer), dtype=torch.long)
        probs = self.rnn(idxs.unsqueeze(0))
        probs = probs.squeeze(0)
        relevant_probs = probs[len(context)-1]
        return relevant_probs[self.indexer.index_of(next_char)]

    def get_log_prob_sequence(self, next_chars, context):
        log_prob = 0.0
        m = len(next_chars)
        context = list(context + next_chars[0:m-1])
        n = len(context)
        idxs = torch.tensor(get_indices(context, self.indexer), dtype=torch.long)
        probs = self.rnn(idxs.unsqueeze(0))
        probs = probs.squeeze(0)
        i = n - m
        for char in next_chars:
            log_prob += probs[i][self.indexer.index_of(char)]
            i += 1
        return log_prob.item()


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    train_text = " " + train_text
    CHUNK_SIZE = 20
    EMBED_DIM = 27
    HIDDEN_DIM = 25
    LR = 0.01
    EPOCHS = 25
    BATCH_SIZE = 32
    OUTPUT_DIM = 27
    rnn = RNNOverChars(EMBED_DIM, HIDDEN_DIM, len(vocab_index), OUTPUT_DIM)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=LR)

    def create_batch(start):
        idxs = []
        labels = []
        for i in range(start, start + CHUNK_SIZE * BATCH_SIZE, CHUNK_SIZE):
            # if remaining text less than chunk size do padding
            if len(train_text) - i >= CHUNK_SIZE:
                idxs.append(get_indices(train_text[i:i+CHUNK_SIZE-1], vocab_index))
                labels.append(get_indices(train_text[i+1:i+CHUNK_SIZE], vocab_index))
            else:
                if i >= len(train_text):
                    break
                print("oof")
                rem = len(train_text) - i
                label = train_text[-1]
                sub_idx = get_indices(train_text[i:i+rem-1], vocab_index)
                idxs.append([vocab_index.index_of(" ")] * (CHUNK_SIZE-rem) + sub_idx)
                labels.append(vocab_index.index_of(label))
                break
        return torch.tensor(idxs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


    def train_batch(start):
        rnn.zero_grad()
        idxs, labels = create_batch(start)
        probs = rnn(idxs)
        probs = torch.transpose(probs, 1, 2)
        # prob_splice = probs[:, CHUNK_SIZE-2:CHUNK_SIZE-1,]
        # loss = loss_function(prob_splice.view(len(idxs), OUTPUT_DIM), labels)
        loss = loss_function(probs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    for i in range(EPOCHS):
        total_loss = 0.0
        for j in range(0, len(train_text), CHUNK_SIZE*BATCH_SIZE):
            total_loss += train_batch(j)
        print("LM los after epoch {} is {}".format(i+1, total_loss))


    return RNNLanguageModel(rnn, vocab_index)