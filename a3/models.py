# models.py

from treedata import *
from utils import *
from collections import Counter
from typing import List

import numpy as np


class BadTaggingModel(object):
    """
    Assigns each word to its most common tag in the training set. Unknown words are classed as NN.
    """
    def __init__(self, words_to_tag_counters):
        self.words_to_tag_counters = words_to_tag_counters

    def decode(self, sentence: List[str]) -> List[str]:
        """
        :param sentence: the sequence of words to tag
        :return: the list of tags, which must match the length of the sentence
        """
        pred_tags = []
        for word in sentence:
            if word in self.words_to_tag_counters:
                pred_tags.append(self.words_to_tag_counters[word].most_common(1)[0][0])
            else:
                pred_tags.append("NN") # unks are often NN
        return labeled_sent_from_words_tags(sentence, pred_tags)


def train_bad_tagging_model(training_set):
    """
    :param training_set: The list of LabeledSentence objects from which to estimate the model
    :return: a BadTaggingModel estimated by counting word-tag pairs
    """
    words_to_tag_counters = {}
    for sentence in training_set:
        for ttok in sentence.tagged_tokens:
            if not ttok.word in words_to_tag_counters:
                words_to_tag_counters[ttok.word] = Counter()
            words_to_tag_counters[ttok.word][ttok.tag] += 1.0
    return BadTaggingModel(words_to_tag_counters)


class HmmTaggingModel(object):
    """
    Fields:
    tag_indexer: maps tags to ints so we know where to look in the parameter matrices for the params associated with that tag
    word_indexer: maps words to ints in the same fashion
    init_log_probs: A [num tags]-length numpy array containing initial log probabilities for each tag (i.e., P(y0)).
    transition_log_probs: A [num-tags x num-tags]-size numpy array containing transitions, where rows correspond to
    the previous tag and columns correspond to the current tag.
    emission_log_probs: A [num-tags x num-words]
    
    score_init, score_transition, and score_emission are provided to assist you with using these
    """
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, tag_idx):
        return self.init_log_probs[tag_idx]

    def score_transition(self, prev_tag_idx, curr_tag_idx):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence, tag_idx, word_posn):
        word = sentence[word_posn]
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]

    def viterbi_decode(self, sentence: List[str]) -> LabeledSentence:
        """
        :param sentence: the words to tag
        :return: a LabeledSentence containing the model's predictions. See BadTaggingModel for an example.
        """
        n_tags = len(self.tag_indexer)
        n_words = len(sentence)
        vt = np.zeros((n_words, n_tags))
        backtags = [0] * n_words
        max_tag_idx = 0
        max_word_tag = float("-inf")
        for i in range(n_tags):
            vt[0][i] = self.score_emission(sentence, i, 0) + self.score_init(i)
            if vt[0][i] > max_word_tag:
                max_tag_idx = i
                max_word_tag = vt[0][i]
        backtags[0] = self.tag_indexer.get_object(max_tag_idx)

        for i in range(1, n_words):
            max_word_tag = float("-inf")
            max_tag_idx = 0
            for j in range(n_tags):
                max_prev_tag = float("-inf")
                for k in range(n_tags):
                    max_prev_tag = max(max_prev_tag, self.score_transition(k, j) + vt[i-1][k])
                vt[i][j] = self.score_emission(sentence, j, i) + max_prev_tag
                if vt[i][j] > max_word_tag:
                    max_tag_idx = j
                    max_word_tag = vt[i][j]
            backtags[i] = self.tag_indexer.get_object(max_tag_idx)
        return labeled_sent_from_words_tags(sentence, backtags)
            

    def beam_decode(self, sentence: List[str]) -> LabeledSentence:
        """
        :param sentence: the words to tag
        :return: a LabeledSentence containing the model's predictions. See BadTaggingModel for an example.
        """
        raise Exception("Implement me")


def train_hmm_model(sentences: List[LabeledSentence]):
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied to all probabilities
    :param sentences: corpus of tagged sentences to read probabilities from
    :return:
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tagged_tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tagged_tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_tags():
            tag_indexer.add_and_get_index(tag)
    tag_indexer.add_and_get_index("STOP")
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        tags = sentence.get_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.index_of(tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tagged_tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_indexer.index_of(tags[i])] += 1.0
            else:
                transition_counts[tag_indexer.index_of(tags[i-1])][tag_idx] += 1.0
        transition_counts[tag_indexer.index_of(tags[-1])][tag_indexer.index_of("STOP")] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print("Tag indexer: %s" % tag_indexer)
    print("Initial state log probabilities: %s" % init_counts)
    print("Transition log probabilities: %s" % transition_counts)
    print("Emission log probs too big to print...")
    print("Emission log probs for India: %s" % emission_counts[:,word_indexer.index_of("India")])
    print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.index_of("Phil")])
    print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmTaggingModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer, word_counter, word):
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: mapping from words to indices
    :param word_counter: counter of train word counts
    :param word: word to check
    :return: index of word, or UNK if it's a singleton
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)
