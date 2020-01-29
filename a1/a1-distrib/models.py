# models.py

from sentiment_data import *
from utils import *
from collections import defaultdict
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np

stopwrds = set(stopwords.words('english'))

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param ex_words: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return:
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer, examples: List[SentimentExample]=None):
        self.indexer = indexer
        self.counts = defaultdict(int)
        if examples:
            self.load_examples(examples)

    def load_examples(self, examples: List[SentimentExample]):
        for ex in examples:
            self.count_and_index(ex.words)
    
    def get_indexer(self) -> Indexer:
        return self.indexer

    def count_and_index(self, words: List[str]):
        for word in words:
            if word.isalpha():
                lower_case = word.lower()
                if lower_case not in stopwrds:
                    self.indexer.add_and_get_index(lower_case)
                    self.counts[lower_case] += 1
            
    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        counts = []
        if add_to_indexer:
            self.count_and_index(ex_words)
        for word in ex_words:
            # TODO What if the word is not in the indexer and add_to_indexer is false
            counts.append(self.counts[word.lower()])
        return counts

    def __str__(self):
        return repr(self.counts)
            

                
        


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex_words: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, ex_words: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feature_extractor = feat_extractor
    
    def predict(self, ex_words: List[str]) -> int:
        feat_vec = self.feature_extractor.extract_features(ex_words)
        prod = weights_dot_features(self.weights, ex_words, 
                                    self.feature_extractor.get_indexer(), feat_vec)
        return 1 if prod > 0 else 0
        


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feature_extractor = feat_extractor
    
    def predict(self, ex_words: List[str]) -> int:
        feat_vec = self.feature_extractor.extract_features(ex_words)
        prod = weights_dot_features(self.weights, ex_words, 
                                    self.feature_extractor.get_indexer(), feat_vec)
        pos = cond_prob_label_features(prod, True)
        if pos > 0.5:
            return 1
        return 0

def weights_dot_features(weights, words: List[str], indexer: Indexer, 
                        feat_vec: List[int]) -> int:
    """
    Find the dot product between weights and a feature vector.
    :param WEIGHTS: fixed length array storing weights for every feature
    :param INDEX: for bijective mapping between WORDS and index
    :param WORDS: list of words in a sentence
    :param FEAT_VEC: feature weight corresponding one to one to WORDS
    :return WEIGHTS.FEAT_VEC
    """
    sum = 0
    for idx, word in enumerate(words):
        #If feat_vec[idx] == 0, its probably a stopword and wasn't stored in indexer
        if feat_vec[idx] != 0:
            weight_idx = indexer.index_of(word.lower())
            sum += feat_vec[idx] * weights[weight_idx]
    return sum


def weight_plus_features(weights, words: List[str], indexer: Indexer, \
                         feat_vec: List[int], subtract: bool=False):
    """
    Find the sum of weights and a feature vector.
    :param WEIGHTS: fixed length array storing weights for every feature
    :param INDEX: for bijective mapping between WORDS and index
    :param WORDS: list of words in a sentence
    :param FEAT_VEC: feature weight corresponding one to one to WORDS
    :return WEIGHTS + FEAT_VEC
    """
    for idx, word in enumerate(words):
        #If feat_vec[idx] == 0, its probably a stop word and wasn't stored in indexer
        sign = -1 if subtract else 1
        if feat_vec[idx] != 0:
            weight_idx = indexer.index_of(word.lower())
            weights[weight_idx] += (feat_vec[idx] * sign)

def feature_scale(feat_vec: List[int], alpha) -> List[int]:
    """
    Scale a feature vector by alpha
    """
    scaled = []
    for x in feat_vec:
        scaled.append(x * alpha)
    return scaled
    
def train_perceptron(train_exs: List[SentimentExample], 
                     feat_extractor: FeatureExtractor, EPOCHS: int = 50,
                     alpha: float = 1.0) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    indexer = feat_extractor.get_indexer()
    weights = np.zeros(len(indexer))
    for _ in range(EPOCHS):
        np.random.shuffle(train_exs)
        for ex in train_exs:
            feat_vec = feat_extractor.extract_features(ex.words)
            prod = weights_dot_features(weights, ex.words, indexer, feat_vec)
            predicted_label = 1 if prod > 0 else 0
            if predicted_label != ex.label:
                feat_vec = feature_scale(feat_vec, alpha) if alpha != 1.0 else feat_vec
                weight_plus_features(weights, ex.words, indexer, feat_vec, not ex.label)           

    return PerceptronClassifier(weights, feat_extractor)

def cond_prob_label_features(w_dot_feat: float, label: bool) -> float:
    """
    Finds P(label | X)
    :param w_dot_feat: The dot product of the weights and feature(x)
    :param label: Positive or negative label
    :return P(label| X)
    """
    exp = np.exp(w_dot_feat)
    if label:
        return exp / (1 + exp)   
    else:
        return 1 / (1 + exp)

def train_logistic_regression(train_exs: List[SentimentExample], 
                              feat_extractor: FeatureExtractor, EPOCHS: int=50,
                              alpha : float=1) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    indexer = feat_extractor.get_indexer()
    weights = np.zeros(len(indexer))
    for _ in range(EPOCHS):
        np.random.shuffle(train_exs)
        for ex in train_exs:
            feat_vec = feat_extractor.extract_features(ex.words)
            # f(ex) * -1 + P(y = ex.label | ex)
            prod = weights_dot_features(weights, ex.words, indexer, feat_vec)
            cond_prob = cond_prob_label_features(prod, bool(ex.label))
            feat_vec = feature_scale(feat_vec, alpha * (-1 + cond_prob))
            #update weights
            weight_plus_features(weights, ex.words, indexer, feat_vec, True)
    
    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer(), train_exs)
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model