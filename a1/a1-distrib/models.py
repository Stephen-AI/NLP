# models.py

from sentiment_data import *
from utils import *
from collections import defaultdict, Counter
import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from enum import Enum
from heapq import heappop, heappush
import matplotlib.pyplot as plt


stopwrds = set(stopwords.words('english')) | set(string.punctuation)
SMOL = 1e-8

class StepType(Enum):
    CONSTANT=0
    FIXED_FACTOR=1
    HARMONIC=2
    

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

    def sentence_preprocess(self, words: List[str]) -> List[str]:
        """
            Preprocess a sentence by filtering or making it into bigrams
        """
        raise Exception("Don't call me, call my subclasses")

class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer, examples: List[SentimentExample]=None):
        self.indexer = indexer
        if examples:
            self.load_examples(examples)

    def load_examples(self, examples: List[SentimentExample]):
        for ex in examples:
            self.index_word(ex.words)
    
    def get_indexer(self) -> Indexer:
        return self.indexer

    def index_word(self, words: List[str]):
        filtered_words = self.sentence_preprocess(words)
        for word in filtered_words:
            self.indexer.add_and_get_index(word)
    
    def sentence_preprocess(self, words: List[str]):
        filtered = list(filter(lambda word: word not in stopwrds, words))
        return list([word.lower() for word in filtered])
    
    # assume ex_words has been stripped of stop words
    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        if add_to_indexer:
            self.index_word(ex_words)
        counter = Counter(ex_words)
        return list([counter[word] for word in ex_words])
             

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer, examples: List[SentimentExample]=None):
        self.indexer = indexer
        if examples:
            self.load_examples(examples)
    
    def load_examples(self, examples: List[SentimentExample]):
        for example in examples:
            self.index_bigram(example.words)
    
    def index_bigram(self, words: List[str]):
        for bigram in self.sentence_preprocess(words):
            self.indexer.add_and_get_index(bigram)

    def get_indexer(self):
        return self.indexer
    
    def sentence_preprocess(self, words: List[str]) -> List[str]:
        i, j = [0, 1]
        filtered_words = list(filter(lambda word: word not in stopwrds, words))
        bigrams = []
        while j < len(filtered_words):
            bigrams.append("{} {}".format(filtered_words[i].lower(), 
                                            filtered_words[j].lower()))
            i += 1
            j += 1
        return bigrams

    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        counter = Counter(ex_words)
        return list([counter[bigram] for bigram in ex_words])


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer, examples: List[SentimentExample]=None):
        self.indexer = indexer
        self.idfs = defaultdict(int)
        if examples:
            self.load_examples(examples)
        for term, idf in self.idfs.items():
            self.idfs[term] = np.log(len(examples)/idf)


    def load_examples(self, examples: List[SentimentExample]):
        for ex in examples:
            self.index_word(ex.words)
    
    def get_indexer(self) -> Indexer:
        return self.indexer

    def index_word(self, words: List[str]):
        filtered_words = self.sentence_preprocess(words)
        visited = set()
        for word in filtered_words:
            self.indexer.add_and_get_index(word)
            if word not in visited:
                visited.add(word)
                self.idfs[word] += 1
    
    def sentence_preprocess(self, words: List[str]):
        filtered = list(filter(lambda word: word not in stopwrds, words))
        return list([word.lower() for word in filtered])
    
    def get_word_idf(self, word: str) -> float:
        res = self.idfs[word]
        return res if res != 0 else 1

    # assume ex_words has been stripped of stop words
    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        if add_to_indexer:
            self.index_word(ex_words)
        counter = Counter(ex_words)
        return list([counter[word] * self.get_word_idf(word) for word in ex_words])



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
        words = self.feature_extractor.sentence_preprocess(ex_words)
        feat_vec = self.feature_extractor.extract_features(words)
        
        prod = weights_dot_features(self.weights, words, 
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
        words = self.feature_extractor.sentence_preprocess(ex_words)
        feat_vec = self.feature_extractor.extract_features(words)
        prod = weights_dot_features(self.weights, words, 
                                    self.feature_extractor.get_indexer(), feat_vec)
        pos = cond_prob_label_features(prod)
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
            weight_idx = indexer.index_of(word)
            prod = 0
            if weight_idx >= 0:
                prod = feat_vec[idx] * weights[weight_idx]
            sum += prod
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
            weight_idx = indexer.index_of(word)
            if weight_idx >= 0:
                weights[weight_idx] += (feat_vec[idx] * sign)
            

def feature_scale(feat_vec: List[int], alpha) -> List[int]:
    """
    Scale a feature vector by alpha
    """
    scaled = []
    for x in feat_vec:
        scaled.append(x * alpha)
    return scaled

def print_top_10(weights, indexer: Indexer, positive: bool=True):
    q = []
    sign = 1 if positive else -1
    for i in range(len(weights)):
        if len(q) < 10:
            heappush(q, (sign * weights[i], i))
        else:
            least, idx = heappop(q)
            if weights[i] * sign > least:
                heappush(q, (sign * weights[i], i))
            else:
                heappush(q, (least, idx))
    q = sorted(q)
    i = 1
    print("top 10 most", "positive" if positive else "negative")
    for weight, idx in q:
        print("{}.) {}:{}".format(i, indexer.get_object(idx), weight))
        i += 1
    print()
        
            

    
    
def train_perceptron(train_exs: List[SentimentExample], 
                    feat_extractor: FeatureExtractor, EPOCHS: int=30,
                    alpha : float=np.exp(-4),
                    step=StepType.CONSTANT) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    indexer = feat_extractor.get_indexer()
    weights = np.zeros(len(indexer))
    for i in range(EPOCHS):
        np.random.shuffle(train_exs)
        if step == StepType.FIXED_FACTOR:
            alpha = 1 / (10 ** i)
        elif step == StepType.HARMONIC:
            alpha = 1 / (i+1)
        for ex in train_exs:
            words = feat_extractor.sentence_preprocess(ex.words)
            feat_vec = feat_extractor.extract_features(words)
            prod = weights_dot_features(weights, words, indexer, feat_vec)
            predicted_label = 1 if prod > 0 else 0
            if predicted_label != ex.label:
                feat_vec = feature_scale(feat_vec, alpha) if alpha != 1.0 else feat_vec
                weight_plus_features(weights, words, indexer, feat_vec, not ex.label)           
    # print_top_10(weights, indexer)
    # print_top_10(weights, indexer, False)
    return PerceptronClassifier(weights, feat_extractor)

def cond_prob_label_features(w_dot_feat: float) -> float:
    """
    Finds P(label=1 | X)
    :param w_dot_feat: The dot product of the weights and feature(x)
    :return P(label=1| X)
    """
    if w_dot_feat > 709:
        return 1e-8
    exp = np.exp(w_dot_feat)
    return exp / (1 + exp)    

def nll(y, p):
    m = -y * np.log(SMOL + p)
    n = (1-y) * np.log(SMOL + (1-p))
    return m - n

def train_logistic_regression(train_exs: List[SentimentExample], 
                              feat_extractor: FeatureExtractor, EPOCHS: int=30,
                              alpha : float=np.exp(-4),
                              step=StepType.CONSTANT) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    indexer = feat_extractor.get_indexer()
    weights = np.zeros(len(indexer))

    for i in range(EPOCHS):
        if step == StepType.FIXED_FACTOR:
            alpha = 1 / (10 ** i)
        elif step == StepType.HARMONIC:
            alpha = 1 / (i+1)
        np.random.shuffle(train_exs)
        # sum = 0.0
        for ex in train_exs:
            words = feat_extractor.sentence_preprocess(ex.words)
            feat_vec = feat_extractor.extract_features(words)
            prod = weights_dot_features(weights, words, indexer, feat_vec)
            pos_prob = cond_prob_label_features(prod)
            # nl = nll(ex.label, pos_prob)
            # sum += nl
            feat_vec = feature_scale(feat_vec, alpha * (pos_prob - ex.label))
            #update weights
            weight_plus_features(weights, words, indexer, feat_vec, True)
        # print("NLL for epoch {}: {}".format(i, sum/len(train_exs)))

    return LogisticRegressionClassifier(weights, feat_extractor)

def dll_avg(model: LogisticRegressionClassifier, exs: List[SentimentExample]):
    loss = 0
    for ex in exs:
        feat_extractor = model.feature_extractor
        words = feat_extractor.sentence_preprocess(ex.words)
        feat_vec = feat_extractor.extract_features(words)
        dot_prod = weights_dot_features(model.weights, words, 
                                        feat_extractor.indexer, feat_vec)
        prob = cond_prob_label_features(dot_prod)
        prob = prob if ex.label else 1 - prob
        loss += np.log(prob)
    return loss / len(exs)


def get_accuracy_count(model: SentimentClassifier, dev_ex: List[SentimentExample]):
    count = 0
    for ex in dev_ex:
        if not (model.predict(ex.words) ^ ex.label):
            count += 1
    return count / len(dev_ex)

def plot_LR_accuracies(train_ex: List[SentimentExample], dev_ex: List[SentimentExample]):
    feat_extractor = UnigramFeatureExtractor(Indexer(), train_ex)
    epochs = [1, 10, 20, 30, 50]
    schedules = [StepType.CONSTANT, StepType.FIXED_FACTOR, StepType.HARMONIC]
    
    plt.rcParams.update({'font.size': 15})
    for step in schedules:
        accuracies = []
        dlls = []
        _, ax = plt.subplots()
        for epoch in epochs:
            model = train_logistic_regression(train_ex, feat_extractor, epoch, step=step)
            acc = get_accuracy_count(model, dev_ex)
            dll = dll_avg(model, train_ex)
            accuracies.append(acc)
            dlls.append(dll)
        ax.set_xlabel('Epochs')
        ax.plot(epochs, dlls, label="Epoch vs Average Loss")
        ax.plot(epochs, accuracies, label="Epoch vs Dev Accuracy")
        plt.savefig("{}.png".format(str(step)),dpi=150)
               

def train_model(args, train_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    def get_step_type():
        if args.fixed_factor:
            return StepType.FIXED_FACTOR
        if args.harmonic:
            return StepType.HARMONIC
        return StepType.CONSTANT

    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer(), train_exs)
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer(),train_exs)
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer(), train_exs)
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor, EPOCHS=args.epochs,
                                 alpha=args.alpha)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor, 
                                          EPOCHS=args.epochs, 
                                          alpha=args.alpha, step=get_step_type())
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model

    