from heapq import heappop, heappush
from collections import defaultdict
import nltk
from enum import Enum
from spacy.lang.en import English
# nlp = English()
# tokenizer = nlp.Defaults.create_tokenizer(nlp)
nltk.download('punkt')

class Tokenizer(Enum):
    DEFAULT=0
    NLTK=1  
    SPACY=2
def tokenize_line(line, tknzr):
    if tknzr == Tokenizer.NLTK:
        return nltk.word_tokenize(line)
    if tknzr == Tokenizer.DEFAULT:
        return line.split()
    if tknzr == Tokenizer.SPACY:
        a = list(tokenizer(line))
        return a

def get_most_freq(file, tokeniser=None):
    counts = defaultdict(int)
    for line in file:
        split = tokenize_line(line, tokeniser)
        for word in split:
            counts[str(word)] += 1
    q = []
    for word, count in counts.items():
        if len(q) < 10:
            heappush(q, (count, word))
        else:
            min_count, min_word = heappop(q)
            if min_count < count:
                heappush(q, (count, word))
            else:
                heappush(q, (min_count, min_word))
    return q
def print_top10(top10):
    for count, word in top10:
            print("{}: {}".format(word, count))
if __name__ == "__main__":
    with open('nyt.txt', 'r') as fi:
        # number 1
        top10 = get_most_freq(fi, Tokenizer.DEFAULT)
        print("1.)\n")
        
    

