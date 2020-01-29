from heapq import heappop, heappush
from collections import defaultdict
import nltk
from enum import Enum
from spacy.lang.en import English
import matplotlib.pyplot as plt

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

def get_most_freq(filename, tokeniser=None, N=10):
    with open(filename, 'r') as file:
        counts = defaultdict(int)
        for line in file:
            split = tokenize_line(line, tokeniser)
            for word in split:
                word = str(word)
                if word.isalpha():
                    counts[word.lower()] += 1
        q = []
        for word, count in counts.items():
            if len(q) < N:
                heappush(q, (count, word))
            else:
                min_count, min_word = heappop(q)
                if min_count < count:
                    heappush(q, (count, word))
                else:
                    heappush(q, (min_count, min_word))
        return q

def print_topN(lst, N=10):
    n = len(lst)
    for i in range(n-1, n-N-1, -1):
        count, word = lst[i]
        print("{}   {}".format(word, count))
            

def get_ideal_zipfs(top10, N=10):
    n = len(top10)
    max_freq = top10[n-1][0]
    counts = []
    for i in range(n-N, n-1):
        rank = n - i
        counts.append(max_freq / rank)
    counts.append(max_freq)
    return counts
    
if __name__ == "__main__":
    file_name = 'nyt.txt'
    # 6a
    most_freq = get_most_freq(file_name, Tokenizer.DEFAULT, 100)
    most_freq = sorted(most_freq)
    print("1.)")
    print_topN(most_freq)
    print()
    
    # 6b
    most_freq = get_most_freq(file_name, Tokenizer.NLTK, 100)
    most_freq = sorted(most_freq)
    print("2.)")
    print_topN(most_freq)
    print()

    # 7a
    counts = list([x[0] for x in most_freq[len(most_freq)-10: len(most_freq)]])
    inv_ranks = []
    for i in range(10):
        rank = 10 - i
        inv_ranks.append(1 / rank)
    
    
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    ax.set_title('Inverse Rank vs. Word Count')
    ax.set_xlabel('Inverse Rank')
    ax.set_ylabel('Word Count')
    ax.scatter(inv_ranks, counts, label="NYT article")
    ax.scatter(inv_ranks, get_ideal_zipfs(most_freq), label="Ideal Zipfs")
    ax.legend()
    # ax.set_yscale('log')


    plt.show()



        
        

    

