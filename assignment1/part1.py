import nltk
import os
import re, pprint, collections
from urllib import request
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
import numpy as np
from os.path import abspath, dirname, join
import inspect
import matplotlib.pyplot as plt
import matplotlib
import math
from math import log
import random
from collections import *
import zipfile, tarfile
import string

""" The following contains all the code used in Part 1 of the assignment, 
without the testing and running code provided in the notebook """

# Part 1.1 
def replace_numbers(tokens):
    return [x if not x.isdigit() else 'N' for x in tokens]

def find_most_common(tokens, top):
    counter = collections.Counter(tokens)
    most_common = counter.most_common(top)
    return [a for a, b in most_common]

def replace_noncommon_tokens(sentences, most_common):
    res = []
    for sentence in sentences:
        res.append([x if (x in most_common) else '<unk>' for x in sentence])
    return res

def ptb_preprocess(filenames, top=10000):
    for single_file in filenames:
        path = nltk.data.find(single_file)
        raw = open(path, 'r').read()
        segments = raw.split("\n")
        sentences = []
        for segment in segments:
            tokens = word_tokenize(segment)
            # remove punctuation
            tokens = [x for x in tokens if x not in string.punctuation]
            # to lowercase
            words = [w.lower() for w in tokens]
            # filter numbers
            sentence = replace_numbers(words)
            
            sentences.append(sentence)
       
        # get most common words and replace all other words with unk
        common_tokens = find_most_common([word for sentence in sentences for word in sentence], top)
        sentences = replace_noncommon_tokens(sentences, common_tokens)
        
        # write out the new data into a file 
        new_filename = single_file + ".out"
        with open(new_filename, 'w') as f:
            for sentence in sentences:
                for word in sentence:
                    f.write("%s " % word)
                f.write("\n")

# part 1.2
class NgramModel(object):
    def __init__(self, n, train, smoothing=False, estimator=None):
        self._n = n
        self.is_unigram_model = (n == 1)
        self.is_smooth = smoothing
        
        cfd = nltk.ConditionalFreqDist((" ".join(train[i : i + n - 1]), "".join(train[i + n - 1])) for i in range(len(train) - n + 1))
        self._probdist = nltk.ConditionalProbDist(cfd, estimator)
        
        # if we are not using smoothing we should implement a backoff model and keep all the seen ngrams
        if not self.is_smooth:
            self._ngramsData = ngrams(train, n)
            self._ngrams = set()
            for ngram in self._ngramsData:
                self._ngrams.add(ngram)
        
        if not self.is_unigram_model:
            if not self.is_smooth:
                self._backoff = NgramModel(n - 1, train, estimator=estimator)
                self._lambda = 1
    
    def prob(self, word, context):
        if (self.is_smooth and self._probdist[context].logprob(word) != 0):
            return self._probdist[context].logprob(word)
        
        # if we are not using smoothing we need to use a different method for avoiding 0 probability 
        elif (tuple(context.split()) + (word, ) in self._ngrams) or (self.is_unigram_model):
            return self._probdist[context].logprob(word)
        else:
            new_context = " ".join(context.split()[1:])
            backoff = self._backoff.prob(word, new_context)
            return self._lambda * backoff
        
    def logprob(self, word, context):
        return - self.prob(word, context)
    
    def get_seed(self):
        return random.choice(self._probdist.conditions())
    
    def generate(self, seed, length):
        out = []
        curr = seed
        end = self._probdist.conditions()[-1]
        i = 0
        while (i <= length and (not curr == end)):
          i += 1
          word = self._probdist[curr].generate()
          curr = " ".join((curr.split())[1:] + [word])
          out.append(word)
        return out

def train_word_lm(dataset, n=2):
    model = NgramModel(n, dataset, estimator=nltk.MLEProbDist)
    return model

# part 1.3
def model_entropy(model, text, n=2):
    H = 0.0
    for i in range(n - 1, len(text)):
        context, word = tuple(text[i - n + 1:i]), text[i]
        context = " ".join(context)
        H += model.logprob(word, context)
    return H / float(len(text) - (n - 1))

def calc_preplexity(model, text, n=2):
    text_entropy = model_entropy(model, text, n)
    return 2 ** (text_entropy)

# part 1.3.1.2
def train_word_lm_lidstone(dataset, n=2, gamma=0.01):
    lidstone_estimator = lambda fd: nltk.LidstoneProbDist(fd, gamma, fd.B() + 1)
    model = NgramModel(n, dataset, estimator=lidstone_estimator)
    return model

# part 1.3.2
def generate(model, seed):
    out = model.generate(seed, 100)
    out = seed + " " + " ".join(out)

    print(out)

# part 1.4
def train_char_lm(fname, order=4):
    data = open(fname, 'r').read()
    lm = defaultdict(Counter)
    pad = "~" * order
    data = pad + data
    for i in range(len(data)-order):
        history, char = data[i:i+order], data[i+order]
        lm[history][char]+=1
    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.items()]
    outlm = {hist:normalize(chars) for hist, chars in lm.items()}
    return outlm

def char_model_entropy(model, text, n=2):
    H = 0.0
    processed_ws = 0 
    for i in range(n - 1, len(text)):
        context, word = tuple(text[i - n:i]), text[i]
        context = "".join(context)
        score = 0
        if(context in model):
            for c,v in model[context]:
                if c == word:
                    score = v
        if(not(score == 0 or score == 0.0)) :
            processed_ws += 1
            H += log(score, 2)
    return - (H / float(len(text) - n))

def calc_preplexity_char(model, text, n=2):
    text_entropy = char_model_entropy(model, text, n)
    return 2 ** (text_entropy)

def generate_letter(lm, history, order):
        history = history[-order:]
        dist = lm[history]
        x = random.random()
        for c,v in dist:
            x = x - v
            if x <= 0: return c
            
def generate_text(lm, order, nletters=1000):
    history = "~" * order
    out = []
    for i in range(nletters):
        c = generate_letter(lm, history, order)
        history = history[-order:] + c
        out.append(c)
    return "".join(out)