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

class NgramModel(object):
    def __init__(self, n, train, smoothing=False, estimator=None):
        self._n = n
        self.is_unigram_model = (n == 0)
        self.is_smooth = smoothing
        cfd = nltk.ConditionalFreqDist((" ".join(train[i : i + n]), "".join(train[i + n])) for i in range(len(train) - n))
        self._probdist = nltk.ConditionalProbDist(cfd, estimator)
        
        # if we are not using smoothing we should implement a backoff model and keep all the seen ngrams
        if not self.is_smooth:
            self._ngramsData = ngrams(train, n - 1)
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
        elif (tuple(context.split()) + (word, ) in self._ngrams):
            return self._probdist[context].logprob(word)
        elif self.is_unigram_model:
          return self._probdist[context].logprob("")
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

ptb_train_path = abspath(join(dirname("__file__"), "ptb.train.txt"))
# ptb_train_path = abspath(join(dirname("__file__"), "dataset.txt"))
ptb_test_path = abspath(join(dirname("__file__"), "ptb.test.txt"))
# ptb_test_path = abspath(join(dirname("__file__"), "datatest.txt"))
with open(ptb_train_path) as f:
    ptb_train = f.read()
with open(ptb_test_path) as f:
    ptb_test = f.read()

def model_entropy(model, text, n=2):
    H = 0.0
    for i in range(n, len(text)):
        context, word = tuple(text[i - n:i]), text[i]
        context = " ".join(context)
        H += model.logprob(word, context)
    return H / float(len(text) - (n - 1))

def calc_preplexity(model, text, n=2):
    text_entropy = model_entropy(model, text, n)
    return 2 ** (text_entropy)

ptb_train_tokenized = ptb_train.split()
ptb_test_tokenized = ptb_test.split()

n = 3
# lm_MLE = train_word_lm(ptb_train_tokenized, n)

# print(calc_preplexity(lm_MLE, ptb_test_tokenized, n))

def train_word_lm_lidstone(dataset, n=2, gamma=0.01):
    lidstone_estimator = lambda fd: nltk.LidstoneProbDist(fd, gamma, fd.B() + 100)
    model = NgramModel(n, dataset, smoothing=True, estimator=lidstone_estimator)
    return model

gammas = np.linspace(0.01, 1, 20)
n = 3
perplexities_l = list(range(20))
i = 0
for gamma in gammas:
    lm_LIDSTONE = train_word_lm_lidstone(ptb_train_tokenized, n, gamma)
    perplexities_l[i] = calc_preplexity(lm_LIDSTONE, ptb_test_tokenized, n)
    print(perplexities_l[i])
    i += 1