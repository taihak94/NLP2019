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
from math import log
import random
from collections import *
import zipfile
import string

# def is_special_char(x):
#   return ((x=='N') or ('\n' in x))

# def replace_numbers(tokens):
#   return [x if not x.isdigit() else 'N' for x in tokens]

# def find_most_common(tokens, top):
#   r = re.compile("^[a-zA-Z]+$")
#   word_tokens = list(filter(r.match, tokens))
#   counter = collections.Counter(word_tokens)
#   most_common = counter.most_common(top)
#   return [a for a, b in most_common]

# def replace_noncommon_tokens(tokens, most_common):
#   return [x if ((x in most_common) or (is_special_char(x))) else '<unk>' for x in tokens]

# def ptb_preprocess(filenames, top=10000):
#   for single_file in filenames:
#     path = nltk.data.find(single_file)
#     raw = open(path, 'rU').read()
    
#     # clean punctuations
#     tokenizer = RegexpTokenizer('\w+|\$|\#|\@|\%|\&|\*|\^|\~|\<|\>|\=|\+|\n')
#     tokens = tokenizer.tokenize(raw)

#     # to lowercase
#     words = [w.lower() for w in tokens]

#     # filter numbers
#     words_no_numbers = replace_numbers(words)

#     # get most common words and replace all other words with unk
#     common_tokens = find_most_common(words, top)
#     words = replace_noncommon_tokens(words_no_numbers, common_tokens)

#     # write out the new data into a file 
#     new_filename = single_file + ".out"
#     with open(new_filename, 'w') as f:
#       for word in words:
#         if ('\n' in word):
#           f.write("\n")
#         else:
#           f.write("%s " % word)
    
# #example  
# ptb_preprocess(["/home/user/Workspace/NLP2019/assignment1/tmpFile.txt"], 10000)

# def train_word_lm(dataset, n=2):
#   pad = ["<unk>" for i in range(n)]
#   padded_data = pad + dataset 
#   cfd = nltk.ConditionalFreqDist((" ".join(padded_data[i : i + n]), "".join(padded_data[i + n])) for i in range(len(padded_data) - n))
#   cpd = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
#   return cpd


# def train_word_with_nltk(dataset, n=2):
#   data = " ".join(dataset)
#   # print(model)
#   n_gram = ngrams()
#   n_grams = model.MLENgramModel(n, data)
#   print(n_grams)


# # yet another example
# path = nltk.data.find("/home/user/tmp/ptb.train.txt")
# raw = open(path, 'rU').read()

# # clean punctuations
# tokenizer = RegexpTokenizer('\w+|\$|\#|\@|\%|\&|\*|\^|\~|\<|\>|\=|\+')
# tokens = tokenizer.tokenize(raw)

# # to lowercase
# words = [w.lower() for w in tokens]
# lm = train_word_lm(words, 4)
# # out = []
# # seed = ['<unk>'] * 4
# # curr = ' '.join(seed)
# # end = " ".join(words[len(words)-4:])
# # while (not curr == end):
# #   word = lm[curr].generate()
# #   curr = " ".join((curr.split())[1:] + [word])
# #   out.append(word)

# # print(" ".join(out))

# def model_entropy(model, text, n=2):
#   n_grams = ngrams(text, n + 1)
#   H = 0.0
#   processed_ws = 0 
#   for gram in n_grams:
#     context, word = tuple(gram[:-1]), gram[-1]
#     context = " ".join(context)
#     processed_ws += 1
#     if(not(model[context].prob(word) == 0)):
#       H += (np.log(model[context].prob(word)))
#   return - (H / processed_ws)

# def calc_preplexity(model, text, n=2):
#   text_entropy = model_entropy(model, text, n)
#   return 2 ** (text_entropy)

# path = nltk.data.find("/home/user/tmp/ptb.test.txt")
# raw = open(path, 'rU').read()

# # clean punctuations
# tokenizer = RegexpTokenizer('\w+|\$|\#|\@|\%|\&|\*|\^|\~|\<|\>|\=|\+')
# tokens = tokenizer.tokenize(raw)

# # to lowercase
# text = [w.lower() for w in tokens]

# print("prep", calc_preplexity(lm, text, 4))

# def train_word_lm(dataset, n=2):
#     vocab = sorted(set(dataset))
#     cfd = nltk.ConditionalFreqDist((" ".join(dataset[i : i + n - 1]), "".join(dataset[i + n - 1])) for i in range(len(dataset) - n + 1 ))
#     for key in cfd.conditions():
#       for j in range(len(vocab)):
#         if not vocab[j] in cfd[key]:
#           cfd[key][vocab[j]] = 1
#     # print(cfd.conditions())
#     for condition in cfd.conditions():
#       # print(condition)
#       for j in range(len(vocab)):
#         print(condition, vocab[j], cfd[condition].freq(vocab[j]))
#         # print(cfd[condition].freq(vocab[j]))
#     cpd = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
#     return cpd

class NgramModel(object):
    def __init__(self, n, train, estimator=None):
        self._n = n
        self.is_unigram_model = (n == 1)
        cfd = nltk.ConditionalFreqDist((" ".join(train[i : i + n - 1]), "".join(train[i + n - 1])) for i in range(len(train) - n + 1))
        self._probdist = nltk.ConditionalProbDist(cfd, estimator)
        self._ngramsData = ngrams(train, n)
        self._ngrams = set()
        for ngram in self._ngramsData:
            print(ngram)
            self._ngrams.add(ngram)
        
        if not self.is_unigram_model:
            self._backoff = NgramModel(n - 1, train, estimator)
            self._lambda = 1
    
    def prob(self, word, context):
        print("c:", context)
        print("w:", word)
        print("tuple", tuple(context.split()) + (word, ))
        print(tuple(context.split()) + (word, ) in self._ngrams)
        if (tuple(context.split()) + (word, ) in self._ngrams) or (self.is_unigram_model):
            print("probdist", self._probdist[context].prob(word))
            return self._probdist[context].prob(word)
        else:
            new_context = " ".join(context.split()[1:])
            backoff = self._backoff.prob(word, new_context)
            print("backoff",backoff)

            return self._lambda * backoff
        
    def logprob(self, word, context):
        return -(log(self.prob(word, context), 2))
    
    def entropy(self, text):
        H = 0.0
        for i in range(self._n - 1, len(text)):
            context, word = tuple(text[i - self._n + 1:i]), text[i]
            context = " ".join(context)
            print("cont", context)
            print("word", word)
            H += self.logprob(word, context)
        return H / float(len(text) - (self._n - 1))
    
    def get_model(self):
        return self._probdist

def train_word_lm(dataset, n=2):
#     vocab = sorted(set(dataset))
#     cfd = nltk.ConditionalFreqDist((" ".join(dataset[i : i + n - 1]), "".join(dataset[i + n - 1])) for i in range(len(dataset) - n + 1))
#     for key in cfd.conditions():
#         for j in range(len(vocab)):
#             if not vocab[j] in cfd[key]:
#                 cfd[key][vocab[j]] = 1
#     cpd = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
    model = NgramModel(n, dataset, estimator=nltk.MLEProbDist)
    return model

# def train_word_lm(dataset, n=2):
#     lm = defaultdict(Counter)
#     pad = ["~" for i in range(n)]
#     dataset = pad + dataset
#     for i in range(len(dataset)-n):
#         history, char = dataset[i:i+n], dataset[i+n]
#         history = " ".join(history)
#         lm[history][char]+=1
#     def normalize(counter):
#         s = float(sum(counter.values()))
#         return [(c,cnt/s) for c,cnt in counter.items()]
#     outlm = {hist:normalize(chars) for hist, chars in lm.items()}
#     return outlm


# def model_entropy(model, text, n=2):
#     H = 0.0
#     processed_ws = len(text)
#     print(processed_ws)
#     for i in range(n - 1, len(text)):
#         context, word = tuple(text[i - n + 1:i]), text[i]
#         context = " ".join(context)
#         # processed_ws += 1
#         print(context)
#         score = 0
#         if(context in model):
#             score = model[context].prob(word)
#         print(score)
#         if(not(score == 0 or score == 0.0)) :
#             H += np.log2(score)
#     print(H)
#     return - (H / float(processed_ws - n))

def model_entropy(model, text, n=2):
    H = 0.0
    for i in range(n - 1, len(text)):
        context, word = tuple(text[i - n + 1:i]), text[i]
        context = " ".join(context)
        print("cont", context)
        print("word", word)
        H += model.logprob(word, context)
    return H / float(len(text) - (n - 1))

def calc_preplexity(model, text, n=2):
    text_entropy = model_entropy(model, text, n)
    print("Ent", text_entropy)
    return 2 ** (text_entropy)

# ptb_train_path = abspath(join(dirname("__file__"), "ptb.train.txt"))
ptb_train_path = abspath(join(dirname("__file__"), "dataset.txt"))
ptb_test_path = abspath(join(dirname("__file__"), "dataTest.txt"))
# ptb_test_path = abspath(join(dirname("__file__"), "ptb.test.txt"))
# tokenizer = RegexpTokenizer('\w+|\$|\#|\@|\%|\&|\*|\^|\~|\=|\+|\<unk>')
ptb_train_tokenized = (open(ptb_train_path, 'r').read()).split()

# Train the ngram model withn = 4
n = 4
lm_MLE = train_word_lm(ptb_train_tokenized, n)

ptb_test_tokenized = (open(ptb_test_path, 'r').read()).split()
print(calc_preplexity(lm_MLE, ptb_test_tokenized, n))