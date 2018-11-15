import nltk
import re, pprint, collections
from urllib import request
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
# import nltk.model as model
import numpy as np
# from nltk.corpus import treebank

def is_special_char(x):
  return ((x=='N') or ('\n' in x))

def replace_numbers(tokens):
  return [x if not x.isdigit() else 'N' for x in tokens]

def find_most_common(tokens, top):
  r = re.compile("^[a-zA-Z]+$")
  word_tokens = list(filter(r.match, tokens))
  counter = collections.Counter(word_tokens)
  most_common = counter.most_common(top)
  return [a for a, b in most_common]

def replace_noncommon_tokens(tokens, most_common):
  return [x if ((x in most_common) or (is_special_char(x))) else '<unk>' for x in tokens]

def ptb_preprocess(filenames, top=10000):
  for single_file in filenames:
    path = nltk.data.find(single_file)
    raw = open(path, 'rU').read()
    
    # clean punctuations
    tokenizer = RegexpTokenizer('\w+|\$|\#|\@|\%|\&|\*|\^|\~|\<|\>|\=|\+|\n')
    tokens = tokenizer.tokenize(raw)

    # to lowercase
    words = [w.lower() for w in tokens]

    # filter numbers
    words_no_numbers = replace_numbers(words)

    # get most common words and replace all other words with unk
    common_tokens = find_most_common(words, top)
    words = replace_noncommon_tokens(words_no_numbers, common_tokens)

    # write out the new data into a file 
    new_filename = single_file + ".out"
    with open(new_filename, 'w') as f:
      for word in words:
        if ('\n' in word):
          f.write("\n")
        else:
          f.write("%s " % word)
    
#example  
ptb_preprocess(["/home/user/Workspace/NLP2019/assignment1/tmpFile.txt"], 10000)

def train_word_lm(dataset, n=2):
  pad = ["<unk>" for i in range(n)]
  padded_data = pad + dataset 
  cfd = nltk.ConditionalFreqDist((" ".join(padded_data[i : i + n]), "".join(padded_data[i + n])) for i in range(len(padded_data) - n))
  cpd = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
  return cpd


def train_word_with_nltk(dataset, n=2):
  data = " ".join(dataset)
  # print(model)
  n_gram = ngrams()
  n_grams = model.MLENgramModel(n, data)
  print(n_grams)


# yet another example
path = nltk.data.find("/home/user/tmp/ptb.train.txt")
raw = open(path, 'rU').read()

# clean punctuations
tokenizer = RegexpTokenizer('\w+|\$|\#|\@|\%|\&|\*|\^|\~|\<|\>|\=|\+')
tokens = tokenizer.tokenize(raw)

# to lowercase
words = [w.lower() for w in tokens]
lm = train_word_lm(words, 4)
# out = []
# seed = ['<unk>'] * 4
# curr = ' '.join(seed)
# end = " ".join(words[len(words)-4:])
# while (not curr == end):
#   word = lm[curr].generate()
#   curr = " ".join((curr.split())[1:] + [word])
#   out.append(word)

# print(" ".join(out))

def model_entropy(model, text, n=2):
  n_grams = ngrams(text, n + 1)
  H = 0.0
  processed_ws = 0 
  for gram in n_grams:
    context, word = tuple(gram[:-1]), gram[-1]
    context = " ".join(context)
    processed_ws += 1
    if(not(model[context].prob(word) == 0)):
      H += (np.log(model[context].prob(word)))
  return - (H / processed_ws)

def calc_preplexity(model, text, n=2):
  text_entropy = model_entropy(model, text, n)
  return 2 ** (text_entropy)

path = nltk.data.find("/home/user/tmp/ptb.test.txt")
raw = open(path, 'rU').read()

# clean punctuations
tokenizer = RegexpTokenizer('\w+|\$|\#|\@|\%|\&|\*|\^|\~|\<|\>|\=|\+')
tokens = tokenizer.tokenize(raw)

# to lowercase
text = [w.lower() for w in tokens]

print("prep", calc_preplexity(lm, text, 4))