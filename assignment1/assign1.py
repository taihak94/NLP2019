import nltk
import re, pprint, collections
from urllib import request
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup

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
    tokenizer = RegexpTokenizer('\w+|\$|\n')
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