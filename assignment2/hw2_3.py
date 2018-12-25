from nltk.corpus import conll2002
import string, nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.feature_selection import RFE

train_sents = list(conll2002.iob_sents('esp.train'))
test_sents = list(conll2002.iob_sents('esp.testa'))
d_train_sents = list(conll2002.iob_sents('ned.train'))
d_test_sents = list(conll2002.iob_sents('ned.testa'))
v = DictVectorizer(sparse=True)
    
def hasNumbers(str):
    return any(c.isdigit() for c in str)

def get_word_features (word):
    w = word[0]
    features = {
     "form": w,
     "pos": word[1],
     "is_number": w.isdigit(),
     "contains_number": hasNumbers(w),
     "beginCapital": w[0].isupper(),
     "allCaps": w.isupper(),
     "isPunc": w in string.punctuation,
     "firstLetter": w[0],
     "first2Letters": w[0:2],
     "first3Letters": w[0:3],
     "lastLetter": w[-1],
     "last2Letters": w[-2:],
     "last3Letters": w[-3:]
    }
    return features

def get_corpus_features (corpus):
#gets a corpus, returns a list of features for every word
    X=[]
    for sent in corpus:
        X+=[get_word_features(w) for w in sent]
    return X

def get_y (corpus):
    y=[]
    for sent in corpus:
        y+=[w[2] for w in sent]
    return y
    
def train(train_sents, v, features):
    y = get_y (train_sents)
    X = v.fit_transform(features)
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
    return clf

def nltk_classify(train_sents):
    train_set=[]
    for sent in train_sents:
        tuple = [(get_word_features(w),w[2]) for w in sent]
        train_set+=tuple
    return nltk.NaiveBayesClassifier.train(train_set)

def accuracy (x,y):
    correct = sum([1 if x[i]==y[i] else 0 for i in range(len(x))])
    return correct / len(x)

def print_errors (x,y,test_sents):
    features = get_corpus_features(test_sents)
    errors=[]
    for i in range(len(x)):
        if x[i]!=y[i]:
            errors.append((y[i], x[i], features[i].get("form")))
    for (tag, guess, name) in sorted(errors):
        print('correct=%-8s guess=%-8s word=%-30s' % (tag, guess, name))

def evaluate (y_predict, y_true):
    print ("accuracy:", accuracy (y_predict,y_true))
    print ("confusion matrix: ")
    print (set(y_true))
    print (metrics.confusion_matrix(y_true, y_predict))
    print(metrics.classification_report(y_true, y_predict))
    
def predict (clf, v, test_features):
    X2 = v.transform(test_features)
    return clf.predict(X2)

def run():
    features = get_corpus_features(train_sents)
    clf = train(train_sents, v, features)
    test_features = get_corpus_features(test_sents)
    y_predict = predict (clf, v, test_features)
    y_true = get_y (test_sents)
    evaluate (y_predict, y_true)
  #  print_errors (y_predict,y_true,test_sents)
  
def dutch_run():
    d_features = get_corpus_features(d_train_sents)
    d_v = DictVectorizer(sparse=True)
    d_clf = train(d_train_sents, d_v, d_features)
    d_test_features = get_corpus_features(d_test_sents)
    d_y_predict = predict (d_clf, d_v, d_test_features)
    d_y_true = get_y (d_test_sents)
    print ("accuracy:", accuracy (d_y_predict,d_y_true))
    print (metrics.classification_report(d_y_true, d_y_predict))
    
#PART 2: use features from previous and following word
  
def get_word_features2 (word, prev, next):
#also includes information about next and previous word
    w = word[0]
    p = prev[0]
    n = next[0]
    features = {
     "form": w,
     "pos": word[1],
     "is_number": w.isdigit(),
     "contains_number": hasNumbers(w),
     "beginCapital": w[0].isupper(),
     "allCaps": w.isupper(),
     "isPunc": w in string.punctuation,
     "firstLetter": w[0],
     "first2Letters": w[0:2],
     "first3Letters": w[0:3],
     "lastLetter": w[-1],
     "last2Letters": w[-2:],
     "last3Letters": w[-3:],
     "p_form": p,
     "p_pos": prev[1],
     "p_is_number": p.isdigit(),
     "p_contains_number": hasNumbers(p),
     "p_beginCapital": p[0].isupper(),
     "p_allCaps": p.isupper(),
     "p_isPunc": p in string.punctuation,
     "p_firstLetter": p[0],
     "p_first2Letters": p[0:2],
     "p_first3Letters": p[0:3],
     "p_lastLetter": p[-1],
     "p_last2Letters": p[-2:],
     "p_last3Letters": p[-3:],
     "n_form": n,
     "n_pos": next[1],
     "n_is_number": n.isdigit(),
     "n_contains_number": hasNumbers(n),
     "n_beginCapital": n[0].isupper(),
     "n_allCaps": n.isupper(),
     "n_ispunc": n in string.punctuation,
     "n_firstLetter": n[0],
     "n_first2Letters": n[0:2],
     "n_first3Letters": n[0:3],
     "n_lastLetter": n[-1],
     "n_last2Letters": n[-2:],
     "n_last3Letters": n[-3:]
    }
    return features

def get_corpus_features2 (corpus):
#gets a corpus, returns a list of features for every word
    flat = [w for sent in corpus for w in sent]
    pad = [("*","*","*")]
    flat = pad + flat + pad
    X=[]
    for i in range(1, len(flat)-1):
        X.append(get_word_features2(flat[i],flat[i-1],flat[i+1]))
    return X


def run2():
    features = get_corpus_features2(train_sents)
    clf2 = train(train_sents, v, features)
    test_features = get_corpus_features2(test_sents)
    y_predict2 = predict (clf2, v, test_features)
    y_true2 = get_y (test_sents)
    evaluate (y_predict2, y_true2)
    
def rfe(clf, ):
    rfe = RFE(clf, 3)
#    rfe = rfe.fit(dataset.data, dataset.target)
    # summarize the selection of the attributes
    print(rfe.support_)
    print(rfe.ranking_)
    
def find_illegal_sequences (guess):
    OIX, IXIY, BXIY = 0,0,0
    for i in range(len(guess)-1):
        curr, next = guess[i], guess[i+1]
        if curr[0]=="O" and next[0]=="I":
            OIX+=1
        elif curr[0]=="I" and next[0]=="I" and curr[1:] != next[1:]:
            IXIY+=1
        elif curr[0]=="B" and next[0]=="I" and curr[1:] != next[1:]:
            BXIY+=1
    return {"O-IX": OIX, "IX-IY": IXIY, "BX-IY": BXIY}
