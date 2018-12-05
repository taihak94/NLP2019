#############################################################
## ASSIGNMENT 2_1 CODE SKELETON
#############################################################

from collections import defaultdict
import gzip
import numpy as np
import matplotlib.pyplot as plt
#### Q1.1 Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    positive_hits_array = np.array([(1 if(y_pred[i] == y_true[i] == 1) else 0) for i in range(len(y_pred))])
    positive_hits = np.count_nonzero(positive_hits_array == 1)
    total_positive = np.count_nonzero(np.array(y_pred) == 1)
    precision = float(positive_hits) / float(total_positive)
    return precision
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    positive_hits_array = np.array([(1 if(y_pred[i] == y_true[i] == 1) else 0) for i in range(len(y_pred))])
    positive_hits = np.count_nonzero(positive_hits_array == 1)
    total_positive = np.count_nonzero(np.array(y_true) == 1)
    recall = float(positive_hits) / float(total_positive)

    return recall

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = 2 * float(precision * recall) / float(precision + recall)

    return fscore

#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

### 1.2.1: A very simple baseline

## Labels every word complex
def all_complex(data_file):
    words, actual_labels = load_file(data_file)
    all_complex_labels = np.ones((len(words),), dtype=int)
    precision = get_precision(all_complex_labels, actual_labels)
    recall = get_recall(all_complex_labels, actual_labels)
    fscore = get_fscore(all_complex_labels, actual_labels)
    performance = [precision, recall, fscore]
    return performance


### 1.2.2: Word length thresholding

def word_length_baseline(data_file, threshold):
    words, actual_labels = load_file(data_file)
    threshold_labels = [(1 if(len(word) >= threshold) else 0) for word in words]
    
    precision = get_precision(threshold_labels, actual_labels)
    recall = get_recall(threshold_labels, actual_labels)
    fscore = get_fscore(threshold_labels, actual_labels)
    preformance = [precision, recall, fscore]
    return preformance

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    best_tfscore = 0.0
    best_i = 1
    i = 1
    while(True):
        tprecision, trecall, tfscore = word_length_baseline(training_file, i)
        if(tfscore < best_tfscore):
            break
        else:
            best_i = i
            i += 1
            best_tfscore = tfscore
            
    tprecision, trecall, tfscore = word_length_baseline(training_file, best_i)
    dprecision, drecall, dfscore = word_length_baseline(development_file, best_i)
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 1.2.3: Word frequency thresholding
def word_frequency_baseline(data_file, counts, threshold):
    words, actual_labels = load_file(data_file)
    threshold_labels = [(1 if(counts[word] >= threshold) else 0) for word in words]
    
    precision = get_precision(threshold_labels, actual_labels)
    recall = get_recall(threshold_labels, actual_labels)
    fscore = get_fscore(threshold_labels, actual_labels)
    preformance = [precision, recall, fscore]
    return preformance

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt') as f: 
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set
def word_frequency_threshold(training_file, development_file, counts):
    best_tfscore = 0.0
    i = 1
    threshold = np.random.randint(10, 2000000, size=1, dtype=int)
    while(True):
        tprecision, trecall, tfscore = word_frequency_baseline(training_file, counts, threshold)
        if(tfscore < best_tfscore):
            break
        else:
            threshold = np.random.randint(10, 2000000, size=1, dtype=int)
            best_tfscore = tfscore
            
    tprecision, trecall, tfscore = word_frequency_baseline(training_file, counts, threshold)
    dprecision, drecall, dfscore = word_frequency_baseline(development_file, counts, threshold)
    
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 1.3.1: Naive Bayes
        
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    #import training dataset
    t_words, t_labels = load_file(training_file)
    t_features = {}
    # get length features
    t_features["length"] = np.array([len(word) for word in t_words])
    # get frequency features
    t_features["frequency"] = np.array([counts[word] for word in t_words])
    # build features array
    X_t_original = np.array([t_features["length"], t_features["frequency"]]).T
    
    # normalize features
    t_mean = np.mean(X_t_original)
    t_sd = np.std(X_t_original)
    X_t_scaled = (X_t_original - t_mean)/t_sd
    
    # train the classifier
    clf = GaussianNB()
    clf.fit(X_t_scaled, t_labels)
    
    # extract features for development file
    d_words, d_labels = load_file(development_file)
    d_features = {}
    # get length features
    d_features["length"] = np.array([len(word) for word in d_words])
    # get frequency features
    d_features["frequency"] = np.array([counts[word] for word in d_words])
    # build features array
    X_d_original = np.array([d_features["length"], d_features["frequency"]]).T
    
    # normalize development features - note how we use the training mean and sd
    X_d_scaled = (X_d_original - t_mean)/t_sd
    
    dev_pred = clf.predict(X_d_scaled)
    train_pred = clf.predict(X_t_scaled)
    
    tprecision = get_precision(train_pred, t_labels)
    trecall = get_recall(train_pred, t_labels)
    tfscore = get_fscore(train_pred, t_labels)
    
    dprecision = get_precision(dev_pred, d_labels)
    drecall = get_recall(dev_pred, d_labels)
    dfscore = get_fscore(dev_pred, d_labels)
    
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

if __name__ == "__main__":
    training_file = "../data/complex_words_training.txt"
    development_file = "../data/complex_words_development.txt"
    test_file = "../data/complex_words_test_unlabeled.txt"
    train_data = load_file(training_file)
    
