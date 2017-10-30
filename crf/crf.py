'''
Reference: http://www.albertauyeung.com/post/python-sequence-labelling-with-crf/
'''
import pycrfsuite
import nltk
import pickle
import os.path
from os import listdir
from sys import argv
import numpy as np
from sklearn.metrics import classification_report

def concat_pkl(files):
    docs = []
    for ff in files:
        with open(ff,'rb') as f:
            doc = pickle.load(f)
        docs.append(doc)
    return docs


def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]
    return features


def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]


def get_labels(doc):
        return [label for (token, postag, label) in doc]



if __name__ == '__main__':
    
    ### preprocessing data
    f_tr = [f for f in listdir('training') if os.path.isfile(os.path.join('training',f))]
    files_tr = [ 'training/'+f for f in f_tr if '.pkl' in f]
    train_docs = concat_pkl(files_tr)

    train_data = []
    for i, doc in enumerate(train_docs):
        tokens = [t for t, label in doc]
        tagged = nltk.pos_tag(tokens)
        train_data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
    
    f_te = [f for f in listdir('testing') if os.path.isfile(os.path.join('testing',f))]
    files_te = [ 'testing/'+f for f in f_te if '.pkl' in f]
    test_docs = concat_pkl(files_te)
    
    test_data = []
    for i, doc in enumerate(test_docs):
        tokens = [t for t, label in doc]
        tagged = nltk.pos_tag(tokens)
        test_data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

    X_train = [extract_features(doc) for doc in train_data]
    X_test = [extract_features(doc) for doc in test_data]

    y_train = [get_labels(doc) for doc in train_data]
    y_test = [get_labels(doc) for doc in test_data]

    ### training
    # submit training data to the trainer
    trainer = pycrfsuite.Trainer(verbose=True)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    # set parameters
    trainer.set_params({
        # coefficient for L1 penalty
        'c1': 0.1,
        # coefficient for L2 penalty
        'c2': 0.01,
        # max numbr of iterations
        'max_iterations': 200,
        # include possible but not observed trainsitions
        'feature.possible_transitions': True
    }) 

    trainer.train('crf.model')

    ### checking results
    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')
    y_pred = [tagger.tag(xseq) for xseq in X_test]

    # take a look at a random sample in testing set
    i = 5
    for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
        print("%s (%s)" % (y, x))


    # numerical mapping of labels
    labels = {"D":0, "C":1, "T": 2, "S": 3, "I": 4, "A":5, "":6}

    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])

    print(classification_report( truths, predictions, target_names=["D","C","T","S","I"]))
