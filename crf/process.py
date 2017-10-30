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
    f_cs = [f for f in listdir('CS') if os.path.isfile(os.path.join('CS',f))]
    files_cs = [ 'CS/'+f for f in f_cs if '.pkl' in f]
    cs_docs = concat_pkl(files_cs)

    cs_data = []
    for i, doc in enumerate(cs_docs):
        tokens = [t for t, label in doc]
        tagged = nltk.pos_tag(tokens)
        cs_data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

    cs = [extract_features(doc) for doc in cs_data]

    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')
    cs_pred = [tagger.tag(xseq) for xseq in cs]

    # take a look at a random sample in testing set
    for i in range(len(cs)):
        for x, y in zip(cs_pred[i], [x[1].split("=")[1] for x in cs[i]]):
            if x in "ST":
                with open("files_cs/cs"+str(i)+".txt", "a") as text_file:
                    text_file.write(y+'\n')

    ### preprocessing data
    f_ba = [f for f in listdir('BA') if os.path.isfile(os.path.join('BA',f))]
    files_ba = [ 'BA/'+f for f in f_ba if '.pkl' in f]
    ba_docs = concat_pkl(files_ba)

    ba_data = []
    for i, doc in enumerate(ba_docs):
        tokens = [t for t, label in doc]
        tagged = nltk.pos_tag(tokens)
        ba_data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

    ba = [extract_features(doc) for doc in ba_data]

    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')
    ba_pred = [tagger.tag(xseq) for xseq in ba]

    # take a look at a random sample in testing set
    for i in range(len(ba)):
        for x, y in zip(ba_pred[i], [x[1].split("=")[1] for x in ba[i]]):
            if x in "ST":
                with open("files_ba/ba"+str(i)+".txt", "a") as text_file:
                    text_file.write(y+'\n')
