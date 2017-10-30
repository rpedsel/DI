import pickle
import os.path
from sys import argv
from os import listdir

def label_file(myfile):
    with open(myfile,'r') as f:
        data = f.read()

    doc = []
    print(data)
    for word in data.split():
        lbl = "?"
        doc.append((word,lbl))
    with open(myfile.split('.')[0]+".pkl", "wb") as f:
        pickle.dump(doc, f)


if __name__ == '__main__':

    fs = [f for f in listdir(argv[1]) if os.path.isfile(os.path.join(argv[1],f))]
    files = [ argv[1]+'/'+f for f in fs if '.txt' in f]
    for f in files:
        print(f)
        label_file(f)
