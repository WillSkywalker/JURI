import math
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk import word_tokenize


def load_data(filename, character=True):
    doc = []
    with open(filename) as f:
        sent = []
        for line in f:
            if line.startswith('-DOCSTART-'):
                continue
            elif (not line) or line[0] == '\n':
                if sent:
                    doc.append(sent)
                    sent = []
            else:
                elems = line.split()
                if character:
                    sent.append((elems[0], elems[-1], tuple(elems[0])))
                else:
                    sent.append((elems[0], elems[-1]))
        if sent:
            doc.append(sent)

    return doc


def load_embedding(filename, word_set=False):
    embeddings = {}
    with open(filename, encoding="utf-8") as f_embeddings:
        for line in f_embeddings:
            paras = line.split()
            word = paras[0]
            if word_set and word not in word_set:
                continue
            weight = [float(num) for num in paras[1:]]
            embeddings[word] = weight
    return embeddings


def create_batch(x, y, embeddings, charmap={}, character=False):
    if character:
        data = [[(embeddings[word.lower()] if word in embeddings else embeddings['UNKNOWN_TOKEN'],
                tuple(word), label) for word in word_tokenize(art)] for art, label in zip(x, y)]
    else:
        data = [([embeddings[word.lower()] if word in embeddings else embeddings['UNKNOWN_TOKEN']
                for word in word_tokenize(art)], label) for art, label in zip(x, y)]
    lengths = {}
    for elem in data:
        length = int(math.ceil(len(elem[0]) / 50.0)) * 50
        text = elem[0][-length:]
        lengths.setdefault(length, []).append((text, elem[-1]))
    batches = []
    if character:
        for batch in lengths.values():
            arts = []
            chars = []
            labels = []
            for art in batch:
                arts.append(art)
                chars.append(pad_sequences([[charmap[c] for c in elem[2]] for elem in sent], 20))
                labels.append([elem[1] for elem in sent])

            batches.append([np.asarray(words), np.expand_dims(np.asarray(labels), -1), np.asarray(chars)])
    else:
        for batch in lengths.values():
            arts = []
            labels = []
            for elem in batch:
                arts.append(elem[0])
                labels.append(elem[-1])
            batches.append([np.array(arts), np.array(labels)])
    return batches
