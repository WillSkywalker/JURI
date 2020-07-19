import os
import json
import datetime
import random
import logging
import time
import numpy as np
from collections import Counter, defaultdict
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists

from config.config import Config
from masterthesis.db import Decisions, Judgments, Prediction, Model
from masterthesis.base import BaseDecisionModel, BaseModel
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from gensim.models import Word2Vec, KeyedVectors, translation_matrix
from gensim.test.utils import common_texts, get_tmpfile
# from masterthesis.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError
from gensim.models.translation_matrix import Space

from nltk.tokenize import word_tokenize

MODEL_NAME = 'W2V V1'
AUTHOR = 'Xu Xiao'
DESCRIPTION = 'just svm lol'
DATE = datetime.datetime.today()

DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
# Session = sessionmaker(bind=engine)
# session = Session()


class TfidfEmbeddingVectorizer:
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = word2vec.vector_size
        self.max_idf = 0
        print('DIM: ', str(self.dim))

    @staticmethod
    def dummy(x):
        return x

    def maxidf(self):
        return self.max_idf

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=self.dummy)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        self.max_idf = max(tfidf.idf_)

        self.word2weight = defaultdict(
            self.maxidf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] * self.word2weight[w]
                        for w in word_tokenize(words) if w in self.word2vec] or
                        # for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0) for words in X])


class MergedTfidfEmbeddingVectorizer:
    def __init__(self, wv_fr, wv_en, wordlist):
        self.wv_fr = wv_fr
        self.wv_en = wv_en
        self.word2vec = {}
        self.word2weight = None
        self.max_idf = 0
        self.dim = wv_fr.vector_size
        print('DIM: ', str(self.dim))

        transmat = translation_matrix.TranslationMatrix(wv_fr, wv_en, wordlist)
        source_space = Space.build(transmat.source_lang_vec, wv_fr.vocab.keys())
        source_space.normalize()
        mapped_source_space = transmat.apply_transmat(source_space)
        for word, idx in mapped_source_space.word2index.items():
            self.word2vec[word] = mapped_source_space.mat[idx]
        for word, vec in self.wv_en.vocab.items():
            if word not in self.word2vec:
                self.word2vec[word] = self.wv_en.word_vec(word)
            # else:
            #     self.word2vec[word] = (self.word2vec[word] + self.wv_en.word_vec(word)) / 2.0

    @staticmethod
    def dummy(x):
        return x

    def maxidf(self):
        return self.max_idf

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=self.dummy)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        self.max_idf = max(tfidf.idf_)

        self.word2weight = defaultdict(
            self.maxidf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] * self.word2weight[w]
                        for w in word_tokenize(words) if w in self.word2vec] or
                        # for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0) for words in X])


class W2VModel(BaseModel):
    """naive bayes"""
    def __init__(self, embedding, name=MODEL_NAME, author=AUTHOR, description=DESCRIPTION, date=DATE):
        super(W2VModel, self).__init__(name, author, description, date)
        wv = KeyedVectors.load_word2vec_format(embedding)
        self.clf = Pipeline([
            ('vect', TfidfEmbeddingVectorizer(wv.wv)),
            ('clf', LinearSVC()),
        ])

    @staticmethod
    def conclusion_simple(desc):
        # Mark the state of conclusion. 0 for pass and 1 for fail, adding more situation possible
        if not desc:
            return 1
        if 'Violation of Article ' in desc or 'Violation of Art. ' in desc or 'Violations of Art. ' in desc:
            return 0
        else:
            return 1

    conclusion = conclusion_simple

    @staticmethod
    def conclusion_fr(desc):
        # Mark the state of conclusion. 0 for pass and 1 for fail, adding more situation possible
        if not desc:
            return 1
        if "Violation de" in desc:
            return 0
        else:
            return 1

    @staticmethod
    def extract_input(decision_texts):
        pass

    def train(self, x, y):

        # results = [session.query(Judgments).filter_by(appno=a).with_entities(Judgments.conclusion).first() for a in new_appnos]
        assert len(x) == len(y)
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)


class CombinedW2VModel(BaseModel):
    """naive bayes"""
    def __init__(self, embedding_fr, embedding_en, word_pairs='french.txt',
                 name=MODEL_NAME, author=AUTHOR, description=DESCRIPTION, date=DATE):
        super(CombinedW2VModel, self).__init__(name, author, description, date)
        wv_fr = KeyedVectors.load_word2vec_format(embedding_fr)
        wv_en = KeyedVectors.load_word2vec_format(embedding_en)
        wordlist = []
        with open(os.path.join(DIRECTORY, word_pairs)) as f:
            for line in f:
                spr = line.split()
                if len(spr) == 3:
                    if spr[1] in wv_fr.wv and spr[2] in wv_en.wv:
                        wordlist.append((spr[1], spr[2]))

        self.clf = Pipeline([
            ('vect', MergedTfidfEmbeddingVectorizer(wv_fr.wv, wv_en.wv, wordlist)),
            ('clf', LinearSVC()),
        ])

    @staticmethod
    def conclusion_simple(desc):
        # Mark the state of conclusion. 0 for pass and 1 for fail, adding more situation possible
        if not desc:
            return 1
        if 'Violation of Article ' in desc or 'Violation of Art. ' in desc or 'Violations of Art. ' in desc:
            return 0
        else:
            return 1

    conclusion = conclusion_simple

    @staticmethod
    def conclusion_fr(desc):
        # Mark the state of conclusion. 0 for pass and 1 for fail, adding more situation possible
        if not desc:
            return 1
        if "Violation de" in desc:
            return 0
        else:
            return 1

    @staticmethod
    def extract_input(decision_texts):
        pass

    def train(self, x, y):

        # results = [session.query(Judgments).filter_by(appno=a).with_entities(Judgments.conclusion).first() for a in new_appnos]
        assert len(x) == len(y)
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)

    def predict_svm_output(self, x):
        # import pdb; pdb.set_trace()

        # SVM output
        vec_en = self.clfs[0].clf['vect'].transform(x)
        vec_fr = self.clfs[1].clf['vect'].transform(x)
        pred_en = self.clfs[0].fscore * self.clfs[0].clf['clf'].decision_function(vec_en)
        pred_fr = self.clfs[1].fscore * self.clfs[1].clf['clf'].decision_function(vec_fr)
        pred = pred_en + pred_fr
        return np.where(pred > 0, 1, 0)

    def predict_svm_decision(self, x):
        # SVM decision
        fscore_eng = self.clfs[0].fscore
        fscore_fre = self.clfs[1].fscore
        pred_en = self.clfs[0].clf.predict(x)
        pred_fr = self.clfs[1].clf.predict(x)
        pred = []
        for ep, fp in zip(pred_en, pred_fr):
            if ep == fp == 1:
                pred.append(1)
            elif ep == fp == 0:
                pred.append(0)
            elif fscore_eng >= fscore_fre:
                pred.append(ep)
            else:
                pred.append(fp)
        return pred







        # if res == 0:
        #     if res == conclusion:
        #         self.tp += 1
        #     else:
        #         self.fp += 1
        # else:
        #     if res == conclusion:
        #         self.tn += 1
        #     else:
        #         self.fn += 1

        # return res, resn, sents, sent_result, sent_proba


if __name__ == '__main__':
    # decision_predict()
    # judgment_predict()
    pass
