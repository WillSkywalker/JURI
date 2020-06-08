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
from db.database import CommunicatedCases, Decisions, Judgments, Prediction, Model
from masterthesis.base import BaseDecisionModel, BaseModel
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
# from masterthesis.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError
from gensim.models.translation_matrix import Space

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
        print('DIM: ', str(self.dim))

    @staticmethod
    def dummy(x):
        return x

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=self.dummy)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] * self.word2weight[w]
                        for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0) for words in X])


class W2VModel(BaseModel):
    """naive bayes"""
    def __init__(self, embedding, name=MODEL_NAME, author=AUTHOR, description=DESCRIPTION, date=DATE):
        super(W2VModel, self).__init__(name, author, description, date)
        wv = KeyedVectors.load(os.path.join(DIRECTORY, 'embeddings/', embedding), mmap='r')
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
    def __init__(self, embedding1, embedding2, word_pairs,
                 name=MODEL_NAME, author=AUTHOR, description=DESCRIPTION, date=DATE):
        super(CombinedW2VModel, self).__init__(name, author, description, date)
        wv1 = KeyedVectors.load(os.path.join(DIRECTORY, 'embeddings/', embedding1), mmap='r')
        wv2 = KeyedVectors.load(os.path.join(DIRECTORY, 'embeddings/', embedding2), mmap='r')


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

    def train(self, X_train_eng, X_test_eng, y_train_eng, y_test_eng,
              X_train_fre, X_test_fre, y_train_fre, y_test_fre,
              x_all, y_all):

        # results = [session.query(Judgments).filter_by(appno=a).with_entities(Judgments.conclusion).first() for a in new_appnos]
        # assert len(x) == len(y)
        print(len(X_train_eng))
        print(len(X_train_fre))
        print(len(x_all))
        # self.vect.fit(x_all, y_all)
        # input_eng = self.vect.transform(X_train_eng)
        # input_fre = self.vect.transform(X_train_fre)
        # self.clf_eng.fit(input_eng, y_train_eng)
        # self.clf_fre.fit(input_fre, y_train_fre)
        # test_input_eng = self.vect.transform(X_test_eng)
        # test_input_fre = self.vect.transform(X_test_fre)
        # predict_eng = self.clf_eng.predict(test_input_eng)
        # predict_fre = self.clf_eng.predict(test_input_fre)
        # self.fscore_eng = f1_score(predict_eng, y_test_eng, average='micro')
        # print('English fscore:', str(self.fscore_eng))
        # print(classification_report(predict_eng, y_test_eng))
        # self.fscore_fre = f1_score(predict_fre, y_test_fre, average='micro')
        # print('French fscore:', str(self.fscore_fre))
        # print(classification_report(predict_fre, y_test_fre))

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
