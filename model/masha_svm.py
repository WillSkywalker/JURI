import re
import json
import datetime
import random
import logging
from collections import Counter
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists

from config.config import Config
from db.database import CommunicatedCases, Decisions, Judgments, Prediction, Model
from model.base import BaseDecisionModel, BaseCommunicatedCasesModel
from collections import Counter
from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from model.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError

MODEL_NAME = 'Masha\'s SVM'
AUTHOR = 'Masha Medvedeva'
DESCRIPTION = 'Russia\'s Finest Classifier'
DATE = datetime.datetime.today()


engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()


class Masha_SVM(BaseCommunicatedCasesModel):
    """Masha's SVM"""
    def __init__(self, name=MODEL_NAME, author=AUTHOR, description=DESCRIPTION, date=DATE):
        super(Masha_SVM, self).__init__(name, author, description, date)
        c = 5
        vec = ('wordvec', TfidfVectorizer(analyzer='word', binary=True, lowercase=True, min_df=2,  ngram_range=(2, 4), norm='l2', stop_words=None, use_idf=True))
        self.clf = Pipeline([vec,
                            ('classifier', SVC(kernel='linear', probability=True, C=c))])

    @staticmethod
    def admissibility(desc):
        '''you may override this with your own implementation'''
        if not desc:
            return 1
        if 'Admissible' in desc or 'Partly ' in desc:
            return 0
        else:
            return 1

    @staticmethod
    def conclusion_simple(desc):
        # Mark the state of conclusion. 0 for pass and 1 for fail, adding more situation possible
        if not desc:
            return 1
        if 'Violation of Article ' in desc or 'Violation of Art. ' in desc or 'Violations of Art. ' in desc or 'Violations of Article ' in desc:
            return 0
        else:
            return 1

    conclusion = conclusion_simple

    @staticmethod
    def extract_input(t):
        if t:
            text = t.split('\n')
            text = [re.sub('^.*v\. .* STATEMENT OF.*', '', i) for i in text]  # remove the tietles of the pages from getting text from pdf
            text = [re.sub('STATEMENT OF FACTS( AND QUESTIONS)?', '', i) for i in text]  # remove the tietles of the pages from getting text from pdf
            text = [re.sub('[A-ZĐĆ ]+ v. [A-Z ]+', '', i) for i in text]
            text = [re.sub('^\n$', '', i) for i in text]  # remove end of the lines
            text = [re.sub('\n', '', i) for i in text]
            text = '  '.join(text)  # combine lines
            text = re.sub('  +', ' ', text)
            m = re.search('(.+) QUESTIONS', text)
            m2 = re.search('(.+) COMPLAINTS', text)
            if m != None:
                text = m.group(1)
            elif m2 != None:
                text = m2.group(1)
        else:
            text = t
        return text

    def train(self, date):
        dt = datetime.datetime.combine(date, datetime.datetime.min.time())
        OLDEST = datetime.datetime(2010, 1, 1)

        texts = []
        labels = []
        appnos = set([])

        for jdg in session.query(Judgments).filter(Judgments.kpdate > OLDEST).filter(Judgments.kpdate < dt):
            if not jdg.appno:
                continue
            comm = session.query(CommunicatedCases).filter(CommunicatedCases.appno.in_(jdg.appno.split(';')+[jdg.appno])).first()
            if not comm:
                continue

            texts.append(self.extract_input(comm.text))
            labels.append(self.conclusion_simple(jdg.conclusion))
            appnos.add(comm.appno)

        violation_num = Counter(labels)[0] - Counter(labels)[1]

        desc_inputs = []
        for desc in session.query(Decisions).filter(Decisions.kpdate > OLDEST).filter(Decisions.kpdate < dt):
            comm = session.query(CommunicatedCases).filter(exists().where(Decisions.appno == CommunicatedCases.appno)).first()

            if not comm:
                continue

            desc_inputs.append((self.extract_input(comm.text), 1))

        for d in random.sample(desc_inputs, violation_num):
            texts.append(d[0])
            labels.append(d[1])

        print('Violation: ', Counter(labels)[0], 'Non-violation: ', Counter(labels)[1])
        self.clf.fit(texts, labels)

    def predict(self, x):
        # conclusion = self.conclusion(x.conclusion)
        resn = random.random()
        text = self.extract_input(x.text)
        sents = sent_tokenize(text)

        # string = ' '.join(sents)  # for prediction
        res = int(self.clf.predict([text])[0])  # class
        resn = float(self.clf.predict_proba([text])[0][res])  # proba

        sent_result = []
        sent_proba = []
        for sent in sents:
            sent_res = self.clf.predict([sent])[0]  # class
            sent_resn = self.clf.predict_proba([sent])[0][sent_res]  # proba
            sent_result.append(int(sent_res))
            sent_proba.append(float(sent_resn))

        return res, resn, sents, sent_result, sent_proba


if __name__ == '__main__':
    # decision_predict()
    # judgment_predict()
    pass
