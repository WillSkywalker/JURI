import json
import datetime
import random
import logging
import numpy as np
from collections import Counter
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

# from masterthesis.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError

MODEL_NAME = 'Baseline SVM v1'
AUTHOR = 'Xu Xiao'
DESCRIPTION = 'just svm lol'
DATE = datetime.datetime.today()


# engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
# Session = sessionmaker(bind=engine)
# session = Session()


class NBModel_judgments(BaseDecisionModel):
    """naive bayes"""
    def __init__(self, name=MODEL_NAME, author=AUTHOR, date=DATE, description=DESCRIPTION):
        super(NBModel_judgments, self).__init__(name, author, description, date)
        self.clf = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', MultinomialNB()),
        ])

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
        if 'Violation of Article ' in desc or 'Violation of Art. ' in desc or 'Violations of Art. ' in desc:
            return 0
        else:
            return 1

    conclusion = conclusion_simple

    @staticmethod
    def extract_input(decision_texts):
        pass

    def train(self):
        # get all appnos
        # judgs = session.query(Judgments).with_entities(Judgments.appno).all()
        # decs = session.query(Decisions).with_entities(Decisions.appno).all()
        # appnos = set(judgs) & set(decs)
        # appnos = [i[0] for i in appnos]

        # get all text
        # decisions = Decisions.query.filter(Decisions.appno.in_(appnos)).with_entities(Decisions.text).all()
        decisions = session.query(Decisions).filter(exists().where(Decisions.appno == Judgments.appno)).all()
        decisions = [decision for decision in decisions if self.admissibility(decision.conclusion) == 0]

        # Filter by time
        # decisions = [session.query(Decisions).filter_by(appno=a).filter(Decisions.date < datetime.date(2019, 1, 1)).with_entities(Decisions.text).first() for a in appnos]

        # decisions = [d for d in decisions if self.admissibility(d.conclusion) == 0]
        # texts = [d.text for d in decisions]  # convert to str
        # texts = [d.split('\n') for d in texts]  # add nltk sent tokenization?
        #decisions = [json.loads(d.sents) for d in decisions]

        # decisions = [' '.join(extract_parts_judgments(d)[7]) for d in decisions]
        new_appnos = []
        new_decisions = []
        for d in decisions:
            try:
                print('OOOOO:', d.appno)
                text = d.text.split('\n')
                new_decisions.append(' '.join(extract_parts_judgments(text)[7]))
                new_appnos.append(d.appno)
            except JudgmentNoTextError:
                logging.warning(d.appno)

        # In case you need CommunicatedCases
        # ccs = CommunicatedCases.query.filter(CommunicatedCases.appno.in_(appnos)).all()

        # all conclusions (strings)
        # results = Judgments.query.filter(Judgments.appno.in_(appnos)).with_entities(Judgments.conclusion).all()
        results = [session.query(Judgments).filter_by(appno=a).with_entities(Judgments.conclusion).first() for a in new_appnos]
        results = [self.conclusion_simple(res.conclusion) for res in results]  # convert to label in integer
        maximum_sample = Counter(results).most_common()[-1][1]
        balanced_decisions = []
        balanced_results = []
        sample_count = {0: 0, 1: 0}
        for idx, res in enumerate(results):
            sample_count[res] += 1
            if sample_count[res] < maximum_sample:
                balanced_decisions.append(new_decisions[idx])
                balanced_results.append(res)
        assert len(new_decisions) == len(results)
        self.clf.fit(balanced_decisions, balanced_results)

    def predict(self, x):
        return self.clf.predict(x)

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


class NBModel(BaseModel):
    """naive bayes"""
    def __init__(self, name=MODEL_NAME, author=AUTHOR, description=DESCRIPTION, date=DATE):
        super(NBModel, self).__init__(name, author, description, date)
        self.clf = Pipeline([
            ('vect', TfidfVectorizer()),
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


class CombinedModel(BaseModel):
    """naive bayes"""
    def __init__(self, clfs, name=MODEL_NAME, author=AUTHOR, description=DESCRIPTION, date=DATE):
        super(CombinedModel, self).__init__(name, author, description, date)
        self.vect = TfidfVectorizer()
        self.vect_eng = TfidfVectorizer()
        self.vect_fre = TfidfVectorizer()
        self.clf_eng = LinearSVC()
        self.clf_fre = LinearSVC()
        self.clfs = clfs

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
