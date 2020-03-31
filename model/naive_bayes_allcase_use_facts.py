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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from model.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError

MODEL_NAME = 'Balanced Naive Bayes all cases v2'
AUTHOR = 'Xu Xiao'
DESCRIPTION = 'Naive Bayes model using the fact section of Admissibility documents'
DATE = datetime.datetime.today()


engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()


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
        conclusion = self.conclusion(x.conclusion)
        resn = random.random()
        sents = json.loads(x.sents)#[:20]  # suppose we use first 20 sents
        try:
            sents = extract_parts_judgments(sents)[7]  # takes in sentences and separates into parts of the case [7] is circumstances
        except JudgmentNoTextError:
            return -1, 1, [], [], []
        string = ' '.join(sents)  # for prediction
        res = int(self.clf.predict([string])[0])  # class
        resn = float(self.clf.predict_proba([string])[0][res])  # proba

        sent_result = []
        sent_proba = []
        for sent in sents:
            sent_res = self.clf.predict([sent])[0]  # class
            sent_resn = self.clf.predict_proba([sent])[0][sent_res]  # proba
            sent_result.append(int(sent_res))
            sent_proba.append(float(sent_resn))

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

        return res, resn, sents, sent_result, sent_proba


class NBModel_comms(BaseCommunicatedCasesModel):
    """naive bayes"""
    def __init__(self, name=MODEL_NAME, author=AUTHOR, description=DESCRIPTION, date=DATE):
        super(NBModel_comms, self).__init__(name, author, description, date)
        self.clf = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', MultinomialNB()),
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
    def extract_input(decision_texts):
        pass

    def train(self):

        # comms = session.query(CommunicatedCases).all()
        comms = session.query(CommunicatedCases).filter(exists().where(Judgments.appno == CommunicatedCases.appno)).all()
        new_appnos = []
        new_comms = []
        for d in comms:
            try:
                print('OOOOO:', d.appno)
                # text = d.text.split('\n')
                new_comms.append(d.text)
                new_appnos.append(d.appno)
            except JudgmentNoTextError:
                logging.warning(d.appno)

        # In case you need CommunicatedCases
        # ccs = CommunicatedCases.query.filter(CommunicatedCases.appno.in_(appnos)).all()

        # all conclusions (strings)
        # results = Judgments.query.filter(Judgments.appno.in_(appnos)).with_entities(Judgments.conclusion).all()
        results = []
        for a in new_appnos:
            j = session.query(Judgments).filter(or_(Judgments.appno == a,
                                                    Judgments.appno.like("{};%".format(a)),
                                                    Judgments.appno.like("%;{}".format(a)),
                                                    Judgments.appno.like("%;{};%".format(a)))).with_entities(Judgments.conclusion).first()
            if j:
                results.append(self.conclusion_simple(j.conclusion))
            else:
                results.append(1)

        violation_num = Counter(results)[0] - Counter(results)[1]
        for comm in random.sample(session.query(CommunicatedCases).filter(~exists().where(Judgments.appno == CommunicatedCases.appno)).all(), violation_num):
            # if i >= violation_num:
            #     break
            new_comms.append(comm.text)
            # j = session.query(Judgments).filter_by(appno=comm.appno).with_entities(Judgments.conclusion).first()
            results.append(1)

        # results = [session.query(Judgments).filter_by(appno=a).with_entities(Judgments.conclusion).first() for a in new_appnos]
        assert len(new_comms) == len(results)
        self.clf.fit(new_comms, results)

    def predict(self, x):
        # conclusion = self.conclusion(x.conclusion)
        resn = random.random()
        sents = json.loads(x.sents)  # [:20]  # suppose we use first 20 sents

        string = ' '.join(sents)  # for prediction
        res = int(self.clf.predict([string])[0])  # class
        resn = float(self.clf.predict_proba([string])[0][res])  # proba

        sent_result = []
        sent_proba = []
        for sent in sents:
            sent_res = self.clf.predict([sent])[0]  # class
            sent_resn = self.clf.predict_proba([sent])[0][sent_res]  # proba
            sent_result.append(int(sent_res))
            sent_proba.append(float(sent_resn))

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

        return res, resn, sents, sent_result, sent_proba


if __name__ == '__main__':
    # decision_predict()
    # judgment_predict()
    pass
