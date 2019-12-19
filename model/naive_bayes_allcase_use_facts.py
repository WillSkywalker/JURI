import json
import datetime
import random
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.config import Config
from db.database import CommunicatedCases, Decisions, Judgments, Prediction, Model
from model.base import BaseDecisionModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from model.extract_facts_judgments import extract_parts_judgments

MODEL_NAME = 'Naive Bayes all cases'
AUTHOR = 'Xu Xiao'
DATE = datetime.datetime.now()


engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()


class NBModel_judgments(BaseDecisionModel):
    """naive bayes"""
    def __init__(self, name=MODEL_NAME, author=AUTHOR, date=DATE):
        super(NBModel_judgments, self).__init__(name, author, date)
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

    def train(self):
        # get all appnos
        judgs = session.query(Judgments).with_entities(Judgments.appno).all()
        decs = session.query(Decisions).with_entities(Decisions.appno).all()
        appnos = set(judgs) & set(decs)
        appnos = [i[0] for i in appnos]

        # get all text
        # decisions = Decisions.query.filter(Decisions.appno.in_(appnos)).with_entities(Decisions.text).all()
        decisions = [session.query(Decisions).filter_by(appno=a).with_entities(Decisions.text).first() for a in appnos]
        #decisions = [d.text for d in decisions]  # convert to str
        decisions = [d.sents for d in decisions]
        decisions = [' '.join(extract_parts_judgments(d)[7]) for d in decisions]

        # In case you need CommunicatedCases
        # ccs = CommunicatedCases.query.filter(CommunicatedCases.appno.in_(appnos)).all()

        # all conclusions (strings)
        # results = Judgments.query.filter(Judgments.appno.in_(appnos)).with_entities(Judgments.conclusion).all()
        results = [session.query(Judgments).filter_by(appno=a).with_entities(Judgments.conclusion).first() for a in appnos]
        results = [self.conclusion_simple(res.conclusion) for res in results]  # convert to label in integer

        self.clf.fit(decisions, results)

    def predict(self, x):
        conclusion = self.conclusion(x.conclusion)
        resn = random.random()
        sents = json.loads(x.sents)#[:20]  # suppose we use first 20 sents
        sents = extract_parts_judgments(sents)[7] #takes in sentences and separates into parts of the case [7] is circumstances
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

        if res == 0:
            if res == conclusion:
                self.tp += 1
            else:
                self.fp += 1
        else:
            if res == conclusion:
                self.tn += 1
            else:
                self.fn += 1

        return res, resn, sents, sent_result, sent_proba


if __name__ == '__main__':
    decision_predict()
    judgment_predict()
