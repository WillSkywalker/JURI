from model.random_guess import RandomModel
from model.naive_bayes_allcase_use_facts import NBModel_judgments

import json
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists

from config.config import Config
from db.database import CommunicatedCases, Decisions, Judgments, Prediction, Model
from model.base import BaseDecisionModel

from sklearn.metrics import accuracy_score, f1_score

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()


def decision_predict(m):
    m.train()
    for decision in session.query(Decisions):
        if decision.conclusion:
            result, proba, sents, sent_result, sent_proba = m.predict(decision)
            old = session.query(Prediction).filter_by(modelname=m.name, appno=decision.appno, pred_type='DECISIONS').first()
            if not old:
                pred = Prediction(result=result, proba=proba, sents=sents, sent_result=json.dumps(sent_result),
                                  sent_proba=json.dumps(sent_proba), modelname=m.name,
                                  appno=decision.appno, pred_type='DECISIONS', gold=m.conclusion(decision.conclusion))
                session.add(pred)
                session.commit()

    precision = m.tp / (m.tp + m.fp)
    recall = m.tp / (m.tp + m.fn)
    m = Model(modelname=m.name,
              description='A class-sample model using random guess, for decision',
              author=m.author,
              date=m.date,
              accuracy=(m.tp + m.tn) / (m.tp + m.tn + m.fp + m.fn),
              fscore=2 * (precision * recall) / (precision + recall))
    session.add(m)
    session.commit()


def predict(m, pred_type):
    m.train()
    for decision in session.query(Decisions):
        if pred_type == 'DECISIONS' or m.conclusion(decision.conclusion) == 0:
            result, proba, sents, sent_result, sent_proba = m.predict(decision)
            old = session.query(Prediction).filter_by(modelname=m.name, appno=decision.appno, pred_type=pred_type).first()
            if not old:
                pred = Prediction(result=result, proba=proba, sents=json.dumps(sents), sent_result=json.dumps(sent_result),
                                  sent_proba=json.dumps(sent_proba), modelname=m.name,
                                  appno=decision.appno, pred_type=pred_type)
                session.add(pred)
                session.commit()

    precision = m.tp / (m.tp + m.fp)
    recall = m.tp / (m.tp + m.fn)
    m = Model(modelname=m.name + pred_type,
              description=m.description,
              author=m.author,
              date=m.date,
              accuracy=(m.tp + m.tn) / (m.tp + m.tn + m.fp + m.fn),
              fscore=2 * (precision * recall) / (precision + recall))
    session.add(m)
    session.commit()

if __name__ == '__main__':
    jm = NBModel_judgments()
    predict(jm, pred_type='JUDGMENTS')
