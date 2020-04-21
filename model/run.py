from model.random_guess import RandomModel
from model.naive_bayes_allcase_use_facts import NBModel_judgments, NBModel_comms

import os
import logging
import json
import datetime
import joblib
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists

from config.config import Config
from db.database import CommunicatedCases, Decisions, Judgments, Prediction, Model
from model.base import BaseDecisionModel
from model.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def decision_predict(m):
    m.train()
    for decision in session.query(Decisions):
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
    m = Model(
        modelname=m.name,
        description='A class-sample model using random guess, for decision',
        author=m.author,
        date=m.date,
        pred_type='DECISIONS',
        accuracy=(m.tp + m.tn) / (m.tp + m.tn + m.fp + m.fn),
        fscore=2 * (precision * recall) / (precision + recall))
    session.add(m)
    session.commit()


def predict(m, pred_type):
    # Train
    m.train()

    # Make predictions on cases that aren't published yet
    for decision in session.query(Decisions).filter(~exists().where(Decisions.appno == Judgments.appno)):
        if pred_type == 'DECISIONS' or m.admissibility(decision.conclusion) == 0:
            result, proba, sents, sent_result, sent_proba = m.predict(decision)
            old = session.query(Prediction).filter_by(modelname=m.name, appno=decision.appno, pred_type=pred_type).first()
            if not old:
                pred = Prediction(result=result, proba=proba, sents=json.dumps(sents), sent_result=json.dumps(sent_result),
                                  sent_proba=json.dumps(sent_proba), modelname=m.name, kpdate=decision.kpdate,
                                  appno=decision.appno, pred_type=pred_type)
                session.add(pred)
                session.commit()

    # Evaluation, further report saved at local
    jdgs = session.query(Judgments).filter(exists().where(Decisions.appno == Judgments.appno)).limit(100).all()
    appnos = [j.appno for j in jdgs]
    #ds = session.query(Decisions).filter(Decisions.appno.in_(appnos)).all()
    ds = [session.query(Decisions).filter_by(appno=appno).first() for appno in appnos]
    ds = [d.text for d in ds]
    ds = [d.split('\n') for d in ds]
    new_appnos = []
    testset = []
    for i, d in enumerate(ds):
        try:
            testset.append(' '.join(extract_parts_judgments(d)[7]))
            new_appnos.append(appnos[i])
        except JudgmentNoTextError:
            logging.warning(appnos[i])

    results = [session.query(Judgments).filter_by(appno=a).with_entities(Judgments.conclusion).first() for a in new_appnos]
    results = [m.conclusion(res.conclusion) for res in results]
    assert len(testset) == len(results)
    predictions = m.clf.predict(testset)
    accuracy = float(accuracy_score(predictions, results))
    fscore = float(f1_score(predictions, results, average='micro'))
    logging.warning(classification_report(predictions, results))
    logging.warning(confusion_matrix(predictions, results))

    m = Model(modelname=m.name,
              description=m.description,
              author=m.author,
              date=m.date,
              pred_type=pred_type,
              accuracy=accuracy,
              fscore=fscore)
    session.add(m)
    session.commit()


def predict_communicated(m, load_model=False):
    if load_model:
        m.clf = joblib.load(os.path.join(DIRECTORY, 'models/', m.name+'.joblib'))
    else:
        m.train()

    # Make predictions on cases that aren't published yet
    for comm in session.query(CommunicatedCases)[:100]:
    # for comm in session.query(CommunicatedCases):
        result, proba, sents, sent_result, sent_proba = m.predict(comm)
        old = session.query(Prediction).filter_by(modelname=m.name, appno=comm.appno, pred_type='COMM').first()
        if not old:
            jdg = session.query(Judgments).filter(or_(Judgments.appno == comm.appno,
                                                      Judgments.appno.like("{};%".format(comm.appno)),
                                                      Judgments.appno.like("%;{}".format(comm.appno)),
                                                      Judgments.appno.like("%;{};%".format(comm.appno)))).first()
            judgment_id = jdg.id if jdg else None
            jdgdate = jdg.kpdate if jdg else None
            gold = m.conclusion(jdg.conclusion) if jdg else None
            pred = Prediction(gold=gold, result=result, proba=proba, sents=json.dumps(sents), sent_result=json.dumps(sent_result),
                              sent_proba=json.dumps(sent_proba), modelname=m.name, kpdate=comm.kpdate, jdgdate=jdgdate,
                              appno=comm.appno, pred_type='COMM', judgment_id=judgment_id)
            session.add(pred)
            session.commit()

    # Evaluation, further report saved at local
    jdgs = session.query(Judgments).filter(exists().where(CommunicatedCases.appno == Judgments.appno)).limit(100).all()
    appnos = [j.appno for j in jdgs]
    ds = [session.query(CommunicatedCases).filter_by(appno=appno).first() for appno in appnos]
    ds = [d.text for d in ds]

    results = [session.query(Judgments).filter_by(appno=a).with_entities(Judgments.conclusion).first() for a in appnos]
    results = [m.conclusion(res.conclusion) for res in results]
    assert len(ds) == len(results)
    predictions = m.clf.predict(ds)
    accuracy = accuracy_score(predictions, results)
    fscore = f1_score(predictions, results, average='micro')
    logging.warning(classification_report(predictions, results))
    logging.warning(confusion_matrix(predictions, results))
    if not os.path.exists(os.path.join(DIRECTORY, 'models/')):
        os.makedirs(os.path.join(DIRECTORY, 'models/'))
    if not load_model:
        joblib.dump(m.clf, os.path.join(DIRECTORY, 'models/', m.name+'.joblib'))

    m = Model(modelname=m.name,
              description=m.description,
              author=m.author,
              date=m.date,
              pred_type='COMM',
              accuracy=float(accuracy),
              fscore=float(fscore))
    session.add(m)
    session.commit()


if __name__ == '__main__':
    # jm = NBModel_judgments()
    # predict(jm, pred_type='JUDGMENTS')
    cm = NBModel_comms()
    # predict_communicated(cm)
    predict_communicated(cm, load_model=True)
