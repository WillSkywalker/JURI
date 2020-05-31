from model.random_guess import RandomModel
from model.naive_bayes import NBModel_judgments, NBModel_comms
from model.lstm import BiLSTM_model, BiLSTM_trim

import os
import re
import logging
import json
import datetime
import joblib
import random
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists
from dateutil.rrule import rrule, MONTHLY
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool

from config.config import Config
from db.database import CommunicatedCases, Decisions, Judgments, Prediction, Model, ECHRArticle, Evaluation
from model.base import BaseDecisionModel
from model.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

random.seed(42)


def predict_may(date, load_model=False):
    # model = session.query(Model).filter_by(modelname=m.name, date=date, pred_type='COMM').first()

    m = NBModel_comms()
    # m = BiLSTM_trim()

    m.train(date)

    # Make predictions on cases that aren't published yet
    # for comm in session.query(CommunicatedCases):

    results = []
    golds = []
    for jdg in session.query(Judgments).filter(Judgments.kpdate > datetime.date(2020, 5, 1)).filter(Judgments.kpdate < datetime.date(2020, 5, 30)):
        comm = session.query(CommunicatedCases).filter(CommunicatedCases.appno.in_(jdg.appno.split(';')+[jdg.appno])).first()
        if not comm:
            continue

        result, proba, sents, sent_result, sent_proba = m.predict(comm)
        old = session.query(Prediction).filter_by(modelname=m.name, appno=comm.appno, pred_type='COMM').first()
        if not old:
            # jdg = session.query(Judgments).filter(Judgments.kpdate > dt).filter(Judgments.kpdate < edt)\
            #                               .filter(or_(Judgments.appno == comm.appno,
            #                                           Judgments.appno.like("{};%".format(comm.appno)),
            #                                           Judgments.appno.like("%;{}".format(comm.appno)),
            #                                           Judgments.appno.like("%;{};%".format(comm.appno)))).first()
            judgment_id = jdg.id
            jdgdate = jdg.kpdate
            gold = m.conclusion(jdg.conclusion)

            results.append(result)
            golds.append(gold)

    if golds:
        accuracy = accuracy_score(golds, results)
        fscore = f1_score(golds, results, average='macro')
        print(classification_report(golds, results))
        print(confusion_matrix(golds, results))

        print(float(accuracy))
        print(float(fscore))


def main():
    today = datetime.date.today()
    end = datetime.date(today.year, today.month, 1)
    with Pool(32) as p:
        for i in p.imap(predict_may, rrule(MONTHLY, dtstart=datetime.date(2017, 1, 1), until=end)):
            print(i)


if __name__ == '__main__':
    # jm = NBModel_judgments()
    # predict(jm, pred_type='JUDGMENTS')
    # cm = NBModel_comms()
    # predict_communicated(datetime.date(2018, 2, 1))
    main()
    # evaluate()
