from model.random_guess import RandomModel
from model.naive_bayes import NBModel_judgments, NBModel_comms
from model.lstm import BiLSTM_model, BiLSTM_trim
from model.masha_svm import Masha_SVM

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


def test(date, m):
    dt = datetime.datetime.combine(date, datetime.datetime.min.time())
    m.train(date)

    results = []
    golds = []
    for jdg in session.query(Judgments).filter(Judgments.kpdate > datetime.datetime(2020, 1, 1)):
        comm = session.query(CommunicatedCases).filter(CommunicatedCases.appno.in_(jdg.appno.split(';')+[jdg.appno])).first()
        if not comm:
            continue

        result, proba, sents, sent_result, sent_proba = m.predict(comm)
        gold = m.conclusion(jdg.conclusion)
        results.append(result)
        golds.append(gold)

    accuracy = accuracy_score(golds, results)
    fscore = f1_score(golds, results, average='macro')
    print(m.name)
    print('=======================================\n\n')
    print('Accuracy: ', accuracy)
    print('Accuracy: ', fscore)
    print('\nClassification report:\n', classification_report(golds, results))
    print('\nConfusion matrix:\n', confusion_matrix(golds, results), '\n\n_______________________\n\n')
    print('\n Normalized confusion matrix:\n', confusion_matrix(golds, results, normalize='true'), '\n\n_______________________\n\n')


def main():
    dt = datetime.date(2011, 1, 1)
    # test(dt, NBModel_comms())
    # test(dt, BiLSTM_trim())
    test(dt, Masha_SVM())

if __name__ == '__main__':
    main()
