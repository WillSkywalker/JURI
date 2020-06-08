from masterthesis.naive_bayes_allcase_use_facts import NBModel_judgments, NBModel, CombinedModel

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
from collections import Counter

from config.config import Config
from masterthesis.db import CommunicatedCases, Decisions, Judgments, Prediction, Model, ECHRArticle, CommunicatedCases_FRE, Decisions_FRE, Judgments_FRE
from masterthesis.base import BaseDecisionModel
from masterthesis.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def predict_en(m, load_model=False):

    # comms = session.query(CommunicatedCases).all()
    decs = session.query(Decisions).filter(exists().where(Judgments.appno == Decisions.appno)).all()
    new_appnos = []
    new_decs = []
    for d in decs:
        try:
            print('OOOOO:', d.appno)
            # text = d.text.split('\n')
            new_decs.append(d.text)
            new_appnos.append(d.appno)
        except JudgmentNoTextError:
            logging.warning(d.appno)

    # all conclusions (strings)
    results = []
    for a in new_appnos:
        j = session.query(Judgments).filter(Judgments.appno == a).with_entities(Judgments.conclusion).first()
        if j:
            results.append(m.conclusion_simple(j.conclusion))
        else:
            results.append(1)

    violation_num = Counter(results)[0] - Counter(results)[1]
    for comm in random.sample(session.query(Decisions).filter(~exists().where(Judgments.appno == Decisions.appno)).all(), violation_num):
        # if i >= violation_num:
        #     break
        new_decs.append(comm.text)
        # j = session.query(Judgments).filter_by(appno=comm.appno).with_entities(Judgments.conclusion).first()
        results.append(1)

    X_train, X_test, y_train, y_test = train_test_split(new_decs, results)
    if load_model:
        m.clf = joblib.load(os.path.join(DIRECTORY, 'models/', m.name+'.joblib'))
    else:
        m.train(X_train, y_train)

    # for comm in session.query(Decisions):
    #     result, proba, sents, sent_result, sent_proba = m.predict(comm)

    predictions = m.clf.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    fscore = f1_score(predictions, y_test, average='micro')
    logging.critical(classification_report(predictions, y_test))
    logging.critical(confusion_matrix(predictions, y_test))
    if not os.path.exists(os.path.join(DIRECTORY, 'models/')):
        os.makedirs(os.path.join(DIRECTORY, 'models/'))
    if not load_model:
        joblib.dump(m.clf, os.path.join(DIRECTORY, 'models/', m.name+'.joblib'))



def predict_fr(m, load_model=False):

    # comms = session.query(CommunicatedCases).all()
    decs = session.query(Decisions_FRE).filter(exists().where(Judgments_FRE.appno == Decisions_FRE.appno)).all()
    new_appnos = []
    new_decs = []
    for d in decs:
        try:
            print('OOOOO:', d.appno)
            # text = d.text.split('\n')
            new_decs.append(d.text)
            new_appnos.append(d.appno)
        except JudgmentNoTextError:
            logging.warning(d.appno)

    # all conclusions (strings)
    results = []
    for a in new_appnos:
        j = session.query(Judgments_FRE).filter(Judgments_FRE.appno == a).with_entities(Judgments_FRE.conclusion).first()
        if j:
            results.append(m.conclusion_fr(j.conclusion))
        else:
            results.append(1)

    i = 0
    violation_num = Counter(results)[0] - Counter(results)[1]
    for comm in session.query(Decisions_FRE).filter(~exists().where(Judgments_FRE.appno == Decisions_FRE.appno)).all():
        if i >= violation_num:
            break
        new_decs.append(comm.text)
        i += 1
        # j = session.query(Judgments).filter_by(appno=comm.appno).with_entities(Judgments.conclusion).first()
        results.append(1)

    X_train, X_test, y_train, y_test = train_test_split(new_decs, results)
    if load_model:
        m.clf = joblib.load(os.path.join(DIRECTORY, 'models/', m.name+'.joblib'))
    else:
        m.train(X_train, y_train)

    # for comm in session.query(Decisions):
    #     result, proba, sents, sent_result, sent_proba = m.predict(comm)

    predictions = m.clf.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    fscore = f1_score(predictions, y_test, average='micro')
    logging.critical(classification_report(predictions, y_test))
    logging.critical(confusion_matrix(predictions, y_test))
    if not os.path.exists(os.path.join(DIRECTORY, 'models/')):
        os.makedirs(os.path.join(DIRECTORY, 'models/'))
    if not load_model:
        joblib.dump(m.clf, os.path.join(DIRECTORY, 'models/', m.name+'_fr.joblib'))


def predict_all(m, load_model=False):
    # comms = session.query(CommunicatedCases).all()
    decs_fr = session.query(Decisions_FRE).filter(exists().where(Judgments_FRE.appno == Decisions_FRE.appno)).all()
    decs_en = session.query(Decisions).filter(exists().where(Judgments_FRE.appno == Decisions.appno)).all()
    decs = decs_en + decs_fr
    new_appnos = []
    new_decs = []
    for d in decs:
        try:
            print('OOOOO:', d.appno)
            # text = d.text.split('\n')
            new_decs.append(d.text)
            new_appnos.append(d.appno)
        except JudgmentNoTextError:
            logging.warning(d.appno)

    # all conclusions (strings)
    results = []
    for a in new_appnos:
        j = session.query(Judgments_FRE).filter(Judgments_FRE.appno == a).with_entities(Judgments_FRE.conclusion).first()
        if j:
            results.append(m.conclusion_fr(j.conclusion))
        else:
            j = session.query(Judgments).filter(Judgments.appno == a).with_entities(Judgments.conclusion).first()
            if j:
                results.append(m.conclusion(j.conclusion))
            else:
                results.append(1)

    i = 0
    violation_num = Counter(results)[0] - Counter(results)[1] / 2
    for comm in session.query(Decisions_FRE).filter(~exists().where(Judgments_FRE.appno == Decisions_FRE.appno)).all():
        if i >= violation_num:
            break
        new_decs.append(comm.text)
        i += 1
        # j = session.query(Judgments).filter_by(appno=comm.appno).with_entities(Judgments.conclusion).first()
        results.append(1)

    j = 0
    for comm in session.query(Decisions).filter(~exists().where(Judgments.appno == Decisions.appno)).all():
        if j >= violation_num:
            break
        new_decs.append(comm.text)
        j += 1
        # j = session.query(Judgments).filter_by(appno=comm.appno).with_entities(Judgments.conclusion).first()
        results.append(1)

    X_train, X_test, y_train, y_test = train_test_split(new_decs, results)
    if load_model:
        m.clf = joblib.load(os.path.join(DIRECTORY, 'models/', m.name+'.joblib'))
    else:
        m.train(X_train, y_train)

    # for comm in session.query(Decisions):
    #     result, proba, sents, sent_result, sent_proba = m.predict(comm)

    predictions = m.clf.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    fscore = f1_score(predictions, y_test, average='micro')
    logging.critical(classification_report(predictions, y_test))
    logging.critical(confusion_matrix(predictions, y_test))
    if not os.path.exists(os.path.join(DIRECTORY, 'models/')):
        os.makedirs(os.path.join(DIRECTORY, 'models/'))
    if not load_model:
        joblib.dump(m.clf, os.path.join(DIRECTORY, 'models/', m.name+'_fr.joblib'))

if __name__ == '__main__':
    # jm = NBModel_judgments()
    # predict(jm, pred_type='JUDGMENTS')
    logging.basicConfig(filename='master.log', level=logging.CRITICAL)

    em = NBModel()
    predict_en(em)
    fm = NBModel()
    predict_fr(fm)
    am = CombinedModel([('eng', em.clf['clf']), ('fre', fm.clf['clf'])])
    predict_en(am)

    # predict_en(cm, load_model=True)
