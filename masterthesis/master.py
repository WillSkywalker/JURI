from masterthesis.naive_bayes_allcase_use_facts import NBModel, CombinedModel
from masterthesis.w2v import W2VModel, CombinedW2VModel

import os
import re
import logging
import json
import datetime
import joblib
import random
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists
from collections import Counter

from config.config import Config
from masterthesis.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def predict_en(m, om=None, load_model=False):

    df = pd.read_csv('eng.csv')
    new_appnos = df['appno']
    new_decs = df['text']

    # all conclusions (strings)
    results = df['result']

    X_train, X_test, y_train, y_test = train_test_split(new_decs, results)
    if load_model:
        m.clf = joblib.load(os.path.join(DIRECTORY, 'models/', 'en_'+m.name+'.joblib'))
    else:
        m.train(X_train, y_train)

    # for comm in session.query(Decisions):
    #     result, proba, sents, sent_result, sent_proba = m.predict(comm)

    predictions = m.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    fscore = f1_score(predictions, y_test, average='micro')
    logging.critical('\nEnglish\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    m.fscore = fscore
    logging.critical(classification_report(predictions, y_test))
    logging.critical(confusion_matrix(predictions, y_test))
    if not os.path.exists(os.path.join(DIRECTORY, 'models/')):
        os.makedirs(os.path.join(DIRECTORY, 'models/'))
    if not load_model:
        joblib.dump(m.clf, os.path.join(DIRECTORY, 'models/', 'en_'+m.name+'.joblib'))
    return X_train, X_test, y_train, y_test


def predict_fr(m, load_model=False):

    df = pd.read_csv('fre.csv')
    new_appnos = df['appno']
    new_decs = df['text']

    # all conclusions (strings)
    results = df['result']

    X_train, X_test, y_train, y_test = train_test_split(new_decs, results)
    if load_model:
        m.clf = joblib.load(os.path.join(DIRECTORY, 'models/', 'fr_'+m.name+'.joblib'))
    else:
        m.train(X_train, y_train)

    # for comm in session.query(Decisions):
    #     result, proba, sents, sent_result, sent_proba = m.predict(comm)

    predictions = m.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    fscore = f1_score(predictions, y_test, average='micro')
    logging.critical('\nFrench\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    m.fscore = fscore
    logging.critical(classification_report(predictions, y_test))
    logging.critical(confusion_matrix(predictions, y_test))
    if not os.path.exists(os.path.join(DIRECTORY, 'models/')):
        os.makedirs(os.path.join(DIRECTORY, 'models/'))
    if not load_model:
        joblib.dump(m.clf, os.path.join(DIRECTORY, 'models/', 'fr_'+m.name+'.joblib'))
    return X_train, X_test, y_train, y_test


# def predict_all(m, load_model=False):
#     # comms = session.query(CommunicatedCases).all()
#     decs_fr = session.query(Decisions_FRE).filter(exists().where(Judgments_FRE.appno == Decisions_FRE.appno)).all()
#     decs_en = session.query(Decisions).filter(exists().where(Judgments_FRE.appno == Decisions.appno)).all()
#     decs = decs_en + decs_fr
#     new_appnos = []
#     new_decs = []
#     for d in decs:
#         try:
#             print('OOOOO:', d.appno)
#             # text = d.text.split('\n')
#             new_decs.append(d.text)
#             new_appnos.append(d.appno)
#         except JudgmentNoTextError:
#             logging.warning(d.appno)

#     # all conclusions (strings)
#     results = []
#     for a in new_appnos:
#         j = session.query(Judgments_FRE).filter(Judgments_FRE.appno == a).with_entities(Judgments_FRE.conclusion).first()
#         if j:
#             results.append(m.conclusion_fr(j.conclusion))
#         else:
#             j = session.query(Judgments).filter(Judgments.appno == a).with_entities(Judgments.conclusion).first()
#             if j:
#                 results.append(m.conclusion(j.conclusion))
#             else:
#                 results.append(1)

#     i = 0
#     violation_num = Counter(results)[0] - Counter(results)[1] / 2
#     for comm in session.query(Decisions_FRE).filter(~exists().where(Judgments_FRE.appno == Decisions_FRE.appno)).all():
#         if i >= violation_num:
#             break
#         new_decs.append(comm.text)
#         i += 1
#         # j = session.query(Judgments).filter_by(appno=comm.appno).with_entities(Judgments.conclusion).first()
#         results.append(1)

#     j = 0
#     for comm in session.query(Decisions).filter(~exists().where(Judgments.appno == Decisions.appno)).all():
#         if j >= violation_num:
#             break
#         new_decs.append(comm.text)
#         j += 1
#         # j = session.query(Judgments).filter_by(appno=comm.appno).with_entities(Judgments.conclusion).first()
#         results.append(1)

#     X_train, X_test, y_train, y_test = train_test_split(new_decs, results)
#     if load_model:
#         m.clf = joblib.load(os.path.join(DIRECTORY, 'models/', m.name+'.joblib'))
#     else:
#         m.train(X_train, y_train)

#     # for comm in session.query(Decisions):
#     #     result, proba, sents, sent_result, sent_proba = m.predict(comm)

#     predictions = m.clf.predict(X_test)
#     accuracy = accuracy_score(predictions, y_test)
#     fscore = f1_score(predictions, y_test, average='micro')
#     logging.critical(classification_report(predictions, y_test))
#     logging.critical(confusion_matrix(predictions, y_test))
#     if not os.path.exists(os.path.join(DIRECTORY, 'models/')):
#         os.makedirs(os.path.join(DIRECTORY, 'models/'))
#     if not load_model:
#         joblib.dump(m.clf, os.path.join(DIRECTORY, 'models/', m.name+'_fr.joblib'))


def predict_all(m, X_train_eng, X_test_eng, y_train_eng, y_test_eng, X_train_fre, X_test_fre, y_train_fre, y_test_fre, load_model=False):
    df1 = pd.read_csv('eng.csv')
    df2 = pd.read_csv('fre.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    eng_texts = df1['text']
    fre_texts = df2['text']
    all_texts = df['text']
    eng_results = df1['result']
    fre_results = df2['result']
    all_results = df['result']

    X_train = pd.concat([X_train_eng, X_train_fre], ignore_index=True)
    X_test = pd.concat([X_test_eng, X_test_fre], ignore_index=True)
    y_train = pd.concat([y_train_eng, y_train_fre], ignore_index=True)
    y_test = pd.concat([y_test_eng, y_test_fre], ignore_index=True)

    if load_model:
        m.clf = joblib.load(os.path.join(DIRECTORY, 'models/', m.name+'.joblib'))
    else:
        m.train(X_train, y_train)

    # for comm in session.query(Decisions):
    #     result, proba, sents, sent_result, sent_proba = m.predict(comm)

    predictions = m.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    fscore = f1_score(predictions, y_test, average='micro')
    logging.critical('\nAll cases\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test))
    logging.critical(confusion_matrix(predictions, y_test))


def predict_all_train(m, X_train_eng, X_test_eng, y_train_eng, y_test_eng, X_train_fre, X_test_fre, y_train_fre, y_test_fre, load_model=False):
    df1 = pd.read_csv('eng.csv')
    df2 = pd.read_csv('fre.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    eng_texts = df1['text']
    fre_texts = df2['text']
    all_texts = df['text']
    eng_results = df1['result']
    fre_results = df2['result']
    all_results = df['result']

    # X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(eng_texts, eng_results, random_state=42)
    # X_train_fre, X_test_fre, y_train_fre, y_test_fre = train_test_split(fre_texts, fre_results, random_state=42)
    X_train_all = pd.concat([X_train_eng, X_train_fre])
    X_test_all = pd.concat([X_test_eng, X_test_fre])
    y_train_all = pd.concat([y_train_eng, y_train_fre])
    y_test_all = pd.concat([y_test_eng, y_test_fre])

    if load_model:
        m.clf = joblib.load(os.path.join(DIRECTORY, 'models/', m.name+'.joblib'))
    else:
        m.train(X_train_eng, X_test_eng, y_train_eng, y_test_eng,
                X_train_fre, X_test_fre, y_train_fre, y_test_fre,
                X_train_all, y_train_all)

    # for comm in session.query(Decisions):
    #     result, proba, sents, sent_result, sent_proba = m.predict(comm)
    logging.critical('\nSVM f1-weighted outputs\n ==============')

    predictions = m.predict_svm_output(X_test_eng)
    accuracy = accuracy_score(predictions, y_test_eng)
    fscore = f1_score(predictions, y_test_eng, average='micro')
    logging.critical('\nEnglish w/ combined model\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test_eng))
    logging.critical(confusion_matrix(predictions, y_test_eng))

    predictions = m.predict_svm_output(X_test_fre)
    accuracy = accuracy_score(predictions, y_test_fre)
    fscore = f1_score(predictions, y_test_fre, average='micro')
    logging.critical('\nFrench w/ combined model\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test_fre))
    logging.critical(confusion_matrix(predictions, y_test_fre))

    predictions = m.predict_svm_output(X_test_all)
    accuracy = accuracy_score(predictions, y_test_all)
    fscore = f1_score(predictions, y_test_all, average='micro')
    logging.critical('\nAll cases w/ combined model\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test_all))
    logging.critical(confusion_matrix(predictions, y_test_all))


    logging.critical('\nSVM f1-weighted decisions\n ==============')

    predictions = m.predict_svm_decision(X_test_eng)
    accuracy = accuracy_score(predictions, y_test_eng)
    fscore = f1_score(predictions, y_test_eng, average='micro')
    logging.critical('\nEnglish w/ combined model\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test_eng))
    logging.critical(confusion_matrix(predictions, y_test_eng))

    predictions = m.predict_svm_decision(X_test_fre)
    accuracy = accuracy_score(predictions, y_test_fre)
    fscore = f1_score(predictions, y_test_fre, average='micro')
    logging.critical('\nFrench w/ combined model\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test_fre))
    logging.critical(confusion_matrix(predictions, y_test_fre))

    predictions = m.predict_svm_decision(X_test_all)
    accuracy = accuracy_score(predictions, y_test_all)
    fscore = f1_score(predictions, y_test_all, average='micro')
    logging.critical('\nAll cases w/ combined model\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test_all))
    logging.critical(confusion_matrix(predictions, y_test_all))


def predict_all_train_vected(X_train_eng, X_test_eng, y_train_eng, y_test_eng, X_train_fre, X_test_fre, y_train_fre, y_test_fre, load_model=False):
    vect = TfidfVectorizer()
    clf_eng = LinearSVC()
    clf_fre = LinearSVC()
    X_train_all = pd.concat([X_train_eng, X_train_fre])
    X_test_all = pd.concat([X_test_eng, X_test_fre])
    y_train_all = pd.concat([y_train_eng, y_train_fre])
    y_test_all = pd.concat([y_test_eng, y_test_fre])

    # if load_model:
    #     m.clf = joblib.load(os.path.join(DIRECTORY, 'models/', m.name+'.joblib'))
    # else:
    #     m.train(X_train_eng, X_test_eng, y_train_eng, y_test_eng,
    #             X_train_fre, X_test_fre, y_train_fre, y_test_fre,
    #             X_train_all, y_train_all)
    logging.critical('\nUniversal vectorizer cross-language\n ==============')

    vect.fit(X_train_all)
    eng_vect = vect.transform(X_train_eng)
    fre_vect = vect.transform(X_train_fre)
    clf_eng.fit(eng_vect, y_train_eng)
    clf_fre.fit(fre_vect, y_train_fre)
    eng_vect = vect.transform(X_test_eng)
    fre_vect = vect.transform(X_test_fre)
    eng_pred = clf_eng.predict(eng_vect)
    fre_pred = clf_fre.predict(fre_vects)
    fscore_eng = f1_score(eng_pred, y_test_eng, average='micro')
    fscore_fre = f1_score(fre_pred, y_test_fre, average='micro')

    logging.critical('\nEnglish w/ universal vectorizer \n')
    logging.critical('fscore: ' + str(fscore_eng))
    logging.critical(classification_report(eng_pred, y_test_eng))
    logging.critical(confusion_matrix(eng_pred, y_test_eng))

    logging.critical('\French w/ universal vectorizer \n')
    logging.critical('fscore: ' + str(fscore_fre))
    logging.critical(classification_report(fre_pred, y_test_fre))
    logging.critical(confusion_matrix(fre_pred, y_test_fre))

    def predict_svm_output(x):
        vec = vect.transform(x)
        pred_en = fscore_eng * clf_eng.decision_function(vec)
        pred_fr = fscore_fre * clf_fre.decision_function(vec)
        pred = pred_en + pred_fr
        return np.where(pred > 0, 1, 0)

    def predict_svm_decision(x):
        # SVM decision
        vec = vect.transform(x)

        pred_en = clf_eng.predict(vec)
        pred_fr = clf_fre.predict(vec)
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

    # for comm in session.query(Decisions):
    #     result, proba, sents, sent_result, sent_proba = m.predict(comm)
    logging.critical('\nSVM f1-weighted outputs\n ==============')

    predictions = predict_svm_output(X_test_eng)
    accuracy = accuracy_score(predictions, y_test_eng)
    fscore = f1_score(predictions, y_test_eng, average='micro')
    logging.critical('\nEnglish w/ combined model\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test_eng))
    logging.critical(confusion_matrix(predictions, y_test_eng))

    predictions = predict_svm_output(X_test_fre)
    accuracy = accuracy_score(predictions, y_test_fre)
    fscore = f1_score(predictions, y_test_fre, average='micro')
    logging.critical('\nFrench w/ combined model\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test_fre))
    logging.critical(confusion_matrix(predictions, y_test_fre))

    predictions = predict_svm_output(X_test_all)
    accuracy = accuracy_score(predictions, y_test_all)
    fscore = f1_score(predictions, y_test_all, average='micro')
    logging.critical('\nAll cases w/ combined model\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test_all))
    logging.critical(confusion_matrix(predictions, y_test_all))


    logging.critical('\nSVM f1-weighted decisions\n ==============')

    predictions = predict_svm_decision(X_test_eng)
    accuracy = accuracy_score(predictions, y_test_eng)
    fscore = f1_score(predictions, y_test_eng, average='micro')
    logging.critical('\nEnglish w/ combined model\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test_eng))
    logging.critical(confusion_matrix(predictions, y_test_eng))

    predictions = predict_svm_decision(X_test_fre)
    accuracy = accuracy_score(predictions, y_test_fre)
    fscore = f1_score(predictions, y_test_fre, average='micro')
    logging.critical('\nFrench w/ combined model\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test_fre))
    logging.critical(confusion_matrix(predictions, y_test_fre))

    predictions = predict_svm_decision(X_test_all)
    accuracy = accuracy_score(predictions, y_test_all)
    fscore = f1_score(predictions, y_test_all, average='micro')
    logging.critical('\nAll cases w/ combined model\n ==============')
    logging.critical('accuracy: ' + str(accuracy))
    logging.critical('fscore: ' + str(fscore))
    logging.critical(classification_report(predictions, y_test_all))
    logging.critical(confusion_matrix(predictions, y_test_all))



if __name__ == '__main__':

    logging.basicConfig(filename='master_wvcom.log', level=logging.CRITICAL)
    logging.critical('\n\n\n\n\n\n\n\n ==============')

    em = W2VModel('en_w2v_0525.model')
    predict_en(em)
    fm = W2VModel('fr_w2v_0525.model')
    predict_fr(fm)
    cm = CombinedW2VModel('fr_w2v_0525.model', 'en_w2v_0525.model')
    predict_all(cm)
