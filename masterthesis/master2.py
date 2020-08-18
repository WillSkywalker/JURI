# from masterthesis.naive_bayes_allcase_use_facts import NBModel, CombinedModel
from masterthesis.aligned_mikolov import W2VModel, CombinedW2VModel
# from masterthesis.aligned_muse import W2VModel, CombinedW2VModel

from masterthesis.plot import plot_learning_curve

import os
import re
import logging
import json
import datetime
import joblib
import random
import pandas as pd
import numpy as np
import argparse
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists
from collections import Counter

from config.config import Config
from masterthesis.db import CommunicatedCases, Decisions, Judgments, Prediction, Model, ECHRArticle, CommunicatedCases_FRE, Decisions_FRE, Judgments_FRE
from masterthesis.base import BaseDecisionModel
from masterthesis.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
random.seed(42)


class Experiment2:

    def __init__(self, eng_embedding, fre_embedding):
        self.eng_embedding = eng_embedding
        self.fre_embedding = fre_embedding

        self.em = W2VModel(eng_embedding)
        self.fm = W2VModel(fre_embedding)
        self.cm = CombinedW2VModel(fre_embedding, eng_embedding)

        self.used_appnos = set([])
        self.appno_train = []
        self.appno_test = []
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        self.name = 'master2-'+str(datetime.datetime.now())
        self.logger = open(self.name+'.log', 'w')

    def log(self, message):
        self.logger.write(str(message))
        self.logger.write('\n')

    def predict_en(self, load_model=False):

        # start with all English decisions, possibly with communications
        decs = session.query(Decisions).filter(exists().where(Judgments.appno == Decisions.appno)).all()
        new_appnos = []
        new_decs = []
        for d in decs:
            print('OOOOO:', d.appno)
            # text = d.text.split('\n')
            c = session.query(CommunicatedCases).filter_by(appno=d.appno).first()
            text = d.text + '\n' + c.text if c else d.text
            new_decs.append(text)
            new_appnos.append(d.appno)
            self.used_appnos.add(d.appno)

        # then add possible English communications
        decs = session.query(CommunicatedCases).filter(exists().where(Judgments.appno == CommunicatedCases.appno))\
            .filter(~CommunicatedCases.appno.in_(self.used_appnos)).all()
        for d in decs:
            print('OOOOO:', d.appno)
            # text = d.text.split('\n')
            text = d.text
            new_decs.append(text)
            new_appnos.append(d.appno)
            self.used_appnos.add(d.appno)

        # all conclusions (strings)
        results = []
        for a in new_appnos:
            j = session.query(Judgments).filter(Judgments.appno == a).with_entities(Judgments.conclusion).first()
            if j:
                results.append(self.em.conclusion_simple(j.conclusion))
            else:
                results.append(1)

        violation_num = max(0, Counter(results)[0] - Counter(results)[1])
        for comm in random.sample(session.query(Decisions).filter(~exists().where(Judgments.appno == Decisions.appno)).all(), violation_num):
            # if i >= violation_num:
            #     break
            new_appnos.append(comm.appno)
            new_decs.append(comm.text)
            # j = session.query(Judgments).filter_by(appno=comm.appno).with_entities(Judgments.conclusion).first()
            results.append(1)

        appno_train, appno_test, X_train, X_test, y_train, y_test = train_test_split(new_appnos, new_decs, results)
        self.appno_train.extend(appno_train)
        self.appno_test.extend(appno_test)
        self.X_train.extend(X_train)
        self.X_test.extend(X_test)
        self.y_train.extend(y_train)
        self.y_test.extend(y_test)
        if load_model:
            self.em.clf = joblib.load(os.path.join(DIRECTORY, 'models/', 'en_'+self.em.name+'.joblib'))
        else:
            self.em.train(X_train, y_train)

        cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
        cv_scores = cross_validate(self.fm.clf, new_appnos, new_decs, scoring=['accuracy', 'f1'], cv=cv)
        plot = plot_learning_curve(self.fm.clf, 'Learning Curves', new_decs, results)
        plot.savefig(self.name+'_english'+'.png')

        # for comm in session.query(Decisions):
        #     result, proba, sents, sent_result, sent_proba = m.predict(comm)

        predictions = self.em.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)
        fscore = f1_score(predictions, y_test, average='micro')
        self.log('\nEnglish\n ==============')
        self.log('accuracy: ' + str(accuracy))
        self.log('fscore: ' + str(fscore))

        self.log('cv_accuracy: ' + str(cv_scores['test_accuracy']))
        self.log('cv_fscore: ' + str(cv_scores['test_f1']))

        self.em.fscore = fscore
        self.log(classification_report(predictions, y_test))
        self.log(confusion_matrix(predictions, y_test))
        if not os.path.exists(os.path.join(DIRECTORY, 'models/')):
            os.makedirs(os.path.join(DIRECTORY, 'models/'))
        if not load_model:
            joblib.dump(self.em.clf, os.path.join(DIRECTORY, 'models/', 'en_'+self.em.name+'.joblib'))
        return X_train, X_test, y_train, y_test


    def predict_fr(self, load_model=False):

        # add all French decisions, possibly with communications
        decs = session.query(Decisions_FRE).filter(exists().where(Judgments_FRE.appno == Decisions_FRE.appno))\
            .filter(~Decisions_FRE.appno.in_(self.used_appnos)).all()
        new_appnos = []
        new_decs = []
        for d in decs:
            print('OOOOO:', d.appno)
            # text = d.text.split('\n')
            c = session.query(CommunicatedCases_FRE).filter_by(appno=d.appno).first()
            text = d.text + '\n' + c.text if c else d.text
            new_decs.append(text)
            new_appnos.append(d.appno)
            self.used_appnos.add(d.appno)

        # then add possible French communications
        decs = session.query(CommunicatedCases_FRE).filter(exists().where(Judgments_FRE.appno == CommunicatedCases_FRE.appno))\
            .filter(~CommunicatedCases_FRE.appno.in_(self.used_appnos)).all()
        for d in decs:
            print('OOOOO:', d.appno)
            # text = d.text.split('\n')
            text = d.text
            new_decs.append(text)
            new_appnos.append(d.appno)
            self.used_appnos.add(d.appno)

        # all conclusions (strings)
        results = []
        for a in new_appnos:
            j = session.query(Judgments_FRE).filter(Judgments_FRE.appno == a).with_entities(Judgments_FRE.conclusion).first()
            if j:
                results.append(self.fm.conclusion_fr(j.conclusion))
            else:
                results.append(1)

        violation_num = max(0, Counter(results)[0] - Counter(results)[1])
        for comm in random.sample(session.query(Decisions_FRE).filter(~exists().where(Judgments_FRE.appno == Decisions_FRE.appno)).all(), violation_num):
            # if i >= violation_num:
            #     break
            new_appnos.append(comm.appno)
            new_decs.append(comm.text)
            # j = session.query(Judgments).filter_by(appno=comm.appno).with_entities(Judgments.conclusion).first()
            results.append(1)

        appno_train, appno_test, X_train, X_test, y_train, y_test = train_test_split(new_appnos, new_decs, results)
        self.appno_train.extend(appno_train)
        self.appno_test.extend(appno_test)
        self.X_train.extend(X_train)
        self.X_test.extend(X_test)
        self.y_train.extend(y_train)
        self.y_test.extend(y_test)
        if load_model:
            self.fm.clf = joblib.load(os.path.join(DIRECTORY, 'models/', 'fr_'+self.fm.name+'.joblib'))
        else:
            self.fm.train(X_train, y_train)

        # for comm in session.query(Decisions):
        #     result, proba, sents, sent_result, sent_proba = m.predict(comm)
        cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
        cv_scores = cross_validate(self.fm.clf, new_appnos, new_decs, scoring=['accuracy', 'f1'], cv=cv)
        plot = plot_learning_curve(self.fm.clf, 'Learning Curves', new_decs, results)
        plot.savefig(self.name+'_french'+'.png')

        predictions = self.fm.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)
        fscore = f1_score(predictions, y_test, average='micro')
        self.log('\nFrench\n ==============')
        self.log('accuracy: ' + str(accuracy))
        self.log('fscore: ' + str(fscore))

        self.log('cv_accuracy: ' + str(cv_scores['test_accuracy']))
        self.log('cv_fscore: ' + str(cv_scores['test_f1']))

        self.fm.fscore = fscore
        self.log(classification_report(predictions, y_test))
        self.log(confusion_matrix(predictions, y_test))
        if not os.path.exists(os.path.join(DIRECTORY, 'models/')):
            os.makedirs(os.path.join(DIRECTORY, 'models/'))
        if not load_model:
            joblib.dump(self.fm.clf, os.path.join(DIRECTORY, 'models/', 'fr_'+self.fm.name+'.joblib'))
        return X_train, X_test, y_train, y_test


    def predict_all(self, load_model=False):
        X_train = []
        X_test = []
        y_train = self.y_train
        y_test = self.y_test
        for appno in self.appno_train:
            text = ''
            eng_desc = session.query(Decisions).filter_by(appno=appno).first()
            if eng_desc:
                text += eng_desc.text
            else:
                fre_desc = session.query(Decisions_FRE).filter_by(appno=appno).first()
                if fre_desc:
                    text += fre_desc.text
            eng_comm = session.query(CommunicatedCases).filter_by(appno=appno).first()
            if eng_comm:
                text += eng_comm.text
            else:
                fre_comm = session.query(CommunicatedCases_FRE).filter_by(appno=appno).first()
                if fre_comm:
                    text += fre_comm.text
            X_train.append(text)

        for appno in self.appno_test:
            text = ''
            eng_desc = session.query(Decisions).filter_by(appno=appno).first()
            if eng_desc:
                text += eng_desc.text
            else:
                fre_desc = session.query(Decisions_FRE).filter_by(appno=appno).first()
                if fre_desc:
                    text += fre_desc.text
            eng_comm = session.query(CommunicatedCases).filter_by(appno=appno).first()
            if eng_comm:
                text += eng_comm.text
            else:
                fre_comm = session.query(CommunicatedCases_FRE).filter_by(appno=appno).first()
                if fre_comm:
                    text += fre_comm.text
            X_test.append(text)


        if load_model:
            self.cm.clf = joblib.load(os.path.join(DIRECTORY, 'models/', 'all_'+self.cm.name+'.joblib'))
        else:
            self.cm.train(X_train, y_train)

        cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
        cv_scores = cross_validate(self.fm.clf, X_train+X_test, y_train+y_test, scoring=['accuracy', 'f1'], cv=cv)
        plot = plot_learning_curve(self.fm.clf, 'Learning Curves', X_train+X_test, y_train+y_test)
        plot.savefig(self.name+'_multilingual'+'.png')
        # for comm in session.query(Decisions):
        #     result, proba, sents, sent_result, sent_proba = m.predict(comm)

        # predictions = m.predict(X_test_eng)
        # accuracy = accuracy_score(predictions, y_test_eng)
        # fscore = f1_score(predictions, y_test_eng, average='micro')
        # self.log('\nCombined on English cases\n ==============')
        # self.log('accuracy: ' + str(accuracy))
        # self.log('fscore: ' + str(fscore))
        # self.log(classification_report(predictions, y_test_eng))
        # self.log(confusion_matrix(predictions, y_test_eng))

        # predictions = m.predict(X_test_fre)
        # accuracy = accuracy_score(predictions, y_test_fre)
        # fscore = f1_score(predictions, y_test_fre, average='micro')
        # self.log('\nCombined on French cases\n ==============')
        # self.log('accuracy: ' + str(accuracy))
        # self.log('fscore: ' + str(fscore))
        # self.log(classification_report(predictions, y_test_fre))
        # self.log(confusion_matrix(predictions, y_test_fre))

        predictions = self.cm.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)
        fscore = f1_score(predictions, y_test, average='micro')
        self.log('\nAll cases\n ==============')
        self.log('accuracy: ' + str(accuracy))
        self.log('fscore: ' + str(fscore))

        self.log('cv_accuracy: ' + str(cv_scores['test_accuracy']))
        self.log('cv_fscore: ' + str(cv_scores['test_f1']))

        self.log(classification_report(predictions, y_test))
        self.log(confusion_matrix(predictions, y_test))

    def close(self):
        self.logger.close()


if __name__ == '__main__':

    # logging.basicConfig(filename='master_wvcom.log', level=self.log)
    # self.log('\n\n\n\n\n\n\n\n ==============')

    parser = argparse.ArgumentParser(description='Run models')
    parser.add_argument('eng', type=str, help='Name of English embedding')
    parser.add_argument('fre', type=str, help='Name of French embedding')

    args = vars(parser.parse_args())
    eng = args['eng']
    fre = args['fre']

    # em = W2VModel(eng)
    # X_train_eng, X_test_eng, y_train_eng, y_test_eng = predict_en(em)
    # fm = W2VModel(fre)
    # X_train_fre, X_test_fre, y_train_fre, y_test_fre = predict_fr(fm)
    # cm = CombinedW2VModel(fre, eng)
    # predict_all(cm, X_train_eng, X_test_eng, y_train_eng, y_test_eng, X_train_fre, X_test_fre, y_train_fre, y_test_fre)
    exp = Experiment2(eng, fre)
    exp.predict_en()
    exp.predict_fr()
    exp.predict_all()
    exp.close()
