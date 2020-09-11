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

random.seed(42)


def decision_predict(m):
    m.train()
    for decision in session.query(Decisions)[:100]:
        arts = [art for art in decision.article.split(';') if art.isnumeric() or re.fullmatch(r'P[0-9]*-[0-9]*$', art)]

        articles = []
        for art in arts:
            article = ECHRArticle.query.filter_by(number=art).first()
            if not article:
                if art.startswith('P'):
                    artname = 'Protocal %s Article %s' % art.split('-')[:2]
                else:
                    artname = 'Article %s' % art
                new_article = ECHRArticle(number=art, name=artname)
                session.add(new_article)
            articles.append(article)

        result, proba, sents, sent_result, sent_proba = m.predict(decision)
        old = session.query(Prediction).filter_by(modelname=m.name, appno=decision.appno, pred_type='DECISIONS').first()
        if not old:
            pred = Prediction(result=result, proba=proba, sents=sents, sent_result=json.dumps(sent_result),
                              sent_proba=json.dumps(sent_proba), modelname=m.name,
                              appno=decision.appno, pred_type='DECISIONS', gold=m.conclusion(decision.conclusion),
                              articles=articles)
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


def predict_communicated(date, load_model=False):
    end_date = date + relativedelta(months=+1)
    dt = datetime.datetime.combine(date, datetime.datetime.min.time())
    edt = datetime.datetime.combine(end_date, datetime.datetime.min.time())

    today = datetime.date.today()
    this_month = datetime.date(today.year, today.month, 1)
    # model = session.query(Model).filter_by(modelname=m.name, date=date, pred_type='COMM').first()
    model = session.query(Model).filter_by(date=date, pred_type='COMM').first()
    if model:
        if model.date == this_month:
            for p in model.predictions:
                for art in p.articles:
                    art.predictions.remove(p)
                session.delete(p)
                session.commit()
            session.delete(model)
            session.commit()
        else:
            return

    # m = NBModel_comms()
    m = Masha_SVM()
    if load_model and os.path.exists(os.path.join(DIRECTORY, 'models/', m.name+str(date)+'.joblib')):
        m.clf = joblib.load(os.path.join(DIRECTORY, 'models/', m.name+str(date)+'.joblib'))
    else:
        m.train(date)

    # Make predictions on cases that aren't published yet
    # for comm in session.query(CommunicatedCases):

    model = Model(modelname=m.name,
                  description=m.description,
                  author=m.author,
                  date=date,
                  pred_type='COMM')

    results = []
    golds = []
    for jdg in session.query(Judgments).filter(Judgments.kpdate > dt).filter(Judgments.kpdate < edt):
        comm = session.query(CommunicatedCases).filter(CommunicatedCases.appno.in_(jdg.appno.split(';')+[jdg.appno])).first()
        if not comm:
            continue

        if comm.article:
            arts = [art for art in comm.article.split(';') if art.isnumeric() or re.fullmatch(r'P[0-9]*-[0-9]*$', art)]
        else:
            arts = []

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
            pred = Prediction(gold=gold, result=result, proba=proba, sents=json.dumps(sents), sent_result=json.dumps(sent_result),
                              sent_proba=json.dumps(sent_proba), kpdate=comm.kpdate, jdgdate=jdgdate,
                              appno=comm.appno, pred_type='COMM', judgment_id=judgment_id)

            for art in arts:
                article = session.query(ECHRArticle).filter_by(number=art).first()
                if not article:
                    if art.startswith('P'):
                        print(art)
                        artname = 'Protocol %s Article %s' % (art.split('-')[0][1:], art.split('-')[1])
                    else:
                        artname = 'Article %s' % art
                    article = ECHRArticle(number=art, name=artname)
                    session.add(article)
                pred.articles.append(article)

            results.append(result)
            golds.append(gold)
            session.add(pred)
            model.predictions.append(pred)
            session.commit()

    # for comm in session.query(CommunicatedCases).filter(CommunicatedCases.kpdate < edt):
    #     if comm.article:
    #         arts = [art for art in comm.article.split(';') if art.isnumeric() or re.fullmatch(r'P[0-9]*-[0-9]*$', art)]
    #     else:
    #         arts = []

    #     result, proba, sents, sent_result, sent_proba = m.predict(comm)
    #     old = session.query(Prediction).filter_by(modelname=m.name, appno=comm.appno, pred_type='COMM').first()
    #     if not old:
    #         jdg = session.query(Judgments).filter(Judgments.kpdate > dt).filter(Judgments.kpdate < edt)\
    #                                       .filter(or_(Judgments.appno == comm.appno,
    #                                                   Judgments.appno.like("{};%".format(comm.appno)),
    #                                                   Judgments.appno.like("%;{}".format(comm.appno)),
    #                                                   Judgments.appno.like("%;{};%".format(comm.appno)))).first()
    #         judgment_id = jdg.id if jdg else None
    #         jdgdate = jdg.kpdate if jdg else None
    #         gold = m.conclusion(jdg.conclusion) if jdg else None
    #         pred = Prediction(gold=gold, result=result, proba=proba, sents=json.dumps(sents), sent_result=json.dumps(sent_result),
    #                           sent_proba=json.dumps(sent_proba), kpdate=comm.kpdate, jdgdate=jdgdate,
    #                           appno=comm.appno, pred_type='COMM', judgment_id=judgment_id)

    #         for art in arts:
    #             article = session.query(ECHRArticle).filter_by(number=art).first()
    #             if not article:
    #                 if art.startswith('P'):
    #                     print(art)
    #                     artname = 'Protocol %s Article %s' % (art.split('-')[0][1:], art.split('-')[1])
    #                 else:
    #                     artname = 'Article %s' % art
    #                 article = ECHRArticle(number=art, name=artname)
    #                 session.add(article)
    #             pred.articles.append(article)

    #         session.add(pred)
    #         model.predictions.append(pred)
    #         session.commit()

    # Evaluation, further report saved at local
    if golds:
        accuracy = accuracy_score(golds, results)
        fscore = f1_score(golds, results, average='macro')
        # logging.warning(classification_report(golds, results))
        # logging.warning(confusion_matrix(golds, results))
        if not os.path.exists(os.path.join(DIRECTORY, 'models/')):
            os.makedirs(os.path.join(DIRECTORY, 'models/'))
        if not os.path.exists(os.path.join(DIRECTORY, 'models/', m.name+str(date)+'.joblib')):
            joblib.dump(m.clf, os.path.join(DIRECTORY, 'models/', m.name+str(date)+'.joblib'))

        model.accuracy = float(accuracy)
        model.fscore = float(fscore)
        session.add(model)
        session.commit()

    # today = datetime.date.today()
    # if date.date() >= datetime.date(today.year, today.month, 1) - relativedelta(months=1):
    #     for comm in session.query(CommunicatedCases).filter(~exists().where(Prediction.appno == CommunicatedCases.appno)):
    #         if comm.article:
    #             arts = [art for art in comm.article.split(';') if art.isnumeric() or re.fullmatch(r'P[0-9]*-[0-9]*$', art)]
    #         else:
    #             arts = []

    #         result, proba, sents, sent_result, sent_proba = m.predict(comm)
    #         old = session.query(Prediction).filter_by(modelname=m.name, appno=comm.appno, pred_type='COMM').first()
    #         if not old:
    #             jdg = session.query(Judgments).filter(or_(Judgments.appno == comm.appno,
    #                                                       Judgments.appno.like("{};%".format(comm.appno)),
    #                                                       Judgments.appno.like("%;{}".format(comm.appno)),
    #                                                       Judgments.appno.like("%;{};%".format(comm.appno)))).first()
    #             judgment_id = jdg.id if jdg else None
    #             jdgdate = jdg.kpdate if jdg else None
    #             gold = m.conclusion(jdg.conclusion) if jdg else None
    #             pred = Prediction(gold=gold, result=result, proba=proba, sents=json.dumps(sents), sent_result=json.dumps(sent_result),
    #                               sent_proba=json.dumps(sent_proba), kpdate=comm.kpdate, jdgdate=jdgdate,
    #                               appno=comm.appno, pred_type='COMM', judgment_id=judgment_id)

    #             for art in arts:
    #                 article = session.query(ECHRArticle).filter_by(number=art).first()
    #                 if not article:
    #                     if art.startswith('P'):
    #                         print(art)
    #                         artname = 'Protocol %s Article %s' % (art.split('-')[0][1:], art.split('-')[1])
    #                     else:
    #                         artname = 'Article %s' % art
    #                     article = ECHRArticle(number=art, name=artname)
    #                     session.add(article)
    #                 pred.articles.append(article)

    #             session.add(pred)
    #             model.predictions.append(pred)
    #             session.commit()


def evaluate():
    today = datetime.date.today()
    end = datetime.date(today.year, today.month, 1)
    last_half_year = end + relativedelta(months=-3)
    lhy = True
    last_year = end + relativedelta(months=-12)
    ly = True
    correct = 0
    length = 0
    for pred in session.query(Prediction).filter(Prediction.gold != None).filter(Prediction.jdgdate < end).order_by(-Prediction.jdgdate):
        if pred.jdgdate < last_half_year and lhy:
            lhy = False
            acc_lhy = correct / float(length) if length != 0 else 0
        if pred.jdgdate < last_year and ly:
            ly = False
            acc_ly = correct / float(length) if length != 0 else 0
        length += 1
        if pred.gold == pred.result:
            correct += 1

    correct = 0
    length = 0
    for pred in session.query(Prediction).filter(Prediction.gold != None).order_by(-Prediction.jdgdate):
        length += 1
        if pred.gold == pred.result:
            correct += 1

    overall = correct / float(length)
    e = Evaluation(overall=overall, last_year=acc_ly, last_half_year=acc_lhy)
    session.add(e)
    session.commit()


def main():
    today = datetime.date.today()
    end = datetime.date(today.year, today.month, 1)
    with Pool(32) as p:
        for i in p.imap(predict_communicated, rrule(MONTHLY, dtstart=datetime.date(2017, 1, 1), until=end)):
            print(i)
    evaluate()

if __name__ == '__main__':
    # jm = NBModel_judgments()
    # predict(jm, pred_type='JUDGMENTS')
    # cm = NBModel_comms()
    # predict_communicated(datetime.date(2018, 2, 1))
    main()
    # evaluate()

