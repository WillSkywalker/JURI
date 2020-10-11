from model.random_guess import RandomModel
from model.naive_bayes import NBModel_judgments, NBModel_comms

import os
import logging
import json
import datetime
import joblib
import numpy as np
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists

from config.config import Config
from db.database import CommunicatedCases, Decisions, Judgments, Prediction, Model, Press, WeeklyReport
from model.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()

def conclusion(desc):
    '''you may override this with your own implementation'''
    if not desc:
        return 1
    if 'Violation' in desc or 'Partly admissible' in desc:
        return 0
    else:
        return 1


def weekly_report(modelname):
    for report in session.query(Press).all():
        generate_weekly_report(report, modelname)


def generate_weekly_report(press, modelname):
    appnos = json.loads(press.appnos)
    # model = session.query(Model).filter_by(modelname=modelname, pred_type='COMM').first()
    predictions = []
    for appno in appnos:
        print("%{}%".format(appno))
        prediction = session.query(Prediction).filter(or_(
            Prediction.appno == appno,
            Prediction.appno.like("%;{};%".format(appno)),
            Prediction.appno.like("%;{}".format(appno)),
            Prediction.appno.like("{};%".format(appno)))).filter_by(modelname=modelname, pred_type='COMM').first()
        if prediction and prediction not in predictions:
            predictions.append(prediction)
    if not predictions:
        return
    predictions = list(set(predictions))
    new_appnos = [x.appno for x in predictions]
    print(press.docname, new_appnos)
    golds = []
    res = []
    for p in predictions:
        print(p.appno)
        jdg = session.query(Judgments).filter_by(id=p.judgment_id).first()
        if jdg:
            golds.append(conclusion(jdg.conclusion))
            res.append(p.result)

    print('=='*100)
    print(golds, res)
    acc = float(accuracy_score(golds, res))
    if np.isnan(acc):
        acc = 0
    old = session.query(WeeklyReport).filter_by(modelname=modelname, press_id=press.id).first()
    print('=='*10)
    print(predictions)
    print('=='*10)
    if not old:
        report = WeeklyReport(
            modelname=modelname,
            docname=press.docname,
            date=press.kpdate.date(),
            accuracy=acc,
            press_id=press.id,
            appnos=json.dumps(new_appnos),
            results=json.dumps(golds),
            preds=predictions
            )
        print(acc, report.preds)
        session.add(report)
        session.commit()


# def generate_weekly_report(press, modelname):
#     appnos = json.loads(press.appnos)
#     # model = session.query(Model).filter_by(modelname=modelname, pred_type='COMM').first()
#     predictions = []
#     for appno in appnos:
#         print("%{}%".format(appno))
#         prediction = Prediction.query.filter(Prediction.appno.like("%{}%".format(appno))).filter_by(modelname=modelname, pred_type='COMM').first()
#         if prediction and prediction not in predictions:
#             predictions.append(prediction)
#     if not predictions:
#         return
#     predictions = list(set(predictions))
#     new_appnos = [x.appno for x in predictions]
#     print(press.docname, new_appnos)
#     golds = []
#     res = []
#     for x in predictions:
#         print(x.appno)
#         jdg = Judgments.query.filter_by(appno=x.appno).first()
#         if jdg:
#             golds.append(conclusion(jdg.conclusion))
#             res.append(x.result)

#     # print('=='*10)
#     # print(golds, results)
#     acc = float(accuracy_score(golds, res))
#     if not acc:
#         acc = 0
#     old = WeeklyReport.query.filter_by(modelname=modelname, press_id=press.id).first()
#     print('=='*10)
#     print(predictions)
#     print('=='*10)
#     if not old:
#         report = WeeklyReport(
#             modelname=modelname,
#             date=datetime.datetime.now().date(),
#             accuracy=acc,
#             press_id=press.id,
#             appnos=json.dumps(new_appnos),
#             results=json.dumps(golds),
#             preds=predictions
#             )
#         print(report.preds)
#         db.session.add(report)
#         db.session.commit()


if __name__ == '__main__':
    # with app.app_context():
    weekly_report('Balanced Naive Bayes all cases v3'+str(datetime.datetime.now())

