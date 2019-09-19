import json
import datetime
import random
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.config import Config
from db.database import CommunicatedCases, Decisions, Judgments, Prediction, Model

MODEL_NAME = 'random'
AUTHOR = 'Xu Xiao'
DATE = datetime.datetime.now()


engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()

#0 - admissible/partly admissible
#1 - no
def admissibility_anal_simple(desc):
    if not desc:
        return 1
    if 'Admissible' in desc or 'Partly admissible' in desc:
        return 0
    else:
        return 1


def decision_predict():
    tp, fp, tn, fn = 0, 0, 0, 0
    for decision in session.query(Decisions):
        conclusion = admissibility_anal_simple(decision.conclusion)
        resn = random.random()
        sent_result = []
        sent_proba = []
        for sent in json.loads(decision.sents):
            rand = random.random()
            if rand > 0.5:
                sent_result.append(1)
                sent_proba.append(rand)
            else:
                sent_result.append(0)
                sent_proba.append(1-rand)

        res = 1 if resn > 0.5 else 0
        if resn < 0.5:
            resn = 1 - resn

        if res == 0:
            if res == conclusion:
                tp += 1
            else:
                fp += 1
        else:
            if res == conclusion:
                tn += 1
            else:
                fn += 1

        pred = Prediction(result=res, proba=resn, sent_result=json.dumps(sent_result),
                          sent_proba=json.dumps(sent_proba), modelname=MODEL_NAME,
                          appno=decision.appno, pred_type='DECISIONS')
        session.add(pred)
        session.commit()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    m = Model(modelname=MODEL_NAME,
              description='A sample model using random guess',
              author=AUTHOR,
              date=DATE,
              accuracy=(tp + tn) / (tp + tn + fp + fn),
              fscore=2 * (precision * recall) / (precision + recall))
    session.add(m)
    session.commit()




def judgment_predict():
    tp, fp, tn, fn = 0, 0, 0, 0
    for decision in session.query(Decisions):

        resn = random.random()
        sent_result = []
        sent_proba = []
        for sent in json.loads(decision.sents):
            rand = random.random()
            if rand > 0.5:
                sent_result.append(1)
                sent_proba.append(rand)
            else:
                sent_result.append(0)
                sent_proba.append(1-rand)

        res = 1 if resn > 0.5 else 0
        if resn < 0.5:
            resn = 1 - resn

        pred = Prediction(result=res, proba=resn, sent_result=json.dumps(sent_result),
                          sent_proba=json.dumps(sent_proba), modelname=MODEL_NAME,
                          appno=decision.appno, pred_type='DECISIONS')
        session.add(pred)
        session.commit()

        if not session.query(Judgments).filter_by(appno=decision.appno).first():
            continue
        judgment = session.query(Judgments).filter_by(appno=decision.appno).first()
        conclusion = admissibility_anal_simple(judgment.conclusion)
        if res == 0:
            if res == conclusion:
                tp += 1
            else:
                fp += 1
        else:
            if res == conclusion:
                tn += 1
            else:
                fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    m = Model(modelname=MODEL_NAME,
              description='A sample model using random guess',
              author=AUTHOR,
              date=DATE,
              accuracy=(tp + tn) / (tp + tn + fp + fn),
              fscore=2 * (precision * recall) / (precision + recall))
    session.add(m)
    session.commit()


if __name__ == '__main__':
    decision_predict()
