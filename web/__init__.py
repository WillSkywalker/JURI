#!/usr/bin/env python3
# -_- coding: utf-8 -_-

from flask import Flask, request, jsonify, render_template, abort

from flask_script import Manager
# from flask_migrate import Migrate, MigrateCommand
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from flask_moment import Moment

import json
import random
import math

from config.config import Config
from db.database import metadata, CommunicatedCases, Decisions, Judgments, Prediction, Model, Press, WeeklyReport

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists

from sklearn.metrics import accuracy_score

#from flask_debugtoolbar_lineprofilerpanel.profile import line_profile


app = Flask(__name__)
app.config.from_object(Config)
manager = Manager(app)
CORS(app)
db = SQLAlchemy(metadata=metadata)
db.init_app(app)
moment = Moment(app)

# engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=True)
# Session = sessionmaker(bind=engine)

# # create a Session
# session = Session()
CommunicatedCases.__bases__ = CommunicatedCases.__bases__ + (db.Model,)
Decisions.__bases__ = Decisions.__bases__ + (db.Model,)
Judgments.__bases__ = Judgments.__bases__ + (db.Model,)
Prediction.__bases__ = Prediction.__bases__ + (db.Model,)
Model.__bases__ = Model.__bases__ + (db.Model,)
Press.__bases__ = Press.__bases__ + (db.Model,)
WeeklyReport.__bases__ = WeeklyReport.__bases__ + (db.Model,)


def admissibility_anal_simple(desc):
    if not desc:
        raise NoDecisionError
    if 'Admissible' in desc or 'Partly admissible' in desc or 'Partly inadmissible' in desc:
        return 0
    else:
        return 1


def conclusion_simple(desc):
    if not desc:
        return 1
    if 'Violation of ' in desc or 'Violations of ' in desc:
        return 0
    else:
        return 1


@app.route('/juri/')
def index():
    mname = request.args.get('modelname')
    if not mname:
        mname = Model.query.filter_by(pred_type='JUDGMENTS').order_by(-Model.fscore).first().modelname
        cmname = Model.query.filter_by(pred_type='COMM').order_by(-Model.fscore).first().modelname

    reports = WeeklyReport.query.order_by(-WeeklyReport.date).limit(2).all()
    right = 0
    wrong = 0
    for g, p in zip(json.loads(reports[1].results), reports[1].preds):
        if g is None:
            continue
        if g == p.result:
            right += 1
        else:
            wrong += 1

    # preds_comm = Prediction.query.filter_by(pred_type='COMM', modelname=cmname).\
    #     filter(Prediction.judgment_id.isnot(None)).order_by(-Prediction.kpdate).limit(5).all()
    # preds_comm = Prediction.query.join(Judgments, Prediction.judgment_id==Judgments.id).\
    #     filter(Prediction.pred_type == 'COMM').filter(Prediction.judgment_id.isnot(None)).\
    #     order_by(-Judgments.kpdate).limit(5).all()

    # res_comm = [Judgments.query.filter_by(id=p.judgment_id).first() for p in preds_comm]
    # res_comm = Judgments.query.filter(exists().\
    #                                   where(Prediction.judgment_id == Judgments.id)).\
    #                                   order_by(-Judgments.kpdate).limit(5).all()
    res_comm = Judgments.query.filter(exists().where(Prediction.appno == Judgments.appno)).\
                                      order_by(-Judgments.kpdate).limit(5).all()


    preds_comm = [Prediction.query.filter_by(appno=j.appno).first() for j in res_comm]


    preds_judg = Prediction.query.filter_by(pred_type='COMM', modelname=cmname).\
        filter(Prediction.judgment_id == None).order_by(-Prediction.kpdate).limit(5).all()
    res_judg = [CommunicatedCases.query.filter_by(appno=p.appno).first() for p in preds_judg]


    preds = zip(preds_comm, res_comm)
    rests = zip(preds_judg, res_judg)
    return render_template('index.html', preds=preds, rests=rests, mname=mname, cmname=cmname,
        right=right, wrong=wrong, reports=reports)


# @line_profile
@app.route('/juri/list')
@app.route('/juri/list/<int:page>')
def list_desc(page=1):
    order = request.args.get('order')
    if order == 'c':
        pagination = Decisions.query.order_by(Decisions.respondent).paginate(page, per_page=30, error_out=False)
    elif order == 'ta':
        pagination = Decisions.query.order_by(Decisions.kpdate).paginate(page, per_page=30, error_out=False)
    else:
        pagination = Decisions.query.order_by(-Decisions.kpdate).paginate(page, per_page=30, error_out=False)
    if pagination.items:
        return render_template('list.html', pagination=pagination, page=page, order=order,
                               list_name='list_judg', entry_name='application_judg')
    else:
        return abort(404)

# @line_profile
@app.route('/juri/list/judg')
@app.route('/juri/list/judg/<int:page>')
def list_judg(page=1):
    order = request.args.get('order')
    if order == 'c':
        pagination = Judgments.query.filter(exists().where(Prediction.appno == Judgments.appno)).\
                                     order_by(Judgments.respondent).paginate(page, per_page=30, error_out=False)
    elif order == 'ta':
        pagination = Judgments.query.filter(exists().where(Prediction.appno == Judgments.appno)).\
                                     order_by(Judgments.kpdate).paginate(page, per_page=30, error_out=False)
    else:
        pagination = Judgments.query.filter(exists().where(Prediction.appno == Judgments.appno)).\
                                     order_by(-Judgments.kpdate).paginate(page, per_page=30, error_out=False)
    if pagination.items:
        return render_template('list.html', pagination=pagination, page=page, order=order,
                               list_name='list_judg', entry_name='application_judg')
    else:
        return abort(404)


@app.route('/juri/list/comm')
@app.route('/juri/list/comm/<int:page>')
def list_comm(page=1):
    order = request.args.get('order')
    if order == 'c':
        pagination = CommunicatedCases.query.filter(exists().where(Prediction.appno == CommunicatedCases.appno)).\
                                     order_by(CommunicatedCases.respondent).paginate(page, per_page=30, error_out=False)
    elif order == 'ta':
        pagination = CommunicatedCases.query.filter(exists().where(Prediction.appno == CommunicatedCases.appno)).\
                                     order_by(CommunicatedCases.kpdate).paginate(page, per_page=30, error_out=False)
    else:
        pagination = CommunicatedCases.query.filter(exists().where(Prediction.appno == CommunicatedCases.appno)).\
                                     order_by(-CommunicatedCases.kpdate).paginate(page, per_page=30, error_out=False)
    if pagination.items:
        return render_template('list.html', pagination=pagination, page=page, order=order,
                               list_name='list_comm', entry_name='application_comm')
    else:
        return abort(404)


@app.route('/juri/latest')
def latest():
    mname = request.args.get('modelname')

    # pred_raw = Decisions.query.order_by(-Decisions.kpdate).limit(10).all()
    resc = Judgments.query.filter(exists().\
                                      where(Prediction.appno == Judgments.appno)).\
                                      order_by(-Judgments.kpdate).limit(50).all()

    appno_resc = [item.appno for item in resc]

    if mname:
        pred = [Prediction.query.filter_by(pred_type='COMM', modelname=mname, appno=appno).first() for appno in appno_resc]
    else:
        pred = [Prediction.query.filter_by(pred_type='COMM', appno=appno).first() for appno in appno_resc]

    mistake = 0
    for i, p in enumerate(pred):
        # p.text = ' '.join(json.loads(p.sents)[:3])
        if conclusion_simple(resc[i].conclusion) == p.result:
            p.wrong = False
            resc[i].wrong = False
        else:
            p.wrong = True
            resc[i].wrong = True
            mistake += 1

        p.title = resc[i].docname
    if pred or resc:
        # print(list(zip(resc_raw, resc)))
        return render_template('latest.html', pred=pred, resc=resc, accuracy='%.1f' % (100*(1-mistake/50)))
    else:
        return abort(404)


@app.route('/juri/list_model')
@app.route('/juri/list_model/<int:page>')
def list_model(page=1):
    order = request.args.get('order')
    if order == 'acc':
        pagination = Model.query.order_by(-Model.accuracy).paginate(page, per_page=10, error_out=False)
    elif order == 'f1':
        pagination = Model.query.order_by(-Model.fscore).paginate(page, per_page=10, error_out=False)
    elif order == 'ta':
        pagination = Model.query.order_by(Model.date).paginate(page, per_page=10, error_out=False)
    else:
        pagination = Model.query.order_by(-Model.date).paginate(page, per_page=10, error_out=False)
    if pagination.items:
        return render_template('list-model.html', pagination=pagination, page=page, order=order)
    else:
        return abort(404)


# @line_profile
@app.route('/juri/app/desc/<appno>')
def application_desc(appno):
    apno = appno.replace('e', '/')
    mname = request.args.get('modelname')
    desc = Decisions.query.filter_by(appno=apno).first_or_404()

    if mname:
        desc_pred = Prediction.query.filter_by(appno=apno, pred_type='DECISIONS', modelname=mname).first()
    else:
        desc_pred = Prediction.query.filter_by(appno=apno, pred_type='DECISIONS').order_by(-Prediction.id).first()

    desc.res = admissibility_anal_simple(desc.conclusion)
    sents = json.loads(desc_pred.sents)

    model = Model.query.filter_by(modelname=desc_pred.modelname).first() if desc_pred else None
    modelnames = Prediction.query.filter_by(appno=apno, pred_type='DECISIONS').with_entities(Prediction.modelname).all()
    sent_result = json.loads(desc_pred.sent_result)
    sent_proba = json.loads(desc_pred.sent_proba)
    critical_indexes = {}
    try:
        max_idx = sent_proba.index(max([p if sent_result[i] == 0 else -1 for i, p in enumerate(sent_proba)]))
        min_idx = sent_proba.index(max([p if sent_result[i] == 1 else -1 for i, p in enumerate(sent_proba)]))
        if len(list(set(sent_result))) > 2:
            for i in list(set(sent_result) ^ set([1, 2])):
                idx = sent_proba.index(max([p if sent_result[i] == i else -1
                                            for i, p in enumerate(sent_proba)]))
                critical_indexes[idx] = '#66ccff'
    except ValueError:
        max_idx = -1
        min_idx = -1
    critical_indexes = {
        max_idx: "#d7ffd9",
        min_idx: "#ffcccb"
    }
    print(critical_indexes)
    return render_template('application.html', d=desc, sents=sents, dp=desc_pred, model=model,
                           modelnames=modelnames, mname=mname, sent_result=sent_result, sent_proba=sent_proba,
                           critical_indexes=critical_indexes)


@app.route('/juri/app/judg/<appno>')
def application_judg(appno):
    apno = appno.replace('e', '/')
    mname = request.args.get('modelname')
    desc = Decisions.query.filter_by(appno=apno).first()
    if desc:
        desc.res = admissibility_anal_simple(desc.conclusion)
        sents = json.loads(desc.sents)
    judg = Judgments.query.filter_by(appno=apno).first()
    if judg:
        judg.res = conclusion_simple(judg.conclusion)
        jsents = json.loads(judg.sents)
    if not desc and not judg:
        abort(404)
    if mname:
        judg_pred = Prediction.query.filter_by(appno=apno, pred_type='JUDGMENTS', modelname=mname).first()
    else:
        judg_pred = Prediction.query.filter_by(appno=apno, pred_type='JUDGMENTS').order_by(-Prediction.id).first()

    model = Model.query.filter_by(modelname=judg_pred.modelname).first() if judg_pred else None
    modelnames = Prediction.query.filter_by(appno=apno, pred_type='JUDGMENTS').with_entities(Prediction.modelname).all()
    sent_result = json.loads(judg_pred.sent_result)
    sent_proba = json.loads(judg_pred.sent_proba)
    # sent_result = None
    # sent_proba = None
    critical_indexes = {}
    try:
        max_prob = max([p for i, p in enumerate(sent_proba) if sent_result[i] == 0], default=None)
        max_sent = sents[sent_proba.index(max_prob)] if max_prob else None
        min_prob = max([p for i, p in enumerate(sent_proba) if sent_result[i] == 1], default=None)
        min_sent = sents[sent_proba.index(min_prob)] if min_prob else None
        max_idxes = [i for i, p in enumerate(sent_proba) if sent_result[i] == 0 and p > 0.9]
        min_idxes = [i for i, p in enumerate(sent_proba) if sent_result[i] == 1 and p > 0.9]
        if len(list(set(sent_result))) > 2:
            for i in list(set(sent_result) ^ set([1, 2])):
                idx = sent_proba.index(max([p if sent_result[i] == i else -1
                                            for i, p in enumerate(sent_proba)]))
                critical_indexes[idx] = '#66ccff'
    except ValueError:
        max_idxes = []
        min_idxes = []
    critical_indexes = {
        # max_idx: "#d7ffd9",
        # min_idx: "#ffcccb"
    }
    for idx in max_idxes:
        critical_indexes[idx] = "#d7ffd9"
    for idx in min_idxes:
        critical_indexes[idx] = "#ffcccb"
    print(critical_indexes)
    sent_num = len(sents)
    rand1 = random.randrange(sent_num//5, sent_num//2)
    rand2 = random.randrange(sent_num//2, sent_num*4//5)
    return render_template('judgment.html', d=desc, j=judg, jp=judg_pred, **locals())


@app.route('/juri/app/comm/<appno>')
def application_comm(appno):
    apno = appno.replace('e', '/')
    mname = request.args.get('modelname')
    comm = CommunicatedCases.query.filter_by(appno=apno).first()

    judg = Judgments.query.filter_by(appno=apno).first()

    if not comm:
        abort(404)
    if mname:
        judg_pred = Prediction.query.filter_by(appno=apno, pred_type='COMM', modelname=mname).first()
    else:
        judg_pred = Prediction.query.filter_by(appno=apno, pred_type='COMM').order_by(-Prediction.id).first()

    if judg and not judg_pred:
        judg_pred = Prediction.query.filter_by(judgment_id=judg.id, pred_type='COMM').order_by(-Prediction.id).first()
    if not judg and judg_pred:
        judg = Judgments.query.filter_by(id=judg_pred.judgment_id).first()

    if judg:
        judg.res = conclusion_simple(judg.conclusion)
        jsents = json.loads(judg.sents)


    model = Model.query.filter_by(modelname=judg_pred.modelname).first() if judg_pred else None
    modelnames = Prediction.query.filter_by(appno=apno, pred_type='COMM').with_entities(Prediction.modelname).all()
    sents = json.loads(judg_pred.sents)
    sent_result = json.loads(judg_pred.sent_result)
    sent_proba = json.loads(judg_pred.sent_proba)
    # sent_result = None
    # sent_proba = None
    critical_indexes = {}
    try:
        max_prob = max([p for i, p in enumerate(sent_proba) if sent_result[i] == 0], default=None)
        max_sent = sents[sent_proba.index(max_prob)] if max_prob else None
        min_prob = max([p for i, p in enumerate(sent_proba) if sent_result[i] == 1], default=None)
        min_sent = sents[sent_proba.index(min_prob)] if min_prob else None
        if judg:
            max_idxes = [i for i, p in enumerate(sent_proba) if sent_result[i] == judg.res and p > 0.5]
            min_idxes = [i for i, p in enumerate(sent_proba) if sent_result[i] != judg.res and p > 0.5]
        elif judg_pred:
            max_idxes = [i for i, p in enumerate(sent_proba) if sent_result[i] == judg_pred.result and p > 0.5]
            min_idxes = [i for i, p in enumerate(sent_proba) if sent_result[i] != judg_pred.result and p > 0.5]
        else:
            max_idxes = []
            min_idxes = []
        if len(list(set(sent_result))) > 2:
            for i in list(set(sent_result) ^ set([1, 2])):
                idx = sent_proba.index(max([p if sent_result[i] == i else -1
                                            for i, p in enumerate(sent_proba)]))
                critical_indexes[idx] = '#66ccff'
    except ValueError as e:
        print('error', e)
        max_idxes = []
        min_idxes = []
    critical_indexes = {
        # max_idx: "#d7ffd9",
        # min_idx: "#ffcccb"
    }
    for idx in max_idxes:
        if judg:
            critical_indexes[idx] = "hsl(123, 100%, {}%)".format((0.6 - math.log(sent_proba[idx]) * 0.5) * 100)
        else:
            critical_indexes[idx] = "hsl(182, 100%, {}%)".format((0.6 - math.log(sent_proba[idx]) * 0.5) * 100)
        # print(sent_proba[idx], math.log(9-sent_proba[idx])*100)
    for idx in min_idxes:
        if judg:
            critical_indexes[idx] = "hsl(1, 100%, {}%)".format((0.6 - math.log(sent_proba[idx]) * 0.5) * 100)
        else:
            critical_indexes[idx] = "hsl(60, 100%, {}%)".format((0.6 - math.log(sent_proba[idx]) * 0.5) * 100)
    print(max_idxes)
    print(critical_indexes)
    sent_num = len(sents)
    rand1 = random.randrange(sent_num//5, sent_num//2)
    rand2 = random.randrange(sent_num//2, sent_num*4//5)
    return render_template('comm.html', d=comm, j=judg, jp=judg_pred, **locals())


@app.route('/juri/reports')
@app.route('/juri/reports/<int:page>')
def list_reports(page=1):
    # order = request.args.get('order')
    pagination = WeeklyReport.query.order_by(-WeeklyReport.date).paginate(page, per_page=30, error_out=False)
    if pagination.items:
        return render_template('list-report.html', pagination=pagination, page=page,
                               list_name='list_reports', entry_name='report')
    else:
        return abort(404)


@app.route('/juri/report/<int:report_id>')
def report(report_id):
    press = Press.query.filter_by(id=report_id).first()
    report = WeeklyReport.query.filter_by(press_id=press.id).first()
    appnos = json.loads(press.appnos)
    preds = []
    # comms = []
    # judgs = []
    for appno in appnos:
        # pred = Prediction.query.filter(Prediction.appno.like("%{}%".format(appno))).first()
        pred = Prediction.query.filter_by(appno=appno, pred_type='COMM').first()
        if pred:
            comm = CommunicatedCases.query.filter_by(appno=appno).first()
            judg = Judgments.query.filter_by(id=pred.judgment_id).first()
            if judg:
                judg.res = conclusion_simple(judg.conclusion)
            preds.append((pred, comm, judg))

    print(preds)

    return render_template('report.html', press=press, report=report, preds=preds)


@app.route('/api/case/<appno>')
@cross_origin()
def word(appno):
    appno = appno[:-2] + '/' + appno[-2:]
    apno = appno.replace('e', '/')
    return str(session.query(Decisions).filter(Decisions.appno == appno).first().text)






# if __name__ == '__main__':
#     manager.run()
