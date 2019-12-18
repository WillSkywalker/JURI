#!/usr/bin/env python3
# -_- coding: utf-8 -_-

from flask import Flask, request, jsonify, render_template, abort

from flask_script import Manager
# from flask_migrate import Migrate, MigrateCommand
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from flask_moment import Moment

import json

from config.config import Config
from db.database import metadata, CommunicatedCases, Decisions, Judgments, Prediction, Model

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


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


def admissibility_anal_simple(desc):
    if not desc:
        raise NoDecisionError
    if 'Admissible' in desc or 'Partly admissible' in desc:
        return 0
    else:
        return 1


def conclusion_simple(desc):
    if not desc:
        raise NoDecisionError
    if 'Violation of Article 6' in desc or 'Violation of Art. 6' in desc or 'Violations of Art. 6' in desc:
        return 0
    else:
        return 1


@app.route('/juri/')
def index():
    return render_template('index.html')

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
        return render_template('list.html', pagination=pagination, page=page, order=order)
    else:
        return abort(404)


@app.route('/juri/latest')
def list_judg():
    mname = request.args.get('modelname')

    pred_raw = Decisions.query.order_by(-Decisions.kpdate).limit(10).all()
    resc_raw = Judgments.query.order_by(-Judgments.kpdate).limit(10).all()

    appno_pred = [item.appno for item in pred_raw]
    appno_resc = [item.appno for item in resc_raw]

    if mname:
        pred = Prediction.query.filter(Prediction.appno.in_(appno_pred)).filter_by(pred_type='DECISIONS', modelname=mname).order_by(-Prediction.id).limit(10).all()
        resc = Prediction.query.filter(Prediction.appno.in_(appno_resc)).filter_by(pred_type='JUDGMENTS', modelname=mname).order_by(-Prediction.id).limit(10).all()
    else:
        pred = Prediction.query.filter(Prediction.appno.in_(appno_pred)).filter_by(pred_type='DECISIONS').order_by(-Prediction.id).limit(10).all()
        resc = Prediction.query.filter(Prediction.appno.in_(appno_resc)).filter_by(pred_type='JUDGMENTS').order_by(-Prediction.id).limit(10).all()

    # if pred or resc:
        # print(len(list(zip(resc_raw, resc))))
    return render_template('list-judg.html', pred=zip(pred_raw, pred), resc=zip(resc_raw, resc))
    # else:
        # return abort(404)


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
        desc_pred = Prediction.query.filter_by(appno=apno, pred_type='DECISIONS').first()

    desc.res = admissibility_anal_simple(desc.conclusion)
    sents = json.loads(desc_pred.sents)

    model = Model.query.filter_by(modelname=desc_pred.modelname).first() if desc_pred else None
    modelnames = Prediction.query.filter_by(appno=apno, pred_type='DECISIONS').with_entities(Prediction.modelname).all()
    sent_result = json.loads(desc_pred.sent_result)
    sent_proba = json.loads(desc_pred.sent_proba)
    try:
        max_idx = sent_proba.index(max([p if sent_result[i] == 0 else -1 for i, p in enumerate(sent_proba)]))
        min_idx = sent_proba.index(max([p if sent_result[i] == 1 else -1 for i, p in enumerate(sent_proba)]))
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
        judg_pred = Prediction.query.filter_by(appno=apno, pred_type='JUDGMENTS').first()

    model = Model.query.filter_by(modelname=judg_pred.modelname).first() if judg_pred else None
    modelnames = Prediction.query.filter_by(appno=apno, pred_type='JUDGMENTS').with_entities(Prediction.modelname).all()
    # sent_result = json.loads(judg_pred.sent_result)
    # sent_proba = json.loads(judg_pred.sent_proba)
    sent_result = None
    sent_proba = None
    return render_template('judgment.html', d=desc, j=judg, jp=judg_pred, **locals())



@app.route('/api/case/<appno>')
@cross_origin()
def word(appno):
    appno = appno[:-2] + '/' + appno[-2:]
    apno = appno.replace('e', '/')
    return str(session.query(Decisions).filter(Decisions.appno == appno).first().text)






# if __name__ == '__main__':
#     manager.run()
