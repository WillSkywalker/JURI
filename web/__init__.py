#!/usr/bin/env python3
# -_- coding: utf-8 -_-

from flask import Flask, request, jsonify, render_template, abort

from flask_script import Manager
# from flask_migrate import Migrate, MigrateCommand
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy

import json

from config.config import Config
from db.database import metadata, CommunicatedCases, Decisions, Judgments, Prediction, Model

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


app = Flask(__name__)
app.config.from_object(Config)
manager = Manager(app)
CORS(app)
db = SQLAlchemy(metadata=metadata)
db.init_app(app)

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=True)
Session = sessionmaker(bind=engine)

# create a Session
session = Session()
CommunicatedCases.__bases__ = CommunicatedCases.__bases__ + (db.Model,)
Decisions.__bases__ = Decisions.__bases__ + (db.Model,)
Judgments.__bases__ = Judgments.__bases__ + (db.Model,)


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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/decisions')
@app.route('/decisions/<int:page>')
def list_desc(page=1):
    pagination = Decisions.query.order_by(-Decisions.kpdate).paginate(page, per_page=30, error_out=False)
    if pagination.items:
        return render_template('list.html', pagination=pagination)
    else:
        return abort(404)


@app.route('/judgments')
@app.route('/judgments/<int:page>')
def list_judg(page=1):
    pagination = Judgments.query.order_by(-Judgments.kpdate).paginate(page, per_page=30, error_out=False)
    if pagination.items:
        return render_template('list.html', pagination=pagination)
    else:
        return abort(404)


@app.route('/<appno>')
def application(appno):
    apno = appno.replace('e', '/')
    desc = Decisions.query.filter_by(appno=apno).first()
    if desc:
        desc.res = admissibility_anal_simple(desc.conclusion)
    judg = Judgments.query.filter_by(appno=apno).first()
    if judg:
        judg.res = conclusion_simple(judg.conclusion)
    if not desc and not judg:
        abort(404)
    return render_template('application.html', d=desc, j=judg)


@app.route('/api/case/<appno>')
@cross_origin()
def word(appno):
    appno = appno[:-2] + '/' + appno[-2:]
    apno = appno.replace('e', '/')
    return str(session.query(Decisions).filter(Decisions.appno == appno).first().text)






# if __name__ == '__main__':
#     manager.run()
