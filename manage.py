import logging
import datetime

from web import manager, app
from crawler import hudoc, press
from model import run, weekly

#`import flask_debugtoolbar

#from flask_debugtoolbar_lineprofilerpanel.profile import line_profile

#@manager.command
#def profile(length=25, profile_dir='tmpp'):
#    from werkzeug.middleware.profiler import ProfilerMiddleware
#    app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[length], profile_dir=profile_dir)
#    toolbar = flask_debugtoolbar.DebugToolbarExtension(app)
#    app.run()


@manager.command
def create():
    from web import db
    db.create_all()


@manager.command
def update():
    logging.basicConfig(filename='UPDATE_%s.log' % str(datetime.date.today()), level=logging.INFO, format='%(message)s')

    # download hudoc cases
    hudoc.get_document_list('COMMUNICATEDCASES', 'ENG')
    hudoc.get_document_list('DECISIONS', 'ENG')
    hudoc.get_document_list('JUDGMENTS', 'ENG')
    hudoc.update_database(lang='ENG')
    logging.log('HUDOC updated.')

    # download weekly press releases
    press.get_document_list()
    press.update_database()
    logging.log('Press releases updated.')

    # train models
    cm = run.NBModel_comms(name='NaiveBayes_'+str(datetime.date.today()))
    run.predict_communicated(cm)
    logging.log('Model trained.')

    # weekly reports
    weekly.weekly_report('NaiveBayes_'+str(datetime.date.today()))
    logging.log('Weekly reports generated.')

if __name__ == '__main__':
    manager.run()
