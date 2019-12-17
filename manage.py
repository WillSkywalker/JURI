from web import manager, app

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


if __name__ == '__main__':
    manager.run()
