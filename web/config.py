class Config:
    # MySQLdb doesn't support Python 3
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:3026954@localhost/hudoc?charset=utf8'
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    TESTING = True
    DEBUG = True
