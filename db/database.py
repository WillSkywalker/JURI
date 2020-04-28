from sqlalchemy import create_engine, MetaData, Table, Column, ForeignKey, Integer
from sqlalchemy import String, Unicode, Text, UnicodeText, Boolean, Float, Date, Index
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.inspection import inspect
from sqlalchemy.ext.automap import automap_base
from config.config import Config


metadata = MetaData()

DeclarativeBase = declarative_base(metadata=metadata)


articlecases = Table('articlecases', DeclarativeBase.metadata,
                     Column('prediction_id', Integer, ForeignKey('Prediction.id')),
                     Column('article_id', Integer, ForeignKey('ECHRArticle.id')))


class Prediction(DeclarativeBase):
    __tablename__ = 'Prediction'
    __table_args__ = (
        Index("idx_appno", "appno", mysql_length=15),
    )
    id = Column(Integer, primary_key=True)
    kpdate = Column(Date())
    jdgdate = Column(Date())
    gold = Column(Integer())
    result = Column(Integer())
    proba = Column(Float())
    sents = Column(Text(4294000000))
    sent_result = Column(Text(4294000000))
    sent_proba = Column(Text(4294000000))
    modelname = Column(Unicode(64), index=True)
    appno = Column(Text(4294000000))
    pred_type = Column(String(16))
    weeklyreport_id = Column(Integer, ForeignKey('WeeklyReport.id'))
    judgment_id = Column(Integer(), index=True)
    articles = relationship('ECHRArticle', secondary=articlecases, backref=backref('predictions', lazy='dynamic'), lazy='dynamic', passive_deletes=True)


class Model(DeclarativeBase):
    __tablename__ = 'Model'
    id = Column(Integer, primary_key=True)
    pred_type = Column(String(16))
    modelname = Column(Unicode(64))
    description = Column(Unicode(128))
    author = Column(Unicode(64))
    date = Column(Date())
    accuracy = Column(Float())
    fscore = Column(Float())
    article = Column(String(64))


class WeeklyReport(DeclarativeBase):
    __tablename__ = 'WeeklyReport'
    id = Column(Integer, primary_key=True)
    modelname = Column(Unicode(64))
    docname = Column(Unicode(64))
    date = Column(Date())
    accuracy = Column(Float())
    press_id = Column(Integer(), index=True)
    appnos = Column(Text(4294000000))
    results = Column(Text(4294000000))
    preds = relationship("Prediction", order_by="Prediction.kpdate")


class ECHRArticle(DeclarativeBase):
    __tablename__ = 'ECHRArticle'
    id = Column(Integer, primary_key=True)
    number = Column(String(16), index=True)
    name = Column(String(64))
    description = Column(Unicode(128))


engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=True)
metadata.reflect(engine)
Base = automap_base(metadata=metadata)
Base.prepare(engine, reflect=True)


CommunicatedCases = getattr(Base.classes, 'CommunicatedCases')
Decisions = getattr(Base.classes, 'Decisions')
Judgments = getattr(Base.classes, 'Judgments')
Press = getattr(Base.classes, 'Press')
