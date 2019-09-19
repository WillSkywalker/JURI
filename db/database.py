from sqlalchemy import create_engine, MetaData, Table, Column, ForeignKey, Integer
from sqlalchemy import String, Unicode, Text, UnicodeText, Boolean, Float, Date
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.inspection import inspect
from sqlalchemy.ext.automap import automap_base
from config.config import Config


metadata = MetaData()

DeclarativeBase = declarative_base(metadata=metadata)


class Prediction(DeclarativeBase):
    __tablename__ = 'Prediction'
    id = Column(Integer, primary_key=True)
    result = Column(Boolean())
    proba = Column(Float())
    sent_result = Column(Text(128))
    sent_proba = Column(Text(128))
    modelname = Column(Unicode(64))
    appno = Column(String(64))
    pred_type = Column(String(16))


class Model(DeclarativeBase):
    __tablename__ = 'Model'
    id = Column(Integer, primary_key=True)
    modelname = Column(Unicode(64))
    description = Column(Unicode(128))
    author = Column(Unicode(64))
    date = Column(Date())
    accuracy = Column(Float())
    fscore = Column(Float())
    article = Column(String(64))

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=True)
metadata.reflect(engine)
Base = automap_base(metadata=metadata)
Base.prepare(engine, reflect=True)


CommunicatedCases = getattr(Base.classes, 'CommunicatedCases')
Decisions = getattr(Base.classes, 'Decisions')
Judgments = getattr(Base.classes, 'Judgments')
