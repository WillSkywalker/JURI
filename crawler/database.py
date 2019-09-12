from sqlalchemy import create_engine, MetaData, Table, Column, ForeignKey, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.inspection import inspect
from sqlalchemy.ext.automap import automap_base
from .config import Config


metadata = MetaData()
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=True)
metadata.reflect(engine)
Base = automap_base(metadata=metadata)
Base.prepare(engine, reflect=True)

# CommunicatedCases = Table('CommunicatedCases', metadata,
#     Column('id', Integer, primary_key=True), extend_existing=True)
# CommunicatedCases.__bases__ = (Base, )

# Decisions = Table('Decisions', metadata,
#     Column('id', Integer, primary_key=True), extend_existing=True)
# Decisions.__bases__ = (Base, )


# Judgments = Table('Judgments', metadata,
#     Column('id', Integer, primary_key=True,), extend_existing=True)
# Judgments.__bases__ = (Base, )

# print(Base.classes)
CommunicatedCases = getattr(Base.classes, '6_CommunicatedCases')
Decisions = getattr(Base.classes, '6_Decisions')
Judgments = getattr(Base.classes, '6_Judgments')