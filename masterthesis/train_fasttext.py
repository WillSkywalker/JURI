from masterthesis.naive_bayes_allcase_use_facts import NBModel_judgments, NBModel, CombinedModel

import os
import re
import logging
import json
import datetime
import joblib
import random
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists
from collections import Counter

from config.config import Config
from masterthesis.db import Decisions, Judgments, Prediction, Model, ECHRArticle, CommunicatedCases_FRE, Decisions_FRE, Judgments_FRE
from masterthesis.base import BaseDecisionModel
from masterthesis.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from nltk import word_tokenize
from gensim.models import FastText

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    eng_sents = []
    for i in session.query(CommunicatedCases).all():
        eng_sents.append(word_tokenize(i.text))
    for i in session.query(Decisions).all():
        eng_sents.append(word_tokenize(i.text))
    for i in session.query(Judgments).all():
        eng_sents.append(word_tokenize(i.text))
    eng_model = FastText(size=100, window=5, min_count=1, sentences=eng_sents, workers=16)
    
    
