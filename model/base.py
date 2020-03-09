import json
import datetime
import random
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.config import Config
from db.database import CommunicatedCases, Decisions, Judgments, Prediction, Model
from model.extract_facts_judgments import extract_parts_judgments

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()


class BaseDecisionModel:
    '''Please inherit your model from BaseModel'''
    def __init__(self, name, author, description, date=datetime.datetime.now()):
        super(BaseDecisionModel, self).__init__()
        self.name = name
        self.author = author
        self.description = description
        self.date = date
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0

        decs = session.query(Decisions).all()
        self.X = [self.input_extraction(dec) for dec in decs]
        # dec.text contains raw text while dec.sents are sentences divided by nltk.
        # Make your own choice for input. We'll need sentence tokenization later on.
        self.Y = [self.conclusion(dec.conclusion) for dec in decs]

    @staticmethod
    def conclusion(desc):
        '''you may override this with your own implementation'''
        if not desc:
            return 1
        if 'Admissible' in desc or 'Partly admissible' in desc or 'Partly inadmissible' in desc:
            return 0
        else:
            return 1

    @staticmethod
    def input_extraction(dec):
        '''
        override this with your own text extraction and vectorization.
        If you want to add communicated cases, use
        session.query(CommunicatedCases).filter_by(appno=dec.appno).first().text
        '''
        return dec.text

    def train(self):
        '''
        override this with your own implementation.
        You should train your classifier here.
        '''
        # from sklearn.model_selection import train_test_split
        # self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = \
        #     train_test_split(self.X, self.Y, test_size=0.2)
        # self.classifier = SomeClassifier().train(self.Xtrain, self.Ytrain)
        pass

    def predict(self, x):
        '''
        This method gets a single input each time.
        It should return a five-tuple: (
            result: prediction,
            proba: probability,
            sents: list of sentences used as input
            sent_result: list of predictions for each sentences,
            sent_proba: list of probabilities for each sentences)
        '''
        # return self.classifier.predict(x)
        return (0, 1, ['Toshiba, Toshiba', 'Shin Jidai no Toshiba'], [0, 0], [0.8, 0.6])


class BaseJudgmentModel:
    '''Please inherit your model from BaseModel'''
    def __init__(self, name, author, description, date=datetime.datetime.now()):
        super(BaseJudgmentModel, self).__init__()
        self.name = name
        self.author = author
        self.description = description
        self.date = date
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0

        judgs = Judgments.query.all()
        judg_pred = [item.appno for item in judgs]
        decs = Decisions.query.filter(Decisions.appno.in_(judg_pred)).all()  # only cases with a judgment
        # Here all the cases are used.
        # If your model is specific on a certain article, you may use
        # judg_pred = [item.appno for item in judgs if article_num in item.conclusion]

        self.X = [self.input_extraction(dec) for dec in decs]
        # dec.text contains raw text while dec.sents are sentences divided by nltk.
        # Make your own choice for input. We'll need sentence tokenization later on.
        self.Y = [self.conclusion(dec.conclusion)]

    @staticmethod
    def conclusion(desc):
        '''you may override this with your own implementation'''
        if not desc:
            return 1
        if 'Violation' in desc or 'Partly admissible' in desc:
            return 0
        else:
            return 1

    @staticmethod
    def input_extraction(dec):
        '''
        override this with your own text extraction and vectorization.
        If you want to add communicated cases, use
        CommunicatedCases.query.filter_by(appno=dec.appno).text
        '''
        return dec.text

    def train(self):
        '''
        override this with your own implementation.
        You should train your classifier here.
        '''
        # from sklearn.model_selection import train_test_split
        # self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = \
        #     train_test_split(self.X, self.Y, test_size=0.2)
        # self.classifier = SomeClassifier().train(self.Xtrain, self.Ytrain)

    def predict(self, x):
        '''
        This method gets a single input each time.
        It should return a five-tuple: (
            result: prediction,
            proba: probability,
            sents: list of sentences used as input
            sent_result: list of predictions for each sentences,
            sent_proba: list of probabilities for each sentences)
        '''
        # return self.classifier.predict(x)
        return (0, 1, ['Toshiba, Toshiba', 'Shin Jidai no Toshiba'], [0, 0], [0.8, 0.6])


class BaseCommunicatedCasesModel:
    '''Please inherit your model from BaseModel'''
    def __init__(self, name, author, description, date=datetime.datetime.now()):
        super(BaseCommunicatedCasesModel, self).__init__()
        self.name = name
        self.author = author
        self.description = description
        self.date = date
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0

        # judgs = Judgments.query.all()
        # judg_pred = [item.appno for item in judgs]
        # decs = Decisions.query.filter(Decisions.appno.in_(judg_pred)).all()  # only cases with a judgment
        # Here all the cases are used.
        # If your model is specific on a certain article, you may use
        # judg_pred = [item.appno for item in judgs if article_num in item.conclusion]

        # self.X = [self.input_extraction(dec) for dec in decs]
        # dec.text contains raw text while dec.sents are sentences divided by nltk.
        # Make your own choice for input. We'll need sentence tokenization later on.
        # self.Y = [self.conclusion(dec.conclusion)]

    @staticmethod
    def conclusion(desc):
        '''you may override this with your own implementation'''
        if not desc:
            return 1
        if 'Violation' in desc or 'Partly admissible' in desc:
            return 0
        else:
            return 1

    @staticmethod
    def input_extraction(dec):
        '''
        override this with your own text extraction and vectorization.
        If you want to add communicated cases, use
        CommunicatedCases.query.filter_by(appno=dec.appno).text
        '''
        return CommunicatedCases.query.filter_by(appno=dec.appno).text

    def train(self):
        '''
        override this with your own implementation.
        You should train your classifier here.
        '''
        pass
        # from sklearn.model_selection import train_test_split
        # self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = \
        #     train_test_split(self.X, self.Y, test_size=0.2)
        # self.classifier = SomeClassifier().train(self.Xtrain, self.Ytrain)

    def predict(self, x):
        '''
        This method gets a single input each time.
        It should return a five-tuple: (
            result: prediction,
            proba: probability,
            sents: list of sentences used as input
            sent_result: list of predictions for each sentences,
            sent_proba: list of probabilities for each sentences)
        '''
        # return self.classifier.predict(x)
        return (0, 1, ['Toshiba, Toshiba', 'Shin Jidai no Toshiba'], [0, 0], [0.8, 0.6])

