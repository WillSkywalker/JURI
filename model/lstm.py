import os
import json
import datetime
import random
import logging
import numpy as np
from collections import Counter
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists

from config.config import Config
from db.database import CommunicatedCases, Decisions, Judgments, Prediction, Model
from model.base import BaseDecisionModel, BaseCommunicatedCasesModel
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (TimeDistributed, Dense, Embedding, Input, LSTM,
                          Bidirectional, Flatten, Masking, concatenate)
from tensorflow.keras.utils import Progbar
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import RandomUniform

from nltk import word_tokenize

from model.utils import load_embedding, create_batch
from model.extract_facts_judgments import extract_parts_judgments, JudgmentNoTextError

MODEL_NAME = 'BiLSTM-v1'
AUTHOR = 'Xu Xiao'
DESCRIPTION = 'Bi-directional LSTM model using the communication documents'
DATE = datetime.datetime.today()
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

MAX_LEN = 1000
MAX_WORDS = 10000

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()


class BiLSTM_model(BaseCommunicatedCasesModel):

    def __init__(self, name=MODEL_NAME, author=AUTHOR, description=DESCRIPTION, date=DATE):
        super(BiLSTM_model, self).__init__(name, author, description, date)
        # self.arg = arg
        embeddings = load_embedding(os.path.join(DIRECTORY, "embeddings/glove.6B.100d.txt"))
        embedding_matrix = list(embeddings.values())
        embeddings = {key: i for i, key in enumerate(embeddings.keys())}

        embeddings['UNKNOWN_TOKEN'] = len(embedding_matrix)
        embedding_matrix.append(np.random.uniform(-0.25, 0.25, len(embedding_matrix[0])))

        embedding_matrix = np.matrix(embedding_matrix)

        words_input = Input(shape=(None,), dtype='int32', name='words_input')
        words = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                          weights=[embedding_matrix])(words_input)
        ao = Bidirectional(LSTM(200, dropout=0.50, recurrent_dropout=0.25))(words)
        output = Dense(2, activation='softmax')(ao)
        model = Model(inputs=[words_input], outputs=[output])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
        model.summary()
        self.embeddings = embeddings
        self.clf = model

    @staticmethod
    def conclusion_simple(desc):
        # Mark the state of conclusion. 0 for pass and 1 for fail, adding more situation possible
        if not desc:
            return 1
        if 'Violation of Article ' in desc or 'Violation of Art. ' in desc or 'Violations of Art. ' in desc:
            return 0
        else:
            return 1

    def train(self, date):

        # comms = session.query(CommunicatedCases).all()
        dt = datetime.datetime.combine(date, datetime.datetime.min.time())
        # judgs = session.query(Judgments).filter(Judgments.kpdate < dt).all()
        comms = session.query(CommunicatedCases).filter(CommunicatedCases.kpdate < dt).filter(exists().where(Judgments.appno == CommunicatedCases.appno)).all()
        appnos = []
        comm_texts = []
        for d in comms:
            try:
                print('OOOOO:', d.appno)
                # text = d.text.split('\n')
                comm_texts.append(d.text)
                appnos.append(d.appno)
            except JudgmentNoTextError:
                logging.warning(d.appno)

        # In case you need CommunicatedCases
        # ccs = CommunicatedCases.query.filter(CommunicatedCases.appno.in_(appnos)).all()

        # all conclusions (strings)
        # results = Judgments.query.filter(Judgments.appno.in_(appnos)).with_entities(Judgments.conclusion).all()
        results = []
        new_comms = []
        for i, a in enumerate(appnos):
            j = session.query(Judgments).filter_by(appno=a).first()
            if j.kpdate > dt:
                continue
            if j:
                results.append(self.conclusion_simple(j.conclusion))
            else:
                results.append(1)
            new_comms.append(comm_texts[i])

        violation_num = Counter(results)[0] - Counter(results)[1]
        for comm in random.sample(session.query(CommunicatedCases).filter(CommunicatedCases.kpdate < dt).filter(~exists().where(Judgments.appno == CommunicatedCases.appno)).all(), violation_num):
            # if i >= violation_num:
            #     break
            new_comms.append(comm.text)
            # j = session.query(Judgments).filter_by(appno=comm.appno).with_entities(Judgments.conclusion).first()
            results.append(1)

        assert len(new_comms) == len(results)
        batches = create_batch(new_comms, results, self.embeddings)
        for batch in batches:
            tokens, labels = batch[0], batch[1]
            self.clf.train_on_batch(tokens, labels)
        # self.clf.fit(new_decisions, results)

    def predict(self, x):
        # conclusion = self.conclusion(x.conclusion)
        resn = random.random()
        sents = json.loads(x.sents)  # [:20]  # suppose we use first 20 sents

        string = create_batch([' '.join(sents)], [-1], self.embeddings)[0][0]  # for prediction
        raw_res = self.clf.predict(string)[0]
        res = int(np.argmax(raw_res))  # class
        resn = float(raw_res[res])  # proba

        sent_result = []
        sent_proba = []
        for sent in sents:
            string = create_batch([' '.join(sents)], [-1], self.embeddings)[0][0]
            raw_res = self.clf.predict(string)[0]
            sent_res = np.argmax(raw_res)  # class
            sent_resn = raw_res[sent_res]  # proba
            sent_result.append(int(sent_res))
            sent_proba.append(float(sent_resn))

        # if res == 0:
        #     if res == conclusion:
        #         self.tp += 1
        #     else:
        #         self.fp += 1
        # else:
        #     if res == conclusion:
        #         self.tn += 1
        #     else:
        #         self.fn += 1

        return res, resn, sents, sent_result, sent_proba


class BiLSTM_trim(BaseCommunicatedCasesModel):

    def __init__(self, name=MODEL_NAME, author=AUTHOR, description=DESCRIPTION, date=DATE):
        super(BiLSTM_trim, self).__init__(name, author, description, date)
        self.le = LabelEncoder()
        self.ohe = OneHotEncoder()
        self.tok = Tokenizer(num_words=MAX_WORDS)
        # tok.fit_on_texts(train_df.cutword)

        inputs = Input(name='inputs', shape=[MAX_LEN])
        layer = Embedding(MAX_WORDS+1, 128, input_length=MAX_LEN)(inputs)
        layer = LSTM(128)(layer)
        layer = Dense(128, activation="relu", name="FC1")(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(2, activation="softmax", name="FC2")(layer)
        self.clf = Model(inputs=inputs, outputs=layer)
        self.clf.summary()
        self.clf.compile(loss="sparse_categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])

    @staticmethod
    def conclusion_simple(desc):
        # Mark the state of conclusion. 0 for pass and 1 for fail, adding more situation possible
        if not desc:
            return 1
        if 'Violation of Article ' in desc or 'Violation of Art. ' in desc or 'Violations of Art. ' in desc:
            return 0
        else:
            return 1

    def train(self, date):

        # comms = session.query(CommunicatedCases).all()
        dt = datetime.datetime.combine(date, datetime.datetime.min.time())
        # judgs = session.query(Judgments).filter(Judgments.kpdate < dt).all()
        comms = session.query(CommunicatedCases).filter(CommunicatedCases.kpdate < dt).filter(exists().where(Judgments.appno == CommunicatedCases.appno)).all()
        appnos = []
        comm_texts = []
        for d in comms:
            try:
                print('OOOOO:', d.appno)
                # text = d.text.split('\n')
                comm_texts.append(d.text)
                appnos.append(d.appno)
            except JudgmentNoTextError:
                logging.warning(d.appno)

        # In case you need CommunicatedCases
        # ccs = CommunicatedCases.query.filter(CommunicatedCases.appno.in_(appnos)).all()

        # all conclusions (strings)
        # results = Judgments.query.filter(Judgments.appno.in_(appnos)).with_entities(Judgments.conclusion).all()
        results = []
        new_comms = []
        for i, a in enumerate(appnos):
            j = session.query(Judgments).filter_by(appno=a).first()
            if j.kpdate > dt:
                continue
            if j:
                results.append(self.conclusion_simple(j.conclusion))
            else:
                results.append(1)
            new_comms.append(comm_texts[i])

        violation_num = Counter(results)[0] - Counter(results)[1]
        for comm in random.sample(session.query(CommunicatedCases).filter(CommunicatedCases.kpdate < dt).filter(~exists().where(Judgments.appno == CommunicatedCases.appno)).all(), violation_num):
            # if i >= violation_num:
            #     break
            new_comms.append(comm.text)
            # j = session.query(Judgments).filter_by(appno=comm.appno).with_entities(Judgments.conclusion).first()
            results.append(1)

        assert len(new_comms) == len(results)

        self.tok.fit_on_texts(new_comms)
        train_seq = self.tok.texts_to_sequences(new_comms)
        train_seq_mat = sequence.pad_sequences(train_seq, maxlen=MAX_LEN)
        train_y = self.le.fit_transform(results).reshape(-1, 1)

        self.clf.fit(train_seq_mat, train_y, batch_size=128, epochs=10)

    def predict(self, x):
        # conclusion = self.conclusion(x.conclusion)
        resn = random.random()
        sents = json.loads(x.sents)  # [:20]  # suppose we use first 20 sents
        seq = self.tok.texts_to_sequences([' '.join(sents)])
        seq_mat = sequence.pad_sequences(seq, maxlen=MAX_LEN)
        raw_res = self.clf.predict(seq_mat)[0]
        res = int(np.argmax(raw_res))  # class
        resn = float(raw_res[res])  # proba

        sent_result = []
        sent_proba = []
        for sent in sents:
            seq = self.tok.texts_to_sequences([sent])
            seq_mat = sequence.pad_sequences(seq, maxlen=MAX_LEN)
            raw_res = self.clf.predict(seq_mat)[0]
            sent_res = np.argmax(raw_res)  # class
            sent_resn = raw_res[sent_res]  # proba
            sent_result.append(int(sent_res))
            sent_proba.append(float(sent_resn))

        # if res == 0:
        #     if res == conclusion:
        #         self.tp += 1
        #     else:
        #         self.fp += 1
        # else:
        #     if res == conclusion:
        #         self.tn += 1
        #     else:
        #         self.fn += 1

        return res, resn, sents, sent_result, sent_proba

