import os
import re
import glob
import random
import pandas as pd
import numpy as np
import unicodedata
from collections import defaultdict

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, SMOTENC, SVMSMOTE, ADASYN

from config.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.database import CommunicatedCases, Decisions, Judgments, Prediction, Model, ECHRArticle, Evaluation

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
Session = sessionmaker(bind=engine)
session = Session()

#extracting labels for communicated cases from judgements
def extract_labels(data_table):  # table_jud_with_com #extract labels from the table containing judgment for corresponding communicated cases

    #creating lists of variables for easier navigation
    #the list are the same length (length of the column in the table)
    app_conc = data_table['conclusion'].tolist()
    #example of conclusion:
    #"Violation of Article 6 - Right to a fair trial (Article 6 - Civil proceedings;Article 6-1 - Reasonable time)""
    app_names = data_table['docname'].tolist()  # title of the case (!!!only here for easier manual check of cases)
    app_numbers = data_table['appno'].tolist()  # application number
    app_articles = data_table['article'].tolist()  # articles involved in the case

    d = {}
    count = -1

    for conclusion_set in app_conc:

        count += 1

        conclusion_set = str(conclusion_set)  # in case there is some noise in the table

        # TODO make this more efficient
        # remove unnecessary ';' that will affect separation of the conclusions and replace with '&'
        conclusion_set = re.sub('(\([^\)]*);([^\)]*\))', '\\1 & \\2', conclusion_set)
        conclusion_set = re.sub('(\([^\)]*);([^\)]*\))', '\\1 & \\2', conclusion_set)
        conclusion_set = re.sub('(\([^\)]*);([^\)]*\))', '\\1 & \\2', conclusion_set)
        conclusion_set = conclusion_set.split(';')  # get separate conclusions

        articles = str(app_articles[count]).split(';')  # find articles of the same case

        violations = []
        non_violations = []
        struck_out = False

        for i in conclusion_set:  # for each conclusion in the conclusion set

            v = re.search('Violation of (Art)?(\.)?(icle)? ?([0-9\-\+P]+)', i)
            nv = re.search('No violation of (Art)?(\.)?(icle)? ?([0-9\-\+P]+)', i)
            #NOTE: Article 1 of Protocol 1 is extracted as Article 1.
            #This is not a problem as Article 1 relates to the court itself (CHECK?)
            #and is never invoked during the trial

            #v = re.search('Violation of Article 3', i) #TODO adapt to specific articles to use per article
            #nv = re.search('No violation of Article 3', i)
            struck_out = re.search('[S|s]truck out of the list', i)

            if v != None:
                label = v.group(4)
                label = re.sub('([^\-\+])(\+|\-)(.*)', '\\1', label)  # for now remove subparagraphs
                #TODO fix article 13+6
                if label not in violations:
                    violations.append(label)
                continue
            if nv != None:
                label = nv.group(4)
                label = re.sub('([^\-\+])(\+|\-)(.*)', '\\1', label)
                if label not in non_violations:
                    non_violations.append(label)

            if struck_out != None:
                struck_out = True

        both = list(set(violations+non_violations))  # all articles for this case

        #dismissed articles are the ones that were either not mentioned in the conclustion,
        #or are not the articles that may end up in violations, often related to compensation(e.g 35, 41)
        #TODO check P7
        dismissed_articles = [re.sub('([^\-\+])(\+|\-)(.*)', '\\1', i) for i in articles if (re.sub('([^\-\+])(\+|\-)(.*)', '\\1', i) not in both and re.sub('([^\-\+])(\+|\-)(.*)', '\\1', i) != 'P1')]

        #dictionary with all the labels associated with the application number of the case
        d[app_numbers[count]] = (app_names[count], conclusion_set, violations, non_violations, list(set(dismissed_articles)), struck_out)

    return d

def extract_text(data_table_com, d_labels,PATH_TO_COMM_CASES, judg=False): #extract text for every case in com_with_jud table ?? AND GET THE LABELS?
    # ?? REMOVE ?? all_cases = glob.glob('./docs/JUDGMENTS/11/new_txt/*')
    Xtrain = []
    Ytrain = []
    files = []

    d = {}

    app_names = data_table_com['docname'].tolist() #list of document names
    app_nos = data_table_com['appno'].tolist() #list of application numbers
    app_nos = [i.split(';') for i in app_nos]
    assert (len(app_names) == len(app_nos))

    for num in range(len(app_names)):  # going through every docname
        if app_names[num] not in files:  # if it's not in files already (in case there are duplicates in the table)
            #REMOVE files.append(app_names[num]) #add the file name
            found = False
            for m in d_labels.keys():  # for every app number in application numbers from the judgment list
                for i in app_nos[num]:
                    if i in m.split(';'):# if the judgement number is one of communication numbers of this case
                        files.append(app_names[num])
                        Ytrain.append(d_labels[m])  # and add it's label to the Ytrain
                        found = True
                        break
                    if found:
                        break
                if found:
                    break  # break to the next case

    assert (len(files) == len(Ytrain))

    app_names = [i+'.txt' for i in files]

    count = -1
    #commun_facts =[]
    #judgements_facts = []

    app_names = [unicodedata.normalize('NFD', app_name) for app_name in app_names]#.encode('ASCII', 'ignore')

    for app_name in app_names:  # for every app name in the table com_with_jud
        count += 1
        try:  # occasional issues with title encoding
            with open(PATH_TO_COMM_CASES + app_name, 'r') as f:
                f = f.readlines()  # read the file
                #f = [i.decode("utf-8") for i in f] #currently solved
                if f != False:
                    f = [re.sub('^.*v\. .* STATEMENT OF.*', '', i) for i in f]  # remove the tietles of the pages from getting text from pdf
                    f = [re.sub('STATEMENT OF FACTS( AND QUESTIONS)?', '', i) for i in f]  # remove the tietles of the pages from getting text from pdf
                    f = [re.sub('[A-ZĐĆ ]+ v. [A-Z ]+', '', i) for i in f]
                    f = [re.sub('^\n$', '', i) for i in f]  # remove end of the lines
                    f = [re.sub('\n', '', i) for i in f]
                    f = ' '.join(f)  # combine lines
                    f = re.sub('  +', ' ', f)
                    if judg == False:
                        m = re.search('(.+) QUESTIONS', f)
                        m2 = re.search('(.+) COMPLAINTS', f)
                        if m != None:
                            f = m.group(1)
                        elif m2 != None:
                            f = m2.group(1)
                    if len(f) > 1:
                        # application name assigned text
                        Xtrain.append(f)
                        if Ytrain[count][2] != []:  # if there is something in the list with violations the label is a violation
                            d[app_name] = (f, 1)
                        else:
                            d[app_name] = (f, 0)
        except Exception as inst:
            #print(type(inst))     # the exception instance
            print('didn\'t work', app_name)
            #pass

    return d  # dictionary, text file name with assigned text + label(from judgment table)


#communicated cases that have decisions and found inadmissible by the court based on merit.
#This does not precisely mean that there was no violation, but in practice ***the court did not
#rule that there was a violation***
##extracting labels for communicated cases from decisions (only non-violation)
def extract_text_dec(data_table_com):  # extract text for every case in com_with_dec table
    Xtrain = []
    Ytrain2 = []
    files = []
    app_names = data_table_com['docname'].tolist()
    app_nos = data_table_com['appno'].tolist()
    app_nos = [i.split(';') for i in app_nos]
    Ytrain = []
    for num in range(len(app_names)):
        if app_names[num] not in files:
            files.append(app_names[num])
            Ytrain.append(0) #and add it's label to the Ytrain
    print('Files:', len(files), 'Labels:', len(Ytrain))
    app_names = [i+'.txt' for i in files]
    app_names = [unicodedata.normalize('NFD', app_name) for app_name in app_names]
    count = -1
    for app_name in app_names:  # for every app name in the table com_with_jud
        count += 1
        try:
            with open(PATH_TO_COMM_CASES + app_name, 'rb') as f:  # change to the name of the folder where comm cases are
                f = f.readlines()  # read the file
                f = [i.decode("utf-8") for i in f]
                if f != False:
                    f = [re.sub('^.*v\. .* STATEMENT OF.*', '', i) for i in f]
                    f = [re.sub('^\n$', '', i) for i in f]
                    f = [re.sub('\n', '', i) for i in f]
                    f = ' '.join(f)  # combine lines
                    f = re.sub('  ', ' ', f)
                    f = re.sub('  ', ' ', f)
                    if len(f) > 1:
                        Xtrain.append(f)
                        Ytrain2.append(0)
        except:
            print('didn\'t work', app_name)
            pass
    return Xtrain, Ytrain2


def retrieve_dec_data():  # extracting text for decisions
    data_table_dec = pd.read_csv('./tables/table_com_with_dec_'+str(article)+'.csv', sep='|', parse_dates=['kpdate'], date_parser=dateparse)
    df = data_table_dec
    df = df.sort_values(by='kpdate')
    df['Year'] = df['kpdate'].dt.year
    df['Month'] = df['kpdate'].dt.month
    df['Day'] = df['kpdate'].dt.day
    dec, dec_dev = [x for _, x in df.groupby(df['Year'] > 2017)]

    Xtrain2, Ytrain2 = extract_text_dec(dec)
    Xtrain3, Ytrain3 = extract_text_dec(dec_dev)
    return Xtrain2, Ytrain2, Xtrain3, Ytrain3


def test_model(Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest, test=False):
    print('Article:', article)
    vec = ('wordvec', TfidfVectorizer(analyzer='word', binary=True, lowercase=True, min_df=2,  ngram_range=(2, 4), norm='l2', stop_words=None, use_idf=True))
    c = 5
#ngram 2,4
    pipeline = Pipeline([vec,
                        ('classifier', LinearSVC(C=c))])
    print('fitting...')
    pipeline.fit(Xtrain, Ytrain)
    #print('testing using cross-validation...')
    #Ypredict = cross_val_predict(pipeline, Xtrain, Ytrain, cv=5)
    #evaluate(Ytrain, Ypredict)
    print('testing on dev set...')
    Ypredict = pipeline.predict(Xdev)
    evaluate(Ydev, Ypredict)

    if test:
        Xtrain += Xdev
        Ytrain += Ydev
        print('fitting...')
        pipeline.fit(Xtrain, Ytrain)
        print('testing on test set...')
        Ypredict = pipeline.predict(Xtest)
        evaluate(Ytest, Ypredict)


def evaluate(Ytest, Ypredict):  # evaluate the model (accuracy, precision, recall, f-score, confusion matrix)
    print('\nClassification report:\n', classification_report(Ytest, Ypredict))
    print('\nConfusion matrix:\n', confusion_matrix(Ytest, Ypredict), '\n\n_______________________\n\n')
    print('\n Normalized confusion matrix:\n', confusion_matrix(Ytest, Ypredict, normalize='true'), '\n\n_______________________\n\n')

article = 0

dateparse = lambda dates: [pd.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]

PATH_TO_COMM_CASES = './data_txt/communicated_txt/'
# table_jud_with_com = pd.read_csv(u'./tables/table_jud_with_com_'+str(article)+'.csv', sep='|', parse_dates=['kpdate'], date_parser=dateparse)  # to extract labels
# table_com_with_jud = pd.read_csv(u'./tables/table_com_with_jud_'+str(article)+'.csv', sep='|')  # to extract text
# table_jud_wo_com = pd.read_csv(u'./tables/table_jud_wo_com_'+str(article)+'.csv', sep='|', parse_dates=['kpdate'], date_parser=dateparse)  # to extract labels
# session.query(Judgments)

query_jud_with_com = session.Query(CommunicatedCases).filter()

table_jud_with_com = pd.read_sql(.statement, query.session.bind)
table_com_with_jud = pd.read_sql(query.statement, query.session.bind)
table_jud_wo_com = pd.read_sql(query.statement, query.session.bind)

#choosing the years for training set (communicated cases)
df = table_jud_with_com
df = df.sort_values(by='kpdate')
df['Year'] = df['kpdate'].dt.year
df['Month'] = df['kpdate'].dt.month
df['Day'] = df['kpdate'].dt.day
train, dev = [x for _, x in df.groupby(df['Year'] > 2017)]
#dev, test = [x for _, x in dev.groupby(dev['Year'] > 2018)]
#garbage, test = [x for _, x in test.groupby(test['Month'] == 2)]
#test

df = table_jud_wo_com  # years for additional data judgements without communicated cases
df = df.sort_values(by='kpdate')
df['Year'] = df['kpdate'].dt.year
df['Month'] = df['kpdate'].dt.month
df['Day'] = df['kpdate'].dt.day
judg, garbage = [x for _, x in df.groupby(df['Year'] > 2017)]
garbage, judg = [x for _, x in judg.groupby(df['Year'] > 2007)]

#tests
appnos_with_labels = extract_labels(train)
comm_text_with_labels = extract_text(table_com_with_jud, appnos_with_labels, PATH_TO_COMM_CASES)

appnos_with_labels_dev = extract_labels(dev)
comm_text_with_labels_dev = extract_text(table_com_with_jud, appnos_with_labels_dev, PATH_TO_COMM_CASES)

#appnos_with_labels_test = extract_labels(test)
#comm_text_with_labels_test = extract_text(table_com_with_jud, appnos_with_labels_test, PATH_TO_COMM_CASES)

appnos_with_labels_judg = extract_labels(judg)  # extract labels for judgmemnts without communicated cases

#extract text for judgmemnts without communicated cases
#I am missong a whole bunch of cases in txt, not sure what happenend there, but should not matter too much
judg_text_with_labels = extract_text(table_jud_wo_com, appnos_with_labels_judg, './data_txt/judgments_txt/', judg=True)

len(judg_text_with_labels), len(comm_text_with_labels)

Xtrain_facts = []
Ytrain_facts = []
count = 0
for key, value in judg_text_with_labels.items():
    text = value[0]
    #print(text)
    m = re.search('(FACTS )(.+?) (AS TO )?(THE)? (LAW|RELEVANT)', text)
    m2 = re.search('(THE CIRCUMSTANCES OF THE CASE) (.+?) (AS TO )?(LAW|(II. )?RELEVANT)', text)
    if m2 != None:
        text = m2.group(2)
        # text = re.sub('[A-Z]+ v. [A-Z ]+ JUDGMENT', '', text)
        Xtrain_facts.append(text)
        Ytrain_facts.append(value[1])
    else:
        if m != None:
            text = m.group(2)
            # text = re.sub('[A-ZĐĆ ]+ v. [A-Z ]+ JUDGMENT', '', text)
            Xtrain_facts.append(text)
            Ytrain_facts.append(value[1])
        else:
            count += 1
            print('Not found', count)
            #print(text)
    
print(len(Ytrain_facts))

Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev = [], [], [], [], [], []  # no balancing

for key, value in comm_text_with_labels.items():
    Xtrain.append(value[0])
    Ytrain.append(value[1])
print('Train:', len(Ytrain), 'Violation:', Ytrain.count(1), 'No violation:', Ytrain.count(0))

for key, value in comm_text_with_labels_dev.items():
    Xdev.append(value[0])
    Ydev.append(value[1])
print('Dev:', len(Ydev), 'Violation:', Ydev.count(1), 'No violation:', Ydev.count(0))


# for key, value in comm_text_with_labels_test.items():
#     Xtest.append(value[0])
#     Ytest.append(value[1])
# print('Test:', len(Ytest),'Violation:', Ytest.count(1), 'No violation:', Ytest.count(0))

Xtrain_non = []
Ytrain_non = []
Xtrain_v = []
Ytrain_v = []
for num in range(len(Xtrain_facts)):
    if Ytrain_facts[num] == 0:
        Xtrain_non.append(Xtrain_facts[num])
        Ytrain_non.append(0)
    else:
        Xtrain_v.append(Xtrain_facts[num])
        Ytrain_v.append(1)

print(len(Ytrain_non), len(Ytrain_v))

# Xtrain += Xtrain_facts[-1000:]
# Ytrain += Ytrain_facts[-1000:]
Xtrain += Xtrain_non#[-100:]
Ytrain += Ytrain_non#[-100:]
Xtrain += Xtrain_v#[-100:]
Ytrain += Ytrain_v#[-100:]

#add decisions with non violation label
Xtrain2, Ytrain2, Xtrain3, Ytrain3 = retrieve_dec_data()
Xtrain += Xtrain2  # add communicated cases that were found inaddmissible to the training set
Ytrain += Ytrain2
Xdev += Xtrain3  # add communicated cases that were found inaddmissible to the training set
Ydev += Ytrain3

print(Ytrain.count(0), Ytrain.count(1))

Xtest, Ytest = [], []  # replace with test if necessary
test_model(Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest, test=False)
