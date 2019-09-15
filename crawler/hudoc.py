#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import argparse
import unicodedata
from urllib.parse import unquote
from multiprocessing.pool import ThreadPool

# import grequests
import json
import requests
import pandas
import tqdm
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from sqlalchemy.dialects import mysql
from nltk.tokenize import sent_tokenize

from config import Config

DOC_URL = 'https://hudoc.echr.coe.int/app/conversion/docx/html/body?library=ECHR&id=%s'
LIST_FULL_URL = 'https://hudoc.echr.coe.int/app/query/results?query=contentsitename:ECHR AND (NOT (doctype=PR OR doctype=HFCOMOLD OR doctype=HECOMOLD)) AND ((languageisocode="%s")) AND ((documentcollectionid="%s"))&select=sharepointid,Rank,ECHRRanking,itemid,docname,doctype,application,appno,conclusion,importance,originatingbody,typedescription,kpdate,extractedappno,doctypebranch,respondent&sort=&start=%d&length=%d&rankingModelId=1111111-0000-0000-0000-0000'

DIRECTORY = os.path.dirname(os.path.abspath(__file__))

HEADER_INFO = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36',
    'Host': 'hudoc.echr.coe.int',
    'Origin': 'https://hudoc.echr.coe.int',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded',
    'X-Requested-With': "XMLHttpRequest",
    }

s = requests.session()
if not os.path.isdir('docs'):
    os.mkdir('docs')


class NoDocxException(Exception):
    pass


def get_document_list(col, lang='ENG'):
    HEAD_URL = LIST_FULL_URL % (lang, col, 0, 20)

    res = s.get(HEAD_URL)
    length = res.json()['resultcount']
    print(length)
    docs = []
    # docs = pandas.DataFrame(columns=['name', 'id', 'appno', 'date', 'type', 'branch', 'conclusion', 'respondent', 'url'])
    for i in range(0, length, 1000):
        resp = s.get(LIST_FULL_URL % (lang, col, i, 1000))
        data = resp.json()
        for result in data['results']:
            res = result['columns']
            res['url'] = DOC_URL % (res['itemid'])
            docs.append(res)

    if i:
        resp = s.get(LIST_FULL_URL % (lang, col, i, 1000))
        data = resp.json()
        for result in data['results']:
            res = result['columns']
            res['url'] = DOC_URL % (res['itemid'])
            docs.append(res)

    df = pandas.DataFrame(data=docs)
    df.to_csv(os.path.join(DIRECTORY, '%s_%s.csv' % (col, lang)))


def get_text(response, **kwargs):
    if response:
        soup = BeautifulSoup(response.text, 'html.parser')
        response.close()
        if soup.find('div'):
            text = soup.find('div').text
        else:
            text = ''
        return unicodedata.normalize("NFKD", text).encode('utf-8').decode('utf-8-sig').strip()
    else:
        print('No response')
        print(response.__dict__)
        raise NoDocxException


def get_text_from_url(url):
    try:
        response = requests.get(url, stream=True, headers=HEADER_INFO)
    except:
        response = requests.get(url, headers=HEADER_INFO)
    try:
        return get_text(response)
    except NoDocxException:
        print('No text available: ', url)
        logging.warning(url)
        return ''


def download_documents(col, lang='ENG'):
    df = pandas.read_csv(os.path.join(DIRECTORY, '%s_%s.csv' % (col, lang)))
    urls = df['url'].tolist()
    texts = list(tqdm.tqdm(ThreadPool(8).imap(get_text_from_url, urls, 16)))

    # texts = list(map(get_text_from_url, urls))

    return texts


def update_database(lang='ENG'):
    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
    collections = pandas.read_csv(os.path.join(DIRECTORY, '%s_%s.csv' % ('COMMUNICATEDCASES', lang)))
    decisions = pandas.read_csv(os.path.join(DIRECTORY, '%s_%s.csv' % ('DECISIONS', lang)))
    judgements = pandas.read_csv(os.path.join(DIRECTORY, '%s_%s.csv' % ('JUDGMENTS', lang)))

    dtype_dict = {'docname': mysql.TEXT(unicode=True),
                  'url': mysql.TEXT(unicode=True),
                  'text': mysql.LONGTEXT(unicode=True),
                  'sents': mysql.LONGTEXT(unicode=True),
                  'extractedappno': mysql.LONGTEXT}

    collection_text = download_documents('COMMUNICATEDCASES', lang='ENG')
    collections['text'] = collection_text
    collections['sents'] = list(map(lambda x: json.dumps(sent_tokenize(x)), collection_text))
    collections.to_sql('CommunicatedCases', engine, if_exists='replace', dtype=dtype_dict)
    del collections
    del collection_text

    decisions_text = download_documents('DECISIONS', lang='ENG')
    decisions['text'] = decisions_text
    decisions['sents'] = list(map(lambda x: json.dumps(sent_tokenize(x)), decisions_text))
    decisions.to_sql('Decisions', engine, if_exists='replace', dtype=dtype_dict)
    del decisions
    del decisions_text

    judgements_text = download_documents('JUDGMENTS', lang='ENG')
    judgements['text'] = judgements_text
    judgements['sents'] = list(map(lambda x: json.dumps(sent_tokenize(x)), judgements_text))
    print(len(judgements['text']))
    judgements.to_sql('Judgments', engine, if_exists='replace', dtype=dtype_dict)

    with engine.connect() as con:
        con.execute('alter table CommunicatedCases add column `id` int(10) unsigned PRIMARY KEY AUTO_INCREMENT;')
        con.execute('alter table Decisions add column `id` int(10) unsigned PRIMARY KEY AUTO_INCREMENT;')
        con.execute('alter table Judgments add column `id` int(10) unsigned PRIMARY KEY AUTO_INCREMENT;')


def main():
    parser = argparse.ArgumentParser(description='Download cases from HUDOC')
    parser.add_argument('collection', type=str, help='Type of documents. Options: DECISIONS, JUDGMENTS, RESOLUTIONS')
    parser.add_argument('language', type=str, nargs='?', default='ENG',
                        help='Language code (default: ENG)')
    parser.add_argument('-d', '--download', help='Download text to database', action='store_true')
    parser.add_argument('-u', '--update', help='Update cases', action='store_true')
    args = vars(parser.parse_args())
    logging.basicConfig(filename='log_%d.log' % time.time(), level=logging.INFO, format='%(message)s')

    if args['update'] or not os.path.exists(os.path.join(DIRECTORY, '%s_%s.csv' % (args['collection'], args['language']))):
        get_document_list(args['collection'], args['language'])

    if args['download']:
        # download_documents(args['collection'], args['language'])
        update_database(lang='ENG')


if __name__ == '__main__':
    main()
