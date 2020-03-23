#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import logging
import itertools
import datetime
import argparse
import unicodedata
import pdftotext
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

from config.config import Config
from crawler.hudoc import update_datetime

DOC_URL = 'https://hudoc.echr.coe.int/app/conversion/pdf?library=ECHR&id=%s&filename=%s.pdf'
# LIST_FULL_URL = 'https://hudoc.echr.coe.int/app/query/results?query=contentsitename:ECHR AND (NOT (doctype=PR OR doctype=HFCOMOLD OR doctype=HECOMOLD)) AND ((languageisocode="%s")) AND ((documentcollectionid="%s"))&select=sharepointid,Rank,ECHRRanking,itemid,docname,doctype,application,appno,conclusion,importance,originatingbody,typedescription,kpdate,extractedappno,doctypebranch,respondent,article&sort=&start=%d&length=%d&rankingModelId=1111111-0000-0000-0000-0000'
LIST_FULL_URL = r'https://hudoc.echr.coe.int/app/query/results?query=contentsitename%3AECHR%20AND%20doctype%3DPR%20AND%20((languageisocode%3D%22ENG%22))%20AND%20((documentcollectionid%3D%22FORTHCOMINGJUDGMENTS%22))&select=sharepointid,Rank,ECHRRanking,languagenumber,itemid,docname,doctype,application,appno,conclusion,importance,originatingbody,typedescription,kpdate,kpdateAsText,documentcollectionid,documentcollectionid2,languageisocode,extractedappno,isplaceholder,doctypebranch,respondent,advopidentifier,advopstatus,ecli,appnoparts,sclappnos&sort=&start=0&length=200&rankingModelId=11111111-0000-0000-0000-000000000000'

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


def get_document_list():
    HEAD_URL = LIST_FULL_URL
    path = os.path.join(DIRECTORY, 'press-releases/')
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    res = s.get(HEAD_URL)
    length = res.json()['resultcount']
    print(length)
    docs = []
    # docs = pandas.DataFrame(columns=['name', 'id', 'appno', 'date', 'type', 'branch', 'conclusion', 'respondent', 'url'])
    # for i in range(0, length, 1000):
    #     resp = s.get(LIST_FULL_URL % (lang, col, i, 1000))
    for result in res.json()['results']:
        res = result['columns']
        if res['docname'][-1] not in '0123456789':
            continue
        res['url'] = DOC_URL % (res['itemid'], res['docname'])
        doc_res = s.get(res['url'])
        filename = res['docname'] + '.pdf'
        with open(os.path.join(path, filename), 'wb') as f:
            f.write(doc_res.content)
        doc_res.close()

        docs.append(res)

    df = pandas.DataFrame(data=docs)
    df.to_csv(os.path.join(DIRECTORY, 'PRESS_RELEASES.csv'))


def get_appnos():
    press = pandas.read_csv(os.path.join(DIRECTORY, 'PRESS_RELEASES.csv'))
    # if os.path.exists(filename+'.txt'):
    #     with open(filename+'.txt') as f:
    #         text = f.read()
    # else:
    all_appnos = []
    for docname in press['docname'].tolist():
        print(docname)
        appnos = get_appnos_from_file(docname)
        all_appnos.append(appnos)
    return all_appnos


def get_appnos_from_file(docname):
    filename = os.path.join(DIRECTORY, 'press-releases/', docname)
    appnos = []
    with open(filename+'.pdf', "rb") as f:
        try:
            for idx, page in enumerate(pdftotext.PDF(f)):
                lines = page.split('\n')
                for line in lines:
                    if 'The Court will give its rulings in writing on the following cases' in line:
                        return list(set(appnos))
                    line_appnos = [x.strip() for x in re.findall(r' [0-9]+/[0-9]+', line)]
                    # put appnos at the same line together. or we remove repetitons later
                    # if len(line_appnos) == 1:
                    #     appnos.append(line_appnos[0])
                    # elif len(line_appnos) >= 2:
                    #     appnos.append(';'.join(line_appnos))
                    appnos.extend(line_appnos)
        except pdftotext.Error:
            logging.log(filename)
            return []
        finally:
            return list(set(appnos))


def update_database(lang='ENG'):
    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)
    press = pandas.read_csv(os.path.join(DIRECTORY, 'PRESS_RELEASES.csv'))

    dtype_dict = {'docname': mysql.TEXT(unicode=True),
                  'url': mysql.TEXT(unicode=True)}

    appnos = get_appnos()
    press['kpdate'] = list(map(update_datetime, press['kpdate']))
    press['appnos'] = list(map(lambda x: json.dumps(x), appnos))
    # press['sents'] = list(map(lambda x: json.dumps(sent_tokenize(x)), judgements_text))
    press.to_sql('Press', engine, if_exists='replace', dtype=dtype_dict)

    with engine.connect() as con:
        con.execute('alter table Press add column `id` int(10) unsigned PRIMARY KEY AUTO_INCREMENT;')
        con.execute('ALTER TABLE Press ADD INDEX idx_text(docname(15));')


def main():
    get_document_list()
    update_database()


if __name__ == '__main__':
    main()
