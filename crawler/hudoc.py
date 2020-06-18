#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import itertools
import datetime
import argparse
import importlib
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
from sqlalchemy.orm import sessionmaker
from nltk.tokenize import sent_tokenize

from config.config import Config

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, encoding='utf-8', echo=True)

DOC_URL = 'https://hudoc.echr.coe.int/app/conversion/docx/html/body?library=ECHR&id=%s'
LIST_FULL_URL = 'https://hudoc.echr.coe.int/app/query/results?query=contentsitename:ECHR AND (NOT (doctype=PR OR doctype=HFCOMOLD OR doctype=HECOMOLD)) AND ((languageisocode="%s")) AND ((documentcollectionid="%s"))&select=sharepointid,Rank,ECHRRanking,itemid,docname,doctype,application,appno,conclusion,importance,originatingbody,typedescription,kpdate,extractedappno,doctypebranch,respondent,article&sort=&start=%d&length=%d&rankingModelId=1111111-0000-0000-0000-0000'

DIRECTORY = os.path.dirname(os.path.abspath(__file__))

HEADER_INFO = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2526.80 Safari/537.36',
    'Host': 'hudoc.echr.coe.int',
    'Origin': 'https://hudoc.echr.coe.int',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded',
    'X-Requested-With': "XMLHttpRequest",
    }


SAMPLE = '''The European Commission of Human Rights sitting in private on1 December 1986, the following members being present:                     MM. C.A. NØRGAARD, President                        E. BUSUTTIL                        G. JÖRUNDSSON                        S. TRECHSEL                        B. KIERNAN                        A.S. GÖZÜBÜYÜK                        A. WEITZEL                        J.C. SOYER                        H.G. SCHERMERS                        H. DANELIUS                        G. BATLINER                    Mrs G.H. THUNE                    Sir Basil HALL                    Mr. F. MARTINEZ                     Mr. J. RAYMOND, Deputy Secretary to the Commission Having regard to Article 25 (art. 25) of the Convention for theProtection of Human Rights and Fundamental Freedoms; Having regard to the application introduced on 24 March 1986 by K.P.against the Federal Republic of Germany and registered on 26 March1986 under file No. 12068/86; Having regard to the report provided for in Rule 40 of the Rules ofProcedure of the Commission; Having deliberated; Decides as follows: THE FACTS The applicant, a Tamil, is a citizen of Sri Lanka.  He was born in1954 and is presently living at Bonn.  In the proceedings before theCommission he is represented by Mr. N. Wingerter and others, lawyersin Heilbronn. On 6 August 1984 the applicant was convicted by the Heilbronn DistrictCourt (Amtsgericht) of having violated the Act on Asylum Proceedings(Asylverfahrensgesetz).  He was fined 50.- DM. According to the findings of the court the applicant entered theFederal Republic of Germany in 1979 and made a request to be grantedasylum.  The proceedings concerning this request were still pendingand the applicant was granted a provisional residence permit(Aufenthaltsgestattung und Aufenthaltserlaubnis) limited to thedistrict (Stadt- und Landkreis) of Heilbronn.  On 30 November 1982 thecompetent authorities at Heilbronn issued a document (Ausweis) to theapplicant called "residence permit for the Federal Republic of Germanyincluding the Land Berlin" for the purpose of the asylum proceedings. On page 3 of the document the following restriction is indicated: "Residence is limited to the district of Heilbronn.  Leaving thisdistrict requires special authorisation by the Office for Foreigners(Ausländerbehörde)". On page 2 a bold faced typed warning states that violations ofconditions or restrictions are punishable.  Receipt of the documenthas to be signed by the bearer. In December 1982 the applicant was fined for a traffic offencecommitted outside the district of Heilbronn.  He was therefore warnedby an officer of the Office for Foreigners when his residence permitwas prolonged in May 1983 that he was not allowed to leave thedistrict of Heilbronn.  Nevertheless on 23 September 1983 he travelledto Stuttgart in order to fetch someone at the railway station.  He wasintercepted by a police control. The court concluded that the applicant had thus violated Sections 35(1) and 20 (2) of the Act on Asylum Proceedings and that he waspunishable as he must have been aware that he did not have the rightto leave the district of Heilbronn.  The Court moreover considered theapplicant's submissions that the restriction in question would violatethe freedom of movement as guaranteed by the German Basic Law(Grundgesetz).  It found, however, that the restriction did notviolate the freedom of movement because the person concerned was freeto move within the district he was allowed to reside in.  Thelegitimate purpose of the restriction was to enable the authorities tosupervise the activities of persons requesting asylum and to avoidtheir going underground. On 7 February 1985 the Heilbronn Regional Court (Landgericht)dismissed both the applicant's and the Public Prosecutor's appeal(Berufung). The applicant's appeal was declared inadmissible because neither theapplicant himself nor his defence counsel attended the hearing on theappeal. With regard to the appeal lodged by the Public Prosecutor the Courtstated that the restriction in question did not violate constitutionalrights because the residence permit was only granted to the extentnecessary to enable the applicant to pursue his request for asylum.There was neither necessity nor a constitutional requirement to allowa person requesting asylum a more extensive right to travel in theFederal Republic before asylum was in effect granted.  Insofar as thePublic Prosecution had argued that the incriminated act should beconsidered as a misdemeanour (Vergehen) and not just as a regulatoryoffence (Ordnungswidrigkeit) the appeal was considered to beunfounded. The applicant lodged a further appeal which was, on 10 October 1985,declared inadmissible by the Stuttgart Court of Appeal(Oberlandesgericht).  This Court likewise dismissed on27 November 1985 an appeal (Revision) lodged by the Public Prosecutor.In rejecting the Prosecutor's appeal the Appellate Court also examinedin accordance with S. 301 of the Code on Criminal Procedure(Strafprozessordnung) whether the judgment of 7 February 1985contained errors of material law (sachlichrechtliche Mängel) to theapplicant's disadvantage.  Referring to a decision of the FederalConstitutional Court (Bundesverfassungsgericht) of 7 July 1983 itstated in this context that it was compatible both with constitutionaland Convention rights to limit to certain districts residence permitsfor persons requesting asylum. The applicant submits that he did not lodge a constitutional complaintas such remedy would have been ineffective in the light of theexisting case law. COMPLAINTS The applicant considers that he was wrongly convicted and finedbecause the limitation of his residence permit violates his freedom ofmovement as guaranteed by Article 2 of Protocol No. 4 (P4-2). He alsoinvokes Articles 5, 8, 11 (art. 5), (art. 8), (art. 11) and 14(art. 14) of the Convention and submits that limitations of the kindin question are unnecessary in a democratic society and thereforearbitrary. THE LAW 1.      The applicant complains under Article 2 of Protocol No. 4(P4-2) that he was fined in criminal proceedings for having violatedthe obligation imposed on him in connection with a provisionalresidence permit to remain within the district of the city ofHeilbronn pending the proceedings concerning his request to be grantedasylum.  Article 2 of Protocol No. 4 (P4-2) provides: "Everyone lawfully within the territory of a State shall, within thatterritory have the right to freedom of movement and freedom to choosehis residence." The Commission notes at the outset that before the respective GermanCourts the applicant did not invoke Article 2 of Protocol No. 4(P4-2).  He rather referred to provisions of the German Basic Law.  Anissue therefore arises as to whether the applicant has properlyexhausted the domestic remedies within the meaning of Article 26(art. 26) of the Convention.  The Commission nevertheless leaves thisquestion open since the above complaint is in any event manifestlyill-founded for the following reasons. The Commission notes that, in accordance with S. 17 para. 1 and S. 7para. 1 of the German Aliens' Act, the applicant was onlyprovisionally permitted to stay in the district of the city ofHeilbronn pending the proceedings concerning his renewed requests forasylum. The Commission observes that Article 2 para. 1 of Protocol No. 4(P4-2-1) secures the freedom of movement to persons "lawfully within theterritory of a State".  This condition refers to the domestic law ofthe State concerned.  It is for the domestic law and organs to laydown the conditions which must be fulfilled for a person's presence inthe territory to be considered "lawful".  The Commission, in thisrespect, recalls its constant case-law according to which there is noright of an alien to enter, reside or remain in a particular country,as such, guaranteed by the Convention (cf. No. 9285/81, Dec. 6.7.82,D.R. 29 p. 205).  The Commission is of the opinion that aliensprovisionally admitted to a certain district of the territory of aState, pending proceedings to determine whether or not they areentitled to a residence permit under the relevant provisions ofdomestic law, can only be regarded as "lawfully" in the territory aslong as they comply with the conditions to which their admission andstay are subjected. In the present case the applicant's provisional admission to theterritory of the Federal Republic of Germany is subject to thecondition that it extends only to the district of the city ofHeilbronn.  His "lawful" stay within the territory is, therefore,geographically limited.  Article 2 of Protocol No. 4 (P4-2) does notextend that right. Consequently, the applicant's complaint that he is not grantedgeographically unlimited permissions to stay within the territory ofthe Federal Republic of Germany is manifestly ill-founded within themeaning of Article 27 para. 2 (art. 27-2) of the Convention. 2.      The applicant has also invoked Articles 5, 8, 11 (art. 5),(art. 8), (art. 11) and 14 (art. 14) of the Convention in respect ofhis complaint concerning the restriction of his freedom of movement.However, the Commission finds that there is no appearance of aviolation of the rights and freedoms set out in Articles 5, 8, 11(art. 5), (art. 8), (art. 11) or 14 (art. 14) of the Convention.  Itfollows that this part of the application is manifestly ill-foundedwithin the meaning of Article 27 para. 2 (art. 27-2) of the Convention. For these reasons, the Commission DECLARES THE APPLICATION INADMISSIBLE. Deputy Secretary to the Commission        President of the Commission (J. RAYMOND)                              (C.A. NØRGAARD)'''


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
            ps = itertools.chain(*list(map(lambda x: x.find_all('p'), soup.find_all('div'))))
            spans = [''.join(list(map(lambda x: x.text, p.find_all('span')))) for p in ps]
            text = '\n'.join(spans)
        else:
            text = ''
        return unicodedata.normalize("NFKC", text).encode('utf-8').decode('utf-8-sig').strip()
    else:
        print('No response')
        print(response.__dict__)
        raise NoDocxException


def get_text_from_url(url):
    # try:
    #     response = s.get(url, stream=True, headers=HEADER_INFO)
    # except:
    response = s.get(url, headers=HEADER_INFO)
    try:
        return get_text(response)
    except NoDocxException:
        print('No text available: ', url)
        logging.warning(url)
        return ''


def download_documents(col, lang='ENG', table=None):
    if table and engine.dialect.has_table(engine, table):
        Session = sessionmaker(bind=engine)
        session = Session()
        database = importlib.import_module("db.database")
        t = getattr(database, table)
        df = pandas.read_csv(os.path.join(DIRECTORY, '%s_%s.csv' % (col, lang)))
        texts = []
        for url in df['url']:
            q = session.query(t).filter_by(url=url).first()
            if q:
                texts.append(q.text)
            else:
                texts.append(get_text_from_url(url))

    else:
        # parallel download
        df = pandas.read_csv(os.path.join(DIRECTORY, '%s_%s.csv' % (col, lang)))
        # urls = df['url'].tolist()
        texts = []
        # texts = list(tqdm.tqdm(ThreadPool(8).imap(get_text_from_url, urls, 16)))

        # when parallel is not working
        for url in df['url']:
            print(url)
            texts.append(get_text_from_url(url))

    # texts = list(map(get_text_from_url, urls))

    return texts


def update_datetime(s):
    return datetime.datetime.strptime(s, '%m/%d/%Y %I:%M:%S %p')


def update_database(lang='ENG'):
    lang_postfix = '' if lang == 'ENG' else '_' + lang
    collections = pandas.read_csv(os.path.join(DIRECTORY, '%s_%s.csv' % ('COMMUNICATEDCASES', lang)))
    decisions = pandas.read_csv(os.path.join(DIRECTORY, '%s_%s.csv' % ('DECISIONS', lang)))
    judgements = pandas.read_csv(os.path.join(DIRECTORY, '%s_%s.csv' % ('JUDGMENTS', lang)))

    dtype_dict = {'docname': mysql.TEXT(unicode=True),
                  'url': mysql.TEXT(unicode=True),
                  'text': mysql.LONGTEXT(unicode=True),
                  'sents': mysql.LONGTEXT(unicode=True),
                  'extractedappno': mysql.LONGTEXT}

    # comm_created = engine.dialect.has_table(engine, 'CommunicatedCases%s' % lang_postfix)
    collection_text = download_documents('COMMUNICATEDCASES', lang=lang, table='CommunicatedCases'+lang_postfix)
    print(len(collections), len(collection_text))

    collections['kpdate'] = list(map(update_datetime, collections['kpdate']))
    collections['text'] = collection_text
    collections['sents'] = list(map(lambda x: json.dumps(sent_tokenize(x)), collection_text))
    collections.to_sql('CommunicatedCases'+lang_postfix, engine, if_exists='replace', dtype=dtype_dict)
    del collections
    del collection_text

    with engine.connect() as con:
        con.execute('alter table CommunicatedCases%s add column `id` int(10) unsigned PRIMARY KEY AUTO_INCREMENT;' % lang_postfix)
        con.execute('ALTER TABLE CommunicatedCases%s ADD INDEX idx_text(appno(15));' % lang_postfix)


    decisions_text = download_documents('DECISIONS', lang=lang, table='Decisions'+lang_postfix)
    decisions['kpdate'] = list(map(update_datetime, decisions['kpdate']))
    decisions['text'] = decisions_text
    decisions['sents'] = list(map(lambda x: json.dumps(sent_tokenize(x)), decisions_text))
    decisions.to_sql('Decisions'+lang_postfix, engine, if_exists='replace', dtype=dtype_dict)
    del decisions
    del decisions_text

    with engine.connect() as con:
        con.execute('alter table Decisions%s add column `id` int(10) unsigned PRIMARY KEY AUTO_INCREMENT;' % lang_postfix)
        con.execute('ALTER TABLE Decisions%s ADD INDEX idx_text(appno(15));' % lang_postfix)

    judgements_text = download_documents('JUDGMENTS', lang=lang, table='Judgments'+lang_postfix)
    judgements['kpdate'] = list(map(update_datetime, judgements['kpdate']))
    judgements['text'] = judgements_text
    judgements['sents'] = list(map(lambda x: json.dumps(sent_tokenize(x)), judgements_text))
    print(len(judgements['text']))
    judgements.to_sql('Judgments'+lang_postfix, engine, if_exists='replace', dtype=dtype_dict)

    with engine.connect() as con:
        con.execute('alter table Judgments%s add column `id` int(10) unsigned PRIMARY KEY AUTO_INCREMENT;' % lang_postfix)
        con.execute('ALTER TABLE Judgments%s ADD INDEX idx_text(appno(15));' % lang_postfix)


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
        update_database(lang=args['language'])




if __name__ == '__main__':
    main()
