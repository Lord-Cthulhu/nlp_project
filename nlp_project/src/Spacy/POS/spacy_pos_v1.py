#Data

import numpy as np
import pandas as pd

#Spacy 
import spacy
from spacy import displacy
from spacy.tokens import Span 

#NLTK - Elmo, Word2vec, etc.
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from yaml import scan
nltk.download('words')
from nltk import stem
#from pysbd.utils import PySBDFactory
import re


#SVM model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics

#Tensorflow
#import tensorflow as tf
#import tensorflow_hub as hub

from pathlib import Path

import time
#import torch
import functools
from functools import cache

from tqdm import tqdm
from collections import Counter


# def run():
#     torch.multiprocessing.freeze_support()
#     print('loop')
    
# if __name__ == '__main__':
#     run()

def excel_to_df(filepath):
    '''
    [Extraction des données excel en df]
    [Spécifier:]
    [filepath = Localisation du fichier "data.csv"]
    '''
    try: 
        try:
            df = pd.read_csv(filepath)
            print('csv passed')
            return df
        except Exception as e:
            pass
        try:
            df = pd.read_excel(filepath)
            print('xlsx passed')
            return df
        except Exception as e:
            pass    
    except Exception as e:
        print('excel_to_df failed')
        print(f'error: {e}'.format(e))
        pass

@cache
class QuatitativeNLP_FR: 
    '''
    [Token: ]
    [Stemming: ]
    [PoS Tagging: ]
    [ ]
    '''

    nlp = spacy.load("en_core_web_sm")
    
    def __init__(self, df):
        self.df = df
        self.nlp = nlp
                
    def remove_spec_char(df):
        df = df.lower()
        #https://www.programcreek.com/python/?CodeExample=remove+accents
        df = re.sub(u"[àáâãäå]", 'a', df)
        df = re.sub(u"[èéêë]", 'e', df)
        df = re.sub(u"[ìíîï]", 'i', df)
        df = re.sub(u"[òóôõö]", 'o', df)
        df = re.sub(u"[ùúûü]", 'u', df)
        df = re.sub(u"[ýÿ]", 'y', df)
        df = re.sub(u"[ß]", 'ss', df)
        df = re.sub(u"[ñ]", 'n', df)
        ####
        
        df = re.sub(r'[^a-zA-Z0-9\s]', ' ', df)
        df = re.sub(' +', ' ',df)

        print(df)
        return df
    
    def tokenize(df):
        try:
            document = nlp(df)
            phrase = []
            for i in document:
                phrase.append(i.text) 
            return phrase
        except Exception as e:
            print('Tokenize Error')
            print(e)
            pass
    
    def remove_stopwords(df):
        try:
            df = df.lower()
            document = nlp(df)
            stpwrds = set(stopwords.words('english'))
            clean = []
            clean_phrase = [] 
            stopword = []
            for i in document:
                if i.text in stpwrds:
                    stopword.append(i.text)
                else:
                    clean.append(i.text)  
                    clean_phrase.append(i.text)  
            #print(stopword) 
            
            phrase = ' '.join(word for word in clean)
                 
            return phrase, stopword, clean
        except Exception as e:
            print('Stopwords Error')
            print(e)
            pass
    
    def stemming(df):
        try:
            stemmer = SnowballStemmer(language='english')
            #stemmer = stem.re('s$|es$|era$|erez$|ions$')
            document = nlp(df)
            stem = []
            for i in document:
                stem.append(stemmer.stem(i.text)) 
                
            phrase = ' '.join(word for word in stem)
            return phrase
        except Exception as e:
            print('Stemming Error')
            print(e)
            pass
    
    def lemmatisation(df):
        try:
            document = nlp(df)
            lemma = []
            for i in document:
                lemma.append((i.lemma_))
            phrase = ' '.join(word for word in lemma)
            return phrase
        except Exception as e:
            print('Lemmatisation Error')
            print(e)
            pass    
    
    def pos_tagging(df):
        try:
            document = nlp(df)
            phrase = []
            for i in document:
                phrase.append((i.text,i.pos_)) 
            return phrase
        except Exception as e:
            print('PoS Error')
            print(e)
            pass
    
    def entities(df):
        try:
            document = nlp(df)
            phrase = []
            for i in document.ents:
                phrase.append((i.text,i.label_)) 
                print('i')
                print(i)
            return phrase
        except Exception as e:
            print('entities Error')
            print(e)
            pass
        
    def morphology(df):
        try:
            document = nlp(df)
            phrase = []
            for i in document:
                 phrase.append((i.text, i.morph)) 
            return phrase    
        except Exception as e:
            print('morphology Error')
            print(e)
            pass
        
    def sentence_boundary(df):
        try:
            document = nlp(df)
            phrase = []
            for i in document.sents:
                 phrase.append(i.sent) 
            return phrase    
        except Exception as e:
            print('morphology Error')
            print(e)
            pass

    def ngrams(df, n):
        try:
            df = re.sub(r'[^a-zA-Z0-9\s]', ' ', df)
            words = [word for word in df.split(' ') if word != '']
            ngrams = zip(*[words[i:] for i in range(n)])
            grams= [' '.join(ngram) for ngram in ngrams]
            return grams
        except Exception as e:
            print('ngrams Error')
            print(e)
            pass
        
    def multi_analysis(df):
        try:
            document = nlp(df)
            phrase = []
            phrase2 = []
            for i in document:
                phrase.append((i.text,i.pos_, i.morph, i.sent)) 
            for i in document.ents:
                phrase2.append((i.text,i.label_)) 
            return phrase, phrase2
        except Exception as e:
            print('multi_analysis Error')
            print(e)
            pass

    def pos_percent(df):
        try:
            document = nlp(df)
            global_pos = []
            count = Counter(([i.pos_ for i in document]))
            print(count)
            c_values = sum(count.values())
            for pos_type, cnt in count.items():
                print(pos_type, cnt, '{0:2.2f}%'.format((100.0* cnt)/c_values))
                global_pos.append((pos_type, cnt))
            print(global_pos)
            return count, global_pos
        except Exception as e:
            print('pos_percent Error')
            print(e)
            pass   
    
    def pos_count(df):
        try:
            document = nlp(df)
            document.text
            pos_X = []
            pos_NOUN = []
            pos_DET = []
            pos_ADV = []
            pos_AUX = []
            pos_PRON = []
            pos_SCONJ = []
            pos_CONJ = []
            pos_VERB = []
            pos_PROPN = []
            pos_PUNCT = []
            pos_INTJ = []
            pos_ADJ = []
            pos_ADP = []
            pos_NUM = []
            pos_SYM = []
            pos_PART = []
            
            for i in document:
                if i.pos_ == 'X':
                    pos_X.append([i.text])
                if i.pos_ == 'NOUN':
                    pos_NOUN.append([i.text])
                if i.pos_ == 'DET':
                    pos_DET.append([i.text])
                if i.pos_ == 'ADV':
                    pos_ADV.append([i.text])
                if i.pos_ == 'AUX':
                    pos_AUX.append([i.text])
                if i.pos_ == 'PRON':
                    pos_PRON .append([i.text])
                if i.pos_ == 'SCONJ':
                    pos_SCONJ.append([i.text])
                if i.pos_ == 'CONJ':
                    pos_CONJ.append([i.text])
                if i.pos_ == 'VERB':
                    pos_VERB.append([i.text])
                if i.pos_ == 'PROPN':
                    pos_PROPN.append([i.text])
                if i.pos_ == 'PUNCT':
                    pos_PUNCT.append([i.text])
                if i.pos_ == 'INTJ':
                    pos_INTJ.append([i.text])
                if i.pos_ == 'ADJ':
                    pos_ADJ.append([i.text])
                if i.pos_ == 'ADP':
                    pos_ADP.append([i.text])
                if i.pos_ == 'NUM':
                    pos_NUM.append([i.text])
                if i.pos_ == 'SYM':
                    pos_SYM.append([i.text])
                if i.pos_ == 'PART':
                    pos_PART.append([i.text])
                    
            return df,pos_X, pos_NOUN, pos_DET,pos_ADV,pos_AUX,pos_PRON,pos_SCONJ,pos_CONJ,pos_VERB,pos_PROPN,pos_PUNCT,pos_INTJ,pos_ADJ,pos_ADP,pos_NUM,pos_SYM,pos_PART

        except Exception as e:
                print('pos_percent Error')
                print(e)
                pass       

    
path = 'out/pos/gads_data_pos_test.xlsx'
df = excel_to_df(path)
print(df)
#df = df.drop_duplicates(subset=['Dispatch No.'], keep=False)
#df = df.drop_duplicates(subset=['Details'], keep=False)
#print(df)

bow=[]
bow2 = []
no_special = []
stpwrds = []
no_stpwrds = []
lemma = []
stemm = []
pos_tagg = []
entities = []
morpho=[]
sent_bound = []
bi_grams = []
tri_grams = []
grams_lem = []
bi_grams_lem = []
tri_grams_lem = []
globalpos = []  
x=0
nlp = spacy.load("en_core_web_sm")

df_pos = pd.DataFrame(columns=['Text','Text',
                                'pos_X', 'pos_X',
                                'pos_NOUN', 'pos_NOUN',
                                'pos_DET','pos_DET',
                                'pos_ADV','pos_ADV',
                                'pos_AUX','pos_AUX',
                                'pos_PRON','pos_PRON',
                                'pos_SCONJ','pos_SCONJ',
                                'pos_CONJ','pos_CONJ',
                                'pos_VERB','pos_VERB',
                                'pos_PROPN','pos_PROPN',
                                'pos_PUNCT','pos_PUNCT',
                                'pos_INTJ','pos_INTJ',
                                'pos_ADJ','pos_ADJ',
                                'pos_ADP','pos_ADP',
                                'pos_NUM','pos_NUM',
                                'pos_SYM','pos_SYM',
                                'pos_PART','pos_PART'])
for i in df['Search Query']:
    print((x/9657)*100)
    
    a = QuatitativeNLP_FR.remove_spec_char(i)
    #ee= dict((y, x) for x, y in e)
    b, stopword, clean = QuatitativeNLP_FR.remove_stopwords(a)
    #print(ee)
    c = QuatitativeNLP_FR.lemmatisation(b)
    lemma.append(c)
    d = QuatitativeNLP_FR.stemming(c)
    stemm.append(d)
    e = QuatitativeNLP_FR.pos_tagging(b) 
    #pos_tagg.append(ee)   
    f = QuatitativeNLP_FR.entities(b)
    print('entities')
    print(f)
    entities.append(f)
    m = QuatitativeNLP_FR.morphology(b)
    print(m)
    morpho.append(m)
    s = QuatitativeNLP_FR.sentence_boundary(b)
    print(s)
    sent_bound.append(s)
    
    print(QuatitativeNLP_FR.pos_percent(b))
    count,global_pos=QuatitativeNLP_FR.pos_percent(b)
    phrase,pos_X, pos_NOUN, pos_DET,pos_ADV,pos_AUX,pos_PRON,pos_SCONJ,pos_CONJ,pos_VERB,pos_PROPN,pos_PUNCT,pos_INTJ,pos_ADJ,pos_ADP,pos_NUM,pos_SYM,pos_PART=QuatitativeNLP_FR.pos_count(b)
    df_pos=df_pos.append(pd.Series([phrase,len(phrase), 
                                    pos_X,len(pos_X), 
                                    pos_NOUN,len(pos_NOUN), 
                                    pos_DET,len(pos_DET),
                                    pos_ADV, len(pos_ADV),
                                    pos_AUX,len(pos_AUX),
                                    pos_PRON,len(pos_PRON),
                                    pos_SCONJ,len(pos_SCONJ),
                                    pos_CONJ,len(pos_CONJ),
                                    pos_VERB,len(pos_VERB),
                                    pos_PROPN,len(pos_PROPN),
                                    pos_PUNCT,len(pos_PUNCT),
                                    pos_INTJ,len(pos_INTJ),
                                    pos_ADJ,len(pos_ADJ),
                                    pos_ADP,len(pos_ADP),
                                    pos_NUM,len(pos_NUM),
                                    pos_SYM,len(pos_SYM),
                                    pos_PART,len(pos_PART)],
                                    index=['Text','Text',
                                            'pos_X', 'pos_X',
                                            'pos_NOUN', 'pos_NOUN',
                                            'pos_DET','pos_DET',
                                            'pos_ADV','pos_ADV',
                                            'pos_AUX','pos_AUX',
                                            'pos_PRON','pos_PRON',
                                            'pos_SCONJ','pos_SCONJ',
                                            'pos_CONJ','pos_CONJ',
                                            'pos_VERB','pos_VERB',
                                            'pos_PROPN','pos_PROPN',
                                            'pos_PUNCT','pos_PUNCT',
                                            'pos_INTJ','pos_INTJ',
                                            'pos_ADJ','pos_ADJ',
                                            'pos_ADP','pos_ADP',
                                            'pos_NUM','pos_NUM',
                                            'pos_SYM','pos_SYM',
                                            'pos_PART','pos_PART']),ignore_index=True)
    print(df_pos)
        #,columns=['Text','pos_X', 'pos_NOUN', 'pos_DET','pos_ADV','pos_AUX','pos_PRON','pos_SCONJ','pos_CONJ','pos_VERB','pos_PROPN','pos_PUNCT','pos_INTJ','pos_ADJ','pos_ADP','pos_NUM','pos_SYM','pos_PART']
        #c= Counter(([i.pos_ for i in df['Details']]))
    #count = Counter(([ii.pos_ for ii in globalpos]))
    print(count)
    #print(QuatitativeNLP_FR.multi_analysis(b)) 
    
    
   
    x=x+1
    
df_pos.to_excel('out/pos/gads_FULL_POS_Analysis.xlsx')   

    
    # TAG_MAP = {
    #     'X':{'pos':"X"},
    #     'NOUN':{'pos':"NOUN"},
    #     'DET':{'pos':"DET"},
    #     'ADV':{'pos':"ADV"},
    #     'AUX':{'pos':"AUX"},
    #     'PRON':{'pos':"PRON"},
    #     'SCONJ':{'pos':"SCONJ"},
    #     'CONJ':{'pos':"CONJ"},
    #     'VERB':{'pos':"VERB"},
    #     'PROPN':{'pos':"PROPN"},
    #     'PUNCT':{'pos':"PUNCT"},
    #     'INTJ':{'pos':"INTJ"},
    #     'ADJ':{'pos':"ADJ"},
    #     'ADP':{'pos':"ADP"},
    #     'NUM':{'pos':"NUM"},
    #     'SYM':{'pos':"SYM"},
    #     'PART':{'pos':"PART"}
    # }
    

