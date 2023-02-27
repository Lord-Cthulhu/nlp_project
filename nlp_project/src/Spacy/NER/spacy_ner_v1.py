import pandas as pd
import numpy as np
import json
import spacy
from spacy.tokens import Doc, DocBin
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
import re
from nltk.corpus import stopwords
import random

import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
import multiprocessing
import subprocess
import sys


import spacy
from spacy.tokens import Span
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
import multiprocessing

import pandas as pd 
import numpy as np


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
            #print('csv failed')
            #print(f'error: {e}'.format(e))
            pass
        try:
            df = pd.read_excel(filepath)
            print('xlsx passed')
            return df
        except Exception as e:
            #print('xlsx failed')
            #print(f'error: {e}'.format(e))
            pass    
    except Exception as e:
        print('excel_to_df failed')
        print(f'error: {e}'.format(e))
        pass


def tokenize(df):
    try:
        nlp = spacy.load("en_core_web_sm")
        document = nlp(df)
        phrase = []
        for i in document:
            phrase.append(i.text) 
        return phrase
    except Exception as e:
        print('Tokenize Error')
        print(e)
        pass
    
# Inspired from Dr. W.J.B. Mattingly 
# https://github.com/wjbmattingly/ner_youtube/blob/main/lessons/04_05_under_spacys_hood.py

def training(model_name, list):
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(
                        min_count=5, #nb minimum keyword
                        window=2, #size ltstm
                        #size=500, #tester le size
                        sample=6e-5, #learning process
                        alpha=0.03, #learning process
                        min_alpha=0.0007, #learning process
                        workers = cores-1
                        )
    w2v_model.build_vocab(list)
    print(w2v_model.build_vocab(list))
    w2v_model.train(list, total_examples=w2v_model.corpus_count, epochs=30)
    w2v_model.save(f'word2vec/{model_name}.model')
    w2v_model.wv.save_word2vec_format(f'word2vec/word2vec_{model_name}.model')


def spacy_similarity(word):
    model = KeyedVectors.load_word2vec_format('word2vec/word2vec_model_w2v.txt', binary=False)
    #nlp = spacy.load("en_core_web_sm")
    
    ms = model.most_similar(
        positive=[word]
    )
    #words = [nlp.vocab.strings[w] for w in ms[0][0]]
    #distances = ms[2]
    print(ms)
    
  
def load_vectors(model, vectors):
    subprocess.run([sys.executable,
                    "-m",
                    "spacy",
                    "init-model",
                    "en",
                    model,
                    "--vectors-loc",
                    vectors])

model = 'model_w2v'
vectors = 'word2vec/word2vec_model_w2v.txt'

load_vectors(model, vectors)
  
##uncomment to generate word vector
# path = 'data/gads_data.xlsx'
# df = excel_to_df(path) 
# list = []
# for i in df['Search Query']:
#     token = tokenize(i)
#     list.append(token)
# print(list)
#training('model_w2v', list)

    
spacy_similarity("youtube")


# nlp = spacy.blank("en")
# ner = nlp.create_pipe("ner")
# ner.add_label("city")
# nlp.add_pipe(ner,name='city')
# nlp.to_disk('city_ner')
