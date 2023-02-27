from gensim.test.utils import common_dictionary, common_corpus, Dictionary
from gensim import corpora
from gensim.matutils import corpus2csc
from gensim.models import  TfidfModel
import numpy as np

dataset = ['drive car ',
           'drive car carefully',
           'student and university']

def tfidf_gensim(dataset):
    
    '''
    [Name] tfidf_gensism
    [Description] Get frequency data from a list of documents
    [Inputs] Dataset as a list ['this is a phrase', 'this is another phrase']
    [Outputs] Terms frequency per strings
              Terms frequency per corpora (TFIDF)
    '''
    try:
        #Generate Tokens
        tokens = [d.split() for d in dataset]
        print(tokens)
        unique_tokens = []
        for token in tokens:
            print(token)
            for i in token:
                print(i)
                if i not in unique_tokens:
                    unique_tokens.append(i)
        
        #Generate Vocab from Tokens
        vocab = corpora.Dictionary(tokens)
        
        #Generate Bag of Words
        bow = [corpora.Dictionary.doc2bow(vocab,document=doc) for doc in tokens]
        print(bow)

        #Get Word Frequency in Doc
        docs = []
        doc_freq = []
        
        for doc in bow :
            
            #docs.append([[vocab[id]] for id in doc])
            doc_freq.append([[vocab[id], freq] for id, freq in doc])
            print([[vocab[id], freq] for id, freq in doc])
        print(docs)

        #Generate TFIDF from BoW
        tfidf = TfidfModel(bow, smartirs='ntc')

        #Get Word Frequency in Corpora
        corp=[]
        corp_freq = []
        for doc in tfidf[bow]:
            #corp.append(vocab[id] for id in doc)
            corp_freq.append([[vocab[id], np.around(freq,decimals=2)] for id, freq in doc])
            print([[vocab[id], np.around(freq,decimals=2)] for id, freq in doc])
        print(corp)
        
       # doc2word = np.array(bow)
        #doc2word_mask = doc2word > 0
        term_doc_mat = corpus2csc(bow)
        print(term_doc_mat)
        term_doc_mat_mask = term_doc_mat >0
        term_term_mat = np.dot(term_doc_mat.T,term_doc_mat_mask)
        #term_term_mat = 1
        
        
        return doc_freq, corp_freq, term_term_mat , tokens, unique_tokens 
    except Exception as e:
        print('tfidf_gensim error')
        pass
        
        
doc_freq, corp_freq, term_term_mat, tokens, unique_tokens  = tfidf_gensim(dataset)

print(doc_freq)
print(corp_freq)
print(term_term_mat)