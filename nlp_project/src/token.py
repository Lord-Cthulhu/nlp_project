#Dataframe
import numpy as np
import pandas as pd

#TFIDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

#NLTK
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

#Tensorflow
import tensorflow as tf
import tensorflow_hub as hub

#elMo
import spacy

#BERT 
from transformers import BertTokenizer

#Installation du modèle linguistique
nlp = spacy.load("en_core_web_sm")
test = ['hello','world','world']
df = pd.DataFrame(test)
#texts = [[word.lower() for word in text.split()] for text in df]
print(df)
text = ["Do pip list to make sure you have actually installed those versions"]
text2 = "Do pip list to make sure you have actually installed those versions"

def tfidf_vectorizer(df):
    try: 
        vectorizer = TfidfVectorizer(
                                        lowercase=True,
                                        max_features=100,
                                        max_df=3,
                                        min_df=1,
                                        ngram_range = (1,3),
                                        stop_words = "english"
                                    )

        vectors = vectorizer.fit_transform(df[0])
        print(vectors)

        feature_names = vectorizer.get_feature_names()
        print(feature_names)

        dense = vectors.todense()
        l_dense = dense.tolist()

        all_keywords = []
        return vectors, feature_names, l_dense, all_keywords 
    except Exception as e:
        print('tfidf_vectorizer fail')
        print(f'error: {e}'.format(e))
        pass          
    
vectors, feature_names, l_dense, all_keywords = tfidf_vectorizer(df)
print(vectors, feature_names, l_dense, all_keywords)    


  
def elmo_vectorizer(df):
    try: 
        #######TensorFlow Deprecated but functionnal#############
        #Forcer la V1 de tensorflow
        tf.compat.v1.disable_eager_execution()

        #Libraire tensorflow de référence
        elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
        
        embeddings = elmo(df, signature="default", as_dict=True)["elmo"]
        embeddings.shape
        embeddings.dtype
        
        print('embeddings.shape')
        print(embeddings.shape)
        
        print('embeddings.dtype')
        print(embeddings.dtype)
        
        # Extract features 
        embeddings = elmo(df, signature="default", as_dict=True)["elmo"]
        print(embeddings)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())
            
        # Average features
            return sess.run(tf.reduce_mean(embeddings,1))
        
        # #Tensorflow V2 (Actuel 30-03-2022)
        # hub_url = "https://tfhub.dev/google/elmo/3" <---La librairie fonctionne uniquement avec TF1
        # embed = hub.KerasLayer(hub_url, input_shape=[], dtype=tf.string, trainable=True)
        # embeddings = embed(df)
        # embeddings.shape
        # embeddings.dtype
        
    except Exception as e:
        print('elmo_vectorizer fail')
        print(f'error: {e}'.format(e))
        pass       

print(elmo_vectorizer(test))    
    
def lemmatization(text):
    # import spaCy's language model
    #nlp = spacy.load('en', disable=['parser', 'ner'])
    nlp = spacy.load("en_core_web_sm")
    output = []
    for i in text:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

print('lemmatize')
print(lemmatization(text))

def stemming(text): 
    return 1   

print('stemming')
print(stemming(text))

def bert_vectorizer(text):
    try: 
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        #Marking
        m_text = "[CLS] " + text + " [SEP]"
        
        #Tokenize Text
        text_token = tokenizer.tokenize(m_text)
        
        #Index token
        index_token = tokenizer.convert_tokens_to_ids(text_token)
        
        # Display the words with their indeces.
        #for tup in zip(tokenized_text, indexed_tokens):
        #    print('{:<12} {:>6,}'.format(tup[0], tup[1]))
        return m_text, text_token, index_token
        
        
    except Exception as e:
        print('bert_vectorizer fail')
        print(f'error: {e}'.format(e))
        pass       
    
print(bert_vectorizer(text2))    
    
def glove_vectorizer():
    try: 



        return 
    except Exception as e:
        print('tfidf_vectorizer fail')
        print(f'error: {e}'.format(e))
        pass
