#Dataframe
import numpy as np
import pandas as pd

#TFIDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

#NLTK
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords') # Remove Stopwords
nltk.download('averaged_perceptron_tagger') #Analyse Morphologique (PoS)
nltk.download('maxent_ne_chunker') #Analyse syntaxique (Chunking)
nltk.download('words')  #Analyse syntaxique (Chunking)

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



def pre_pos_tag(text):
    '''
        [Abbreviation 	Meaning
        CC 	    coordinating conjunction
        CD 	    cardinal digit
        DT 	    determiner
        EX 	    existential there
        FW 	    foreign word
        IN 	    preposition/subordinating conjunction
        JJ 	    This NLTK POS Tag is an adjective (large)
        JJR 	adjective, comparative (larger)
        JJS 	adjective, superlative (largest)
        LS 	    list market
        MD 	    modal (could, will)
        NN      noun, singular (cat, tree)
        NNS 	noun plural (desks)
        NNP 	proper noun, singular (sarah)
        NNPS 	proper noun, plural (indians or americans)
        PDT 	predeterminer (all, both, half)
        POS 	possessive ending (parent s)
        PRP 	personal pronoun (hers, herself, him, himself)
        PRP$ 	possessive pronoun (her, his, mine, my, our )
        RB 	    adverb (occasionally, swiftly)
        RBR 	adverb, comparative (greater)
        RBS 	adverb, superlative (biggest)
        RP 	    particle (about)
        TO 	    infinite marker (to)
        UH 	    interjection (goodbye)
        VB 	    verb (ask)
        VBG 	verb gerund (judging)
        VBD 	verb past tense (pleaded)
        VBN 	verb past participle (reunified)
        VBP 	verb, present tense not 3rd person singular(wrap)
        VBZ 	verb, present tense with 3rd person singular (bases)
        WDT 	wh-determiner (that, what)
        WP 	    wh- pronoun (who)
        WRB 	wh- adverb (how)]
    '''
    words = word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    return tagged_words


def chunking(tagged_words): 
    entities = nltk.chunk.ne_chunk(tagged_words)
    return entities


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
        #######TensorFlow########
        # Forcer la V1 de tensorflow
        tf.compat.v1.disable_eager_execution()

        # Libraire tensorflow de référence
        elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
        
        embeddings = elmo(df, signature="default", as_dict=True)["elmo"]
        embeddings.shape
        embeddings.dtype
        
        print('embeddings.shape')
        print(embeddings.shape)
        
        print('embeddings.dtype')
        print(embeddings.dtype)
        
        # Extract features 
        embeddings = elmo(df, signature="default", as_dict=True)["elmo"] #With the tokens signature, the module takes tokenized sentences as input. https://tfhub.dev/google/elmo/3
        print(embeddings)
        
        # Executer la session avec TFV1
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

def stemming(text,a):
    try:
        if a == 'p':
            #https://www.nltk.org/howto/stem.html
            porter = PorterStemmer()
            #C.J. van Rijsbergen, S.E. Robertson and M.F. Porter, 1980. New models in probabilistic information retrieval. London: British Library. (British Library Research and Development Report, no. 5587).
            #Alternative fonctionnelle 
            #snowball = SnowballStemmer('porter', ignore_stopwords=False)
            token = word_tokenize(text)
            p_stem=[]
            for i in token:
                p = porter.stem(i)
                p_stem.append(' '.join(p))
            return p_stem
            
            
        if a == 'l':    
            #L'algorithme le plus agressif 
            lancaster = LancasterStemmer()
            #snowball = SnowballStemmer('lancaster', ignore_stopwords=False)
            token = word_tokenize(text)
            l_stem=[]
            for i in token:
                l = lancaster.stem(i)
                l_stem.append(' '.join(l))
            return l_stem
        
        
        if a == 's':
            #TBD
            snowball = SnowballStemmer('english', ignore_stopwords=True)
            token = word_tokenize(text)
            s_stem=[]
            for i in token:
                s = snowball.stem(i)
                s_stem.append(' '.join(s))
            return s_stem 
        
        else:
            porter = PorterStemmer()
            lancaster = LancasterStemmer()
            snowball = SnowballStemmer('english', ignore_stopwords=True)
            
            token = word_tokenize(text)
            
            p_stem=[]
            l_stem=[]
            s_stem=[]
            for i in token:
                p = porter.stem(i)
                l = lancaster.stem(i)
                s = snowball.stem(i)
                p_stem.append(' '.join(p))
                l_stem.append(' '.join(l))
                s_stem.append(' '.join(s))
            return p_stem, l_stem, s_stem 
    except Exception as e:
        print('stemming fail')
        print(f'error: {e}'.format(e))
        pass 
        

print('stemming')
print(stemming(lemmm[0],"p")) 

def bert_vectorizer(text, fast):
    try: 
        if fast == 0:
            #https://huggingface.co/docs/transformers/model_doc/bert
            #https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
            
            #Marking [CLS],[SEP],[UNK],[PAD]
            m_text = "[CLS] " + text + " [SEP]"
            
            #Tokenize Text
            token = tokenizer.tokenize(m_text)
            
            #Index token
            index_token = tokenizer.convert_tokens_to_ids(token)
            return m_text, token, index_token
        if fast == 1:
            #https://huggingface.co/docs/transformers/model_doc/bert
            fast_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case = True)
            
            #Marking [CLS],[SEP],[UNK],[PAD]
            m_text = "[CLS] " + text + " [SEP]"
            
            #Tokenize Text
            f_token = fast_tokenizer.tokenize(m_text)
            
            #Index token
            f_index_token = fast_tokenizer.convert_tokens_to_ids(f_token)
            
            return m_text, f_token,f_index_token
            # Display the words with their indeces.
            #for tup in zip(tokenized_text, indexed_tokens):
            #    print('{:<12} {:>6,}'.format(tup[0], tup[1]))
    except Exception as e:
        print('bert_vectorizer fail')
        print(f'error: {e}'.format(e))
        pass       
    
print(bert_vectorizer(lemmm[0],1))    
    
def glove_vectorizer():
    try: 



        return 
    except Exception as e:
        print('tfidf_vectorizer fail')
        print(f'error: {e}'.format(e))
        pass
