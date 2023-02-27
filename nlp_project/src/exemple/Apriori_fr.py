
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import itertools
from scipy.sparse import csr_matrix
from matplotlib.pyplot import cm
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer



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

class QuatitativeNLP_FR: 
    '''
    [Token: ]
    [Stemming: ]
    [PoS Tagging: ]
    [ ]
    '''

    nlp = spacy.load("fr_core_news_md")
    
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
        
        df = re.sub(r'[^a-zA-Z0-9\s]', '', df)
        df = re.sub(' +', ' ',df)
        print(df)
        document = nlp(df)
        
        
        spec_char = ['!','@','#','$', '%','?','&','*','(',')','_', '-', '=','+','{','}','[',']','|','<','>',"\n", "'", ".", ':','  ', '//', '...', '..','....','.....','/-', ';','    ', '.......','........', '«', '»', ':-)', '))' ]
        clean = []
        for i in document:
            
            if i.text not in spec_char:
                clean.append(i.text)
                
        phrase = ' '.join(word for word in clean)
        phrase = re.sub(r'[^a-zA-Z0-9\s]', ' ', phrase)
        print(phrase)
        return phrase
    
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
        
    def stemming(df):
        try:
            stemmer = SnowballStemmer(language='french')
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
        
def remove_stopwords(df):
    try:
        df = df.lower()
        document = nlp(df)
        stpwrds = set(stopwords.words('french'))
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

def doc_word_matrix(data, plot, out):
    '''
    [Name] 
    doc_word_coocc_matrix
    
    [Description] 
    Generate Word Co-Occurence Matrix
    
    [Inputs] 
    Dataset as a list ['this is a phrase', 'this is another phrase']
    
    [Outputs] 
    word_coocc_matrix(data, plot, out)
    data = Inputs
    plot = 1 to show plot
    out = 1 to export in .xlsx
    '''    
    try:
        vec = CountVectorizer()
        print("D")
        print(data)
        
        data = np.array(data)

        X = vec.fit_transform(data)
        print(vec.get_feature_names())
        
        print("x")
        print(X)
        df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
        df2 = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
        print("df")
        print(df)
        print("df2")
        print(df2)
        #s_data = pd.Series([data])
        df['doc'] = data
        
        #Mettre le doc à la première position 
        df.insert(0, 'doc', df.pop('doc'))
        

        if plot == 1:
            #Generate Matplotlib plot
            fig = plt.figure()
            ax = fig.add_subplot()
            #plt.style.use("fivethirtyeight")
            cax = ax.matshow(df.values, interpolation='nearest', cmap=cm.coolwarm)
            fig.colorbar(cax)

            #['']+ Col Name to align labels with ticks
            #ax.set_xticklabels(['']+df.values['doc'])
            #ax.set_yticklabels(['']+'a')

            #plt.matshow(cooc_matrix)
            plt.show()
        if out == 1:
            df.to_excel('word_co_matrix_stag_ann1.xlsx') 
            print('out')
        return df,df2
    except Exception as e:
        print('doc_word_matrix error')

path = 'stag_ann1.csv'
df = excel_to_df(path)
print(df.head(7))

#df['ExigencesParticulieres'] = df['ExigencesParticulieres'].map(lambda x: re.sub(r'\W+', '', x))
df['ExigencesParticulieres'] = df['ExigencesParticulieres'].str.replace("[àáâãäå]", 'a')
df['ExigencesParticulieres'] = df['ExigencesParticulieres'].str.replace('[èéêë]', 'e')
df['ExigencesParticulieres'] = df['ExigencesParticulieres'].str.replace('[ìíîï]', 'i')
df['ExigencesParticulieres'] = df['ExigencesParticulieres'].str.replace("[òóôõö]", 'o')
df['ExigencesParticulieres'] = df['ExigencesParticulieres'].str.replace("[ùúûü]", 'u')
df['ExigencesParticulieres'] = df['ExigencesParticulieres'].str.replace('[ýÿ]', 'y')
df['ExigencesParticulieres'] = df['ExigencesParticulieres'].str.replace('[^\w\s]', '')
data = df['ExigencesParticulieres'].to_list()
print(data)
   
df_,ohe_df2 =doc_word_matrix(data, 0, 0)
print(df_)
print(ohe_df2)
df3 = ohe_df2.columns.values.tolist()
df4 = set(df3)
dataFrame = ohe_df2.sum(axis = 1)
print(df4)
print('dataFrame')
print(dataFrame)

# nan = []
# for i in dataFrame.values:
#     if i > 1:
#         nan.append(0)
#     else:
#         nan.append(1)
#print('nan')
#print(nan)

#ohe_df2['nan'] = nan

print(ohe_df2)
#transform Frequency to binary 
ohe_df2[ohe_df2 > 0] =1

nlp = spacy.load("fr_core_news_md")

# bow = []
# x=0
# for i in df['Search Query']:
#     print((x/55001)*100)
#     yyy = QuatitativeNLP_FR.tokenize(i)
#     bow.extend(yyy)
    
#     a = QuatitativeNLP_FR.remove_spec_char(i)
#     x+=1

# print('bow')
# print(bow)


## Use this to read data directly from github
# df = pd.read_csv('retail_dataset.csv', sep=',')## Use this to read data from the csv file on local system.
# df.head(10)

# items = set()
# for col in df:
#     items.update(df[col].unique())
    
# print(items)

# itemset = set(items)
# encoded_vals = []
# for index, row in df.iterrows():
#     rowset = set(row) 
#     labels = {}
#     uncommons = list(itemset - rowset)
#     commons = list(itemset.intersection(rowset))
#     for uc in uncommons:
#         labels[uc] = 0
#     for com in commons:
#         labels[com] = 1
#     encoded_vals.append(labels)
# encoded_vals[0]
# ohe_df = pd.DataFrame(encoded_vals)

print(ohe_df2)
#ohe_df2.fillna(0,inplace=True)
#ohe_df2.to_excel('wtf.xlsx')
freq_items = apriori(ohe_df2, min_support=0.1, use_colnames=True, verbose=1)
#print(freq_items.head(7))

rules = association_rules(freq_items, metric="confidence", min_threshold=0.15)
print(rules.head())

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
label = ohe_df2.index.name
# for w,z,x,y in zip(rules['antecedents'].str,rules['consequents'],rules['support'], rules['confidence']):
#     plt.annotate((w,z), (x,y),textcoords="offset points", 
#                  xytext=(0,10), 
#                  ha='center')
plt.show()

plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()

fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift']))

rules.to_excel('token_stagann1.xlsx')
freq_items.to_excel('token_freq_stag_ann1.xlsx')


