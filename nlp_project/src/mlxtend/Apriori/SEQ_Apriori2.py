from inspect import Traceback
import spacy
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.feature_extraction.text import CountVectorizer
import traceback
import sys

nlp = spacy.load("en_core_web_sm")


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


path = 'C:/Users/phili/OneDrive/Bureau/Google_Demo/Google_Demo/Prototype/data/no_dup_trans.xlsx'
df2 = excel_to_df(path)
print(df2)
corp = list(df2['no_special'].astype(str))
    

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
        x = vec.fit_transform(data)
        df = pd.DataFrame(x.toarray(), columns=vec.get_feature_names())
        df2 = pd.DataFrame(x.toarray(), columns=vec.get_feature_names())
        print(df2)
        #s_data = pd.Series([data])
        df['doc'] = data
        #df['doc'] = df['doc'].astype(str)
        
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
            df.to_excel('word_co_matrix_no_dup.xlsx') 
            print('out')
        return df, df2
    except Exception as e:
        print('doc_word_matrix error')
    
df1,ohe_df3 =doc_word_matrix(corp, 0, 0)
df3 = ohe_df3.columns.values.tolist()
df4 = set(df3)
dataFrame = ohe_df3.sum(axis = 0)
print(df4)
print('dataFrame')
print(dataFrame)

# nan = []
# for i in dataFrame.values:
#     if i > 1:
#         nan.append(1)
#     else:
#         nan.append(0)
# print('nan')
# print(nan)

# ohe_df3['nan'] = nan
ohe_df3['Transactions'] = df2['Transactions']
print(ohe_df3)


ohe_df3[ohe_df3 > 0] =1
print(ohe_df3)


freq_items2 = apriori(ohe_df3, min_support=0.005, use_colnames=True,verbose=1)
print(freq_items2.head(7))



rules2 = association_rules(freq_items2, metric="lift", min_threshold=0.5)
print(rules2.head())

rules2.to_excel('C:/Users/phili/OneDrive/Bureau/Google_Demo/Google_Demo/Prototype/data/token_rules_new99_no_dup_trans_only.xlsx')
freq_items2.to_excel('C:/Users/phili/OneDrive/Bureau/Google_Demo/Google_Demo/Prototype/data/token_freq_new99_no_dup_trans_only.xlsx')

plt.scatter(rules2['support'], rules2['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
# label = ohe_df3.index.name
# for x,y in zip(rules2['support'], rules2['confidence']):
#     plt.annotate(label, (x,y),textcoords="offset points", 
#                  xytext=(0,10), 
#                  ha='center')
    
plt.show()

plt.scatter(rules2['support'], rules2['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()

fit = np.polyfit(rules2['lift'], rules2['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules2['lift'], rules2['confidence'], 'yo', rules2['lift'], 
 fit_fn(rules2['lift']))


