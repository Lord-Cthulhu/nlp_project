import numpy as np
import pandas as pd
import itertools
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.feature_extraction.text import CountVectorizer

dataset = ['google dino dino',
           'google dino game',
           'dino google clothing', 
           'google clothing',
           'dino chrome']

def word_co_matrix(data, plot, out):
    '''
    [Name] 
    word_coocc_matrix
    
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
        #Transform documents to tokens
        tokens = [token.split() for token in data]
        
        #Generate unique tokens 
        unique_tokens = []
        for token in tokens:
            print(token)
            for i in token:
                print(i)
                if i not in unique_tokens:
                    unique_tokens.append(i)
        
        ##################################################################################
        #######https://localcoder.org/co-occurrence-matrix-from-nested-list-of-words######            
        #Set Words Ids in dict
        word_ids = dict(zip(unique_tokens, range(len(unique_tokens))))
        
        #Match Words ids with words from documents
        doc_ids = [np.sort([word_ids[w] for w in doc if w in word_ids]).astype('uint32') for doc in tokens]
        
        #https://stackoverflow.com/questions/39214810/zip-unknown-number-of-lists-with-python-for-more-than-one-list
        row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in doc] for i, doc in enumerate(doc_ids)]))
        print(row_ind)
        print(col_ind)
        
        
        #https://numpy.org/doc/stable/reference/generated/numpy.ones.html
        #Technique d'optimisation
        data = np.ones(len(row_ind), dtype='uint32') 
        print(data)
        
        #
        max_w_id = max(itertools.chain(*doc_ids)) + 1
        print(max_w_id)
    
        #Generate Co-Occurence Matrix
        docs_words_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(doc_ids), max_w_id))  # efficient arithmetic operations with CSR * CSR
        
        #Transpose Matrix
        words_co_matrix = docs_words_matrix.T * docs_words_matrix  # multiplying docs_words_matrix with its transpose matrix would generate the co-occurences matrix
        words_co_matrix.setdiag(0)
        ####################################################################################
        
        
        #Get Dense matrix for pandas dataframe
        dense_matrix = words_co_matrix.todense()
        
        #Generate a dataframe from Numpy array
        co_matrix = pd.DataFrame(dense_matrix, columns=unique_tokens, index = unique_tokens)
        
        if plot == 1:
            #Generate Matplotlib plot
            fig = plt.figure()
            ax = fig.add_subplot()
            #plt.style.use("fivethirtyeight")
            cax = ax.matshow(co_matrix, interpolation='nearest', cmap=cm.coolwarm)
            fig.colorbar(cax)

            #['']+ Col Name to align labels with ticks
            ax.set_xticklabels(['']+unique_tokens)
            ax.set_yticklabels(['']+unique_tokens)

            #plt.matshow(cooc_matrix)
            plt.show()
            
        if out == 1:
            co_matrix.to_excel('word_co_matrix.xlsx') 
            print('out')
        
        return dense_matrix
    except Exception as e:
        print('word_co_matrix error')


word_co_matrix(dataset, 1, 0)

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
            df.to_excel('word_co_matrix.xlsx') 
            print('out')
        return df
    except Exception as e:
        print('doc_word_matrix error')
    
print(doc_word_matrix(dataset, 0, 0))
    
    
    
    
#Miyamoto 2008 describes computation of a number of statistical properties of the co-occurrence matrix first, followed by computation of the Haralick texture features.
#R=∑Ngi=1∑Ngj=1P(i,j) = sum of all elements of co-occurrence frequency matrix
#p(i,j)=P(i,j)R = co-occurence probability matrix
#px(i)=∑Ngj=1p(i,j) = ith entry in the marginal-probability matrix obtained by summing the rows of p(i,j)
#py(j)=∑Ngi=1p(i,j) = jth entry in the marginal-probability matrix obtained by summing the columns of p(i,j).
#http://earlglynn.github.io/kc-r-users-jupyter/Co-occurrence%20Matrix.html