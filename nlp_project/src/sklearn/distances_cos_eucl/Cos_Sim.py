#2 vectors
#V=(v1,v2,…,v n),W=(w1,w2,…,wn)

#Dot product between 2 vectors
#V*W = Dot Product = (v1*w1)+(v2*w2)+(vn*wn)
#dot_prod = np.dot(V, W)

#Magnitude of vector
#for vector V=(v1,v2,…,v n) 
#||V|| = rac((v1)**2 + (v2)**2 + (vn)**2)

#Cosine (-1,1)
#0 (no similarity)
#1 (same)


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from openpyxl import Workbook

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

path = 'regex.xlsx'
df = excel_to_df(path)
df= df.sample(n=10000)
corpus = df['Search Query'].tolist()
corpus= list(dict.fromkeys(corpus))
# corpus = [
#     'google merch google ds google merch',
#     'google merch store gamba',
#     'android tablet',
#     'youtube google store',
#     'youtube video taco',
#     'video content',
#     'youtube content video',
#     'youtube merch store'
#     ]

def cos_similarity_matrix(corpus):
    #Init Vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    #Generate TFIDF Vector
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    print(tfidf_vectorizer.get_feature_names())
    # compute and print the cosine similarity matrix
    #cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim 

def top_cos_similarity(corpus):
    #Init Vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    #Generate TFIDF Vector
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    print(tfidf_vectorizer.get_feature_names())
    # compute and print the cosine similarity matrix
    #cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    df = pd.DataFrame(columns=["Text", "Similar_Text_1", "Cos_Similarity"])
    x = 0
    for s in cosine_sim:
        string_sim = list(s)
        #Remove Ref
        string_sim = np.delete(string_sim, x)
        #Index of max value
        sim_index=np.argwhere(s == np.amax(string_sim))[0]
        #Cos Sim Score
        sim_score=np.amax(string_sim)
        df.loc[len(df)] = [corpus[x], corpus[sim_index[0]], sim_score]
        x = x+1
    return df


def euclidean_dist(corpus):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    distance_matrix = euclidean_distances(tfidf_matrix)
    return distance_matrix

def top_euclidean_dist(corpus):
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    distance_matrix = euclidean_distances(tfidf_matrix)
    
    df2 = pd.DataFrame(columns=["Text", "Similar_Text", "Euclidean_Dist"])
    x = 0
    for s in distance_matrix:
        string_dist = list(s)
        #Remove Ref
        string_dist = np.delete(string_dist, x)
        #Index of max value
        dist_index=np.argwhere(s == np.amin(string_dist))[0]
        #Cos Sim Score
        dist_score=np.amin(string_dist)
        df2.loc[len(df2)] = [corpus[x], corpus[dist_index[0]], dist_score]
        x = x+1
    return df2

def all_similar_doc(corpus, sim_matrix, threshold):
    similar_doc = []
    for r in range(len(sim_matrix)):
        lst = [r+1+idx for idx, num in \
            enumerate(sim_matrix[r, r+1:]) if num >= threshold]
        for item in lst:
            similar_doc.append((corpus[r], corpus[item]) )
    return len(similar_doc), similar_doc

def all_distance_doc(corpus, distance_matrix, threshold):
    distance_doc = []
    for r in range(len(distance_matrix)):
        lst = [r+1+idx for idx, num in \
            enumerate(distance_matrix[r, r+1:]) if num <= threshold]
        for item in lst:
            distance_doc.append((corpus[r], corpus[item]) )
    return len(distance_doc), distance_doc

print(cos_similarity_matrix(corpus))
print(top_cos_similarity(corpus))
print(euclidean_dist(corpus))
print(top_euclidean_dist(corpus))

sim_matrix = cos_similarity_matrix(corpus)
threshold = 0.8
threshold2 = 0.2
similar_doc = all_similar_doc(corpus, sim_matrix, threshold)
distance_doc = all_distance_doc(corpus, sim_matrix, threshold2)
print(distance_doc)

#Activate Workbook
wb = Workbook() 
ws = wb.active 
#Append data to Workbook row by row 
for row in similar_doc[1]:
    ws.append(row) 
#Export most similar documents
wb.save('Similar_Doc_Cos.xlsx')

df1 = top_cos_similarity(corpus)
df2 = top_euclidean_dist(corpus)
df3 = pd.concat([df1, df2], axis=1)
df_cos_m = pd.DataFrame(cos_similarity_matrix(corpus))
print(df3)

#df_cos_m.to_excel('cos_sim_matrix.xlsx')
