#svc_vs_logistic_porter 
from pstats import Stats
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats import bernoulli, binom
import seaborn as sns
from numpy.random import binomial, normal
import math
import scipy.stats as st
from scipy.stats import normaltest
# chi-squared test with similar proportions
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import statsmodels.api as sm
from sklearn.feature_selection import RFE, RFECV
import nltk
from nltk.stem.snowball import SnowballStemmer

#Run Linear SVC 
def run_linearsvc(learning_df, x, y,rstate):

    #Split Data into Train Test Partitions
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=rstate)
    
    #LinearSVC Pipeline
    text_clf = Pipeline([
                        ('tfidf', TfidfVectorizer()),
                         ('clf', LinearSVC(C=1,  random_state=rstate)), #squared hinge loss by default 1/2||w||^2  pour transformer la fonction loss='hinge' #max_iter=1000 default convergence issue
                         ])
    
    #Fit Data & Make predictions
    text_clf.fit(x_train, y_train)  
    predictions = text_clf.predict(x_test)
    
    #Classification Performance
    confusionmatrix = metrics.confusion_matrix(y_test,predictions)
    classificationreport = metrics.classification_report(y_test,predictions)
    accuracyscore = metrics.accuracy_score(y_test,predictions)
    
    #Predictions
    learning_predictions=[]
    
    for keywords in learning_df:
        learn_predictions = text_clf.predict(keywords)
        learning_predictions.append(learn_predictions)
        

    return predictions, confusionmatrix,classificationreport,accuracyscore, learning_predictions




#Run Logistic Regression Classifier     
def run_logis_reg(learning_df, x, y,rstate):
    
    # feature_array = np.array(tfidf.get_feature_names())
    # tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

    # n = 3
    # top_n = feature_array[tfidf_sorting][:n]
    
    
    #Split Data into Train Test Partitions
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=rstate)
    mx_features = 830
    #Logistic Regression Pipeline
    text_clf = Pipeline([
                        ('vect', CountVectorizer(max_features=mx_features, min_df = 5)),
                        ('tfidf', TfidfTransformer()), # Vect+TfidfTransformer = TfidfVectorizer
                        #("rfe" , RFECV(LogisticRegression(multi_class='ovr',  random_state=rstate,penalty="l2", n_jobs=-2 ), n_features_to_select=mx_features)),
                        ('clf', LogisticRegression(multi_class='ovr',  random_state=rstate,penalty="l2", n_jobs=-2 )) #C=1e9 doesn't converge. 
                         ])
    
    #Fit Data & Make predictions
    text_clf.fit(x_train, y_train)
    #model = text_clf.fit(x_train, y_train)    
    lr_predictions = text_clf.predict(x_test)

    
    #Classification Performance
    lr_confusionmatrix = metrics.confusion_matrix(y_test,lr_predictions)
    print(lr_confusionmatrix)
    lr_classificationreport = metrics.classification_report(y_test,lr_predictions)
    lr_accuracyscore = metrics.accuracy_score(y_test,lr_predictions)
    
    ##################
    #TBD
    #pos_LR, neg_LR = metrics.class_likelihood_ratios(y_test, lr_predictions)
    #print("log_loss")
    #print(log_loss(y_test,lr_predictions))
    ##################
    
    
    #Predictions
    lr_learning_predictions=[]
    for keywords in learning_df:
        learn_predictions = text_clf.predict(keywords)
        lr_learning_predictions.append(learn_predictions)
    #feature = text_clf.named_steps['clf'].get_feature_name()
    
    #Extract features
    c_vector_features= text_clf.named_steps['vect'].get_feature_names()
    print("count vectorizer features")
    print(c_vector_features)
    

    #Training Corpus
    train_c_vector_corpus = []
    for i in range(0, len(x_train)):
        text = re.sub('[^a-zA-Z]', ' ', x[i])
        text = text.lower()
        text = text.split()
        text = ' '.join(text)
        train_c_vector_corpus.append(text) 
    
    #Count Vectorizer Matrix
    train_c_vector_array = text_clf.named_steps['vect'].fit_transform(train_c_vector_corpus).toarray()
    train_c_vector_matrix = pd.DataFrame(data=train_c_vector_array, columns=c_vector_features[:len(c_vector_features)-7]) #[:len(c_vector_features)-7]
    print(y_train)
    train_c_vector_matrix_class=pd.get_dummies(y_train.values)
    train_count_vector_matrix = pd.concat([train_c_vector_matrix, train_c_vector_matrix_class], axis=1)
    train_count_vector_matrix.drop(columns=['info_g','info_p', 'nav', 'trans'])
    print("count vectorizer array")
    print(train_count_vector_matrix)
    
    #TFIDF Matrix
    train_tfidf_vector_array = text_clf.named_steps['tfidf'].fit_transform(train_c_vector_array).toarray()
    train_tfidf_vector_matrix = pd.DataFrame(data=train_tfidf_vector_array, columns=c_vector_features[:len(c_vector_features)-7]) #[:len(c_vector_features)-36] 36 8 10 7
    train_tfidf_vector_matrix = pd.concat([train_tfidf_vector_matrix , train_c_vector_matrix_class], axis=1)
    print("tfidf array")
    train_tfidf_vector_matrix = train_tfidf_vector_matrix.drop(columns=['info_g','info_p', 'nav', 'trans'])
    print(train_tfidf_vector_matrix)
    
        
    #Constants Values
    intercept=text_clf.named_steps['clf'].intercept_
    
    #Coefficient per Features
    coef=text_clf.named_steps['clf'].coef_
    
    #Number of features
    nb_features=text_clf.named_steps['clf'].n_features_in_
    print("Number of features")
    print(nb_features)
    
    #Iteration Convergence 
    nb_iter=text_clf.named_steps['clf'].n_iter_
    print("Iteration Convergence")
    print(nb_iter)
    
    #####################################################

 
    model = text_clf.named_steps['clf']
    #rfe = RFE(model,verbose=1)


    
    #######################
    #TBD
    #mse = metrics.mean_squared_error(y_test,lr_predictions, multioutput='raw_values')
    #print(mse)
    ########################
    
    #Coefficient Table
    for x in range(len(intercept)):
        print(x)
        coef_table = pd.DataFrame()
        coef_table['variable'] = c_vector_features
        coef_table['coef (B)'] = coef[x]
        
        #coef_table['std error'] = [st.sem(i) for i in coef[x]]
        
        
        hhh = [coef_table['coef (B)'].sem()]
        coef_table['exp(B)'] = [math.exp(i) for i in coef[x]]
        #coef_table['std_err_not_working'] = [math.sqrt(i) for i in coef_table['exp(B)']]
        #coef_table['prob_idk'] = [(i/(1+i)) for i in coef[x]]
        #coef_table['sig.'] = [normaltest(i,coef[x]) for i in coef[x]]
        #probability=text_clf.named_steps['clf'].predict_proba(coef[x])
        ggg = [normaltest(coef[x]) ]
        fff=   st.t.interval(alpha=0.95, df=len(coef_table['exp(B)'])-1, loc=np.mean(coef_table['exp(B)']), scale=st.sem(coef_table['exp(B)']))
        print(fff)
        print('Coefficient Table')
        print(intercept[x])
        print(coef_table)
        ##print(fff)
        ##print(ggg)
        ##print(hhh)
        #for y in coef[x]:
            #print(y)
        #print(probability)
        
    # df = pd.DataFrame({'x': y_test, 'y': lr_predictions})
    # df = df.sort_values(by='x')
    # from scipy.special import expit
    # sigmoid_function = expit(df['x'] * coef[0][0] + intercept[0]).ravel()
    # plt.plot(df['x'], sigmoid_function)
    # plt.scatter(df['x'], df['y'], c=df['y'], cmap='rainbow', edgecolors='b')
    # plt.show()
    
    #print("nagelkerke")    
    #print(Nagelkerke_Rsquare(learning_df,y_test,lr_predictions))
    #print(intercept)
    #print(coef)
    
    return lr_predictions, lr_confusionmatrix,lr_classificationreport,lr_accuracyscore, lr_learning_predictions, train_tfidf_vector_matrix

def get_features(dataset):
    print('Features')
    nltk.download('stopwords')
    corpus = []
    porter_corpus = []
    dataset.head()
    dataset.isnull().sum()

    print(dataset)
    x = dataset['ï»¿Search Query']
    y = dataset['Title']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=42)
    
    for i in range(0, len(dataset)):
        text = re.sub('[^a-zA-Z]', ' ', x[i])
        text = text.lower()
        text = text.split()
        
        #text = ps.stem(text)
        text = ' '.join(text)
        corpus.append(text) 
        
    for i in corpus:
        ii = i.split()
        ps = PorterStemmer()
        for iii in ii:
            pi = ps.stem(iii)
        ii = ''.join(pi)
        if pi not in porter_corpus:
            porter_corpus.append(ii) 
            
    
    cv = CountVectorizer(max_features = 1500, min_df = 5, binary=True)
    X = cv.fit_transform(corpus).toarray()
    print("count vectorizer array")
    print(X)
    print(X.shape)
    xy=cv.get_feature_names()
    
    #print(run_check_corr(X, xy,y))
    gst=cv.get_stop_words()
    print(gst)
    
    # cv2 = CountVectorizer(max_features = 1500, min_df = 5)
    # X2 = cv.fit_transform(porter_corpus).toarray()
    # xy2=cv2.get_feature_names()
    
    y = dataset.iloc[:, 1].values
    
    print(porter_corpus)
    print(len(xy))
    print(len(porter_corpus))
    # print(len(xy2))
    return xy


dataset = pd.read_csv('train_svm.csv', sep=',', encoding='ISO-8859-1')
# cv_df, cv_df2=test_classifier(dataset)
xy = get_features(dataset)
# print(cv_df)
# print(cv_df2)

#Training Set
#df_train = pd.read_csv('data_training2.csv', sep=',', encoding='ISO-8859-1')
df_train = pd.read_csv('train_svm.csv', sep=',', encoding='ISO-8859-1')
df_train.head()
df_train.isnull().sum()

print(df_train)
x = df_train['ï»¿Search Query']
y = df_train['Title']

#p_stemmer=PorterStemmer()
p_stemmer= SnowballStemmer(language='english')
p_s = []
for word in x:
   p_s.append(p_stemmer.stem(word))

df_train['Porter_Query'] = p_s
porter = df_train['Porter_Query']
print('Porter')
print(porter)

porter_train_df = pd.DataFrame()
porter_train_df['ï»¿Search Query'] = porter
porter_train_df['Title'] = df_train['Title']

p_x = porter_train_df['ï»¿Search Query']
p_y = porter_train_df['Title']

print("porter training dataframe")
print(porter_train_df)

#Learning Set 
df_learn= pd.read_csv('test_svm.csv', sep=',')
#df_learn= pd.read_csv('data_training.csv', sep=',')
learning_df = df_learn.values.tolist()


#Porter learning set

# x_l = df_learn['ï»¿Search Query']
# y_l = df_learn['Title']

# p_l_s = []
# for word in x_l:
#    p_l_s.append(p_stemmer.stem(word))

# df_learn['Porter_Query'] 

############
print('LinearSVC')
predictions, confusionmatrix,classificationreport,accuracyscore, learning_predictions = run_linearsvc( learning_df, p_x, p_y,9001)
print(predictions)
print(confusionmatrix)
print(classificationreport)
print(accuracyscore)
# #print(learning_predictions)
print(len(learning_predictions))    
####################   
    
print('Logistic Regression')
lr_predictions, lr_confusionmatrix,lr_classificationreport,lr_accuracyscore, lr_learning_predictions,train_tfidf_vector_matrix = run_logis_reg( learning_df, x, y,9001)
print(lr_predictions)
print(lr_confusionmatrix)
print(lr_classificationreport)
print(lr_accuracyscore)
#print(learning_predictions)
print(len(lr_learning_predictions))