#Data
import pandas as pd
import numpy as np

#Regexp
import re

#NLTK
#Feature Transformation
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#sklearn
#Vectorisation, Pipeline, Data Split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#Statsmodels
#Logistic regression, AIC, BIC, T-Test, Pseudo R2
import statsmodels.api as sm
from scipy.stats.distributions import chi2
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from sklearn.feature_selection import RFE

###https://stackoverflow.com/questions/69162676/statsmodels-metric-for-comparing-logistic-regression-models
def likelihood_ratio(ll0, ll1):
    return -2 * (ll0-ll1)

def lrtest(fitted_model0: BinaryResultsWrapper, 
           fitted_model1: BinaryResultsWrapper):
    
    L0, L1 = fitted_model0.llf, fitted_model1.llf
    df0, df1 = fitted_model0.df_model, fitted_model1.df_model
    
    chi2_stat = likelihood_ratio(L0,L1)
    p = chi2.sf(chi2_stat, df1-df0)

    return (chi2_stat, p)
###

def transformation_pipeline(learning_df, x, y,rstate):
    #Data Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=rstate)
    print(x_train)
    
    #Matrix Transformation Pipeline 
    text_clf = Pipeline([
                        ('vect', CountVectorizer(max_features=75, min_df = 5)),
                        ('tfidf', TfidfTransformer()), # Vect+TfidfTransformer = TfidfVectorizer
                        ])
    #Training Corpus
    train_c_vector_corpus = []
    for i in range(0, len(x_train)):
        text = re.sub('[^a-zA-Z]', ' ', x[i])
        text = text.lower()
        text = text.split()
        text = ' '.join(text)
        train_c_vector_corpus.append(text) 
    
    #Count Vectorizer Matrix
    train_c_vector_array = text_clf.named_steps['vect'].fit_transform(train_c_vector_corpus).toarray() #Fitted Data
    c_vector_features= text_clf.named_steps['vect'].get_feature_names() #CV Top Features
    train_c_vector_matrix = pd.DataFrame(data=train_c_vector_array, columns=c_vector_features) #Vector Matrix
    #print(y_train)
    train_c_vector_matrix_class=pd.get_dummies(y_train.values)  
    train_count_vector_matrix = pd.concat([train_c_vector_matrix, train_c_vector_matrix_class], axis=1)
    train_count_vector_matrix.drop(columns=['info_g','info_p', 'nav', 'trans'])
    
    print('count vectorizer features')
    print(c_vector_features)
    
    print("count vectorizer array")
    print(train_count_vector_matrix)
    
    #TFIDF Matrix
    train_tfidf_vector_array = text_clf.named_steps['tfidf'].fit_transform(train_c_vector_array).toarray() #Fitted Data
    train_tfidf_vector_matrix = pd.DataFrame(data=train_tfidf_vector_array, columns=c_vector_features)
    train_tfidf_vector_matrix = pd.concat([train_tfidf_vector_matrix , train_c_vector_matrix_class], axis=1)
    
    print("tfidf array")
    print(train_tfidf_vector_matrix)
    
    return train_tfidf_vector_matrix
    
    
#Logistic Regression Model
def statsmod_logreg(train_tfidf_vector_matrix):
    print("Statsmodel Logistic Regression")
    #List Columns
    cols = train_tfidf_vector_matrix.columns.to_numpy().tolist() 
    print("Columns")
    print(cols)
    X_train = train_tfidf_vector_matrix[cols]
    
    #Cat com
    x_train_col = X_train.iloc[: , :-6]
    print("Training Data")
    print(x_train_col)
    
    #Categories Binary Dataframe
    nc = 5
    cat = X_train.iloc[: , -nc:]
    print("cat")
    print(cat)
    
    #Get Categories
    columns = []
    for col in cat.columns:
        columns.append(col)
        
    print("Columns")
    print(columns)
    print(list(cat["com"]))
    print(list(cat["info_g"]))
    print(list(cat["info_p"]))
    print(list(cat["nav"]))
    print(list(cat["trans"]))
    log_reg2  =sm.Logit(list(cat["nav"]),x_train_col).fit_regularized(maxiter = 500, L1_wt=0.0) #L1 à 0 inférence standard ne fonctionne pas 
    print(log_reg2.summary())
    
    # for col in columns:
    #     #y_cat = list(cat.iloc[:,i])
    #     y_cat = list(cat[col])
    #     print("y_cat")
    #     print(y_cat)
    #     #log_reg2  =sm.Logit(list(cat["info_p"]),x_train_col, sm.add_constant()).fit_regularized(maxiter = 150, L1_wt=0.0)
    #     log_reg2  =sm.Logit(list(cat["com"]),x_train_col).fit_regularized(maxiter = 500, L1_wt=0.0) #L1 à 0 inférence standard ne fonctionne pas 
    #     print(log_reg2.summary())
    #     r = np.zeros_like(log_reg2.params)
    #     print(r)
    #     #lrloglike = log_reg2.loglike(log_reg2.params)
    #     print(1)
    #     #print(lrloglike)
    #     #lrloglikeobs = log_reg2.loglikeobs(log_reg2.params)
    #     print(2)
    #     #print(lrloglikeobs)
    #     #lrscore = log_reg2.score(log_reg2.params)
    #     print(3)
    #     #print(lrscore)
    #     #lrscorefactor = log_reg2.score_factor(log_reg2.params)
    #     print(4)
    #     #print(lrscorefactor)
    #     #lrscoreobs = log_reg2.score_obs(log_reg2.params)
    #     print(5)
    #     #print(lrscoreobs)
    #     #lrinformation = log_reg2.information(log_reg2.params)
    #     print(6)
    #     #print(lrinformation)
    #     #lrhessian = log_reg2.hessian(log_reg2.params)
    #     print(7)
    #     #print(lrhessian)
    #     #lrhessianfactor = log_reg2.hessian_factor(log_reg2.params)
    #     print(8)
    #     #print(lrhessianfactor)
    #     #t_test = log_reg2.t_test(r)
    #     #print(t_test)
        
    #     #pred =sm.Logit(list(cat["trans"]),x_train_col).predict()
    #     #print(pred)
        
        
    com = X_train.iloc[: , :-2]
    print("com")
    print(com)
    
    y_train = com.iloc[:,-1:]
    print("Y_train")
    print(y_train)

    print("info_p")
    print(list(y_train['info_p']))
    log_reg  =sm.Logit(list(y_train['info_p']),x_train_col).fit_regularized(maxiter = 1500)
    #llf = log_reg.llr() #Max log_likelihood
    #llnull = log_reg.pvalues
    #llr = log_reg.llf()
    #llpvalue = log_reg.llr_pvalue()
    
    print(log_reg.summary())
    #print(log_reg.resid_dev())
    print(log_reg.pvalues)
    #print(log_reg.aic())
    #print(log_reg.bic())
    #print(log_reg.llr_pvalue())
    #print(llf)
    #print(llnull)
    #print(llr)
    #print(llpvalue)
    
    #rfe = RFE(log_reg , 3)
    #fit = rfe.fit(y_train,x_train_col)
    #print(fit.ranking_)
    list(cat["nav"]),x_train_col
    
    log_mod0 = sm.Logit(list(cat["nav"]),x_train_col).fit_regularized(maxiter = 1500)
    log_mod1 = sm.Logit(list(cat["info_p"]),x_train_col).fit_regularized(maxiter = 1500)

    chi2_stat, p = lrtest(log_mod0, log_mod1)
    
    print("rfe")
    print(chi2_stat)
    print(p)
    

    return X_train, log_reg


dataset = pd.read_csv('train_svm.csv', sep=',', encoding='ISO-8859-1')
# cv_df, cv_df2=test_classifier(dataset)
#xy = get_features(dataset)
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

p_stemmer=PorterStemmer()
p_s = []
for word in x:
   p_s.append(p_stemmer.stem(word))

df_train['Porter_Query'] = p_s
xx = df_train['Porter_Query']
print('xx')
print(xx)

#Learning Set 
df_learn= pd.read_csv('test_svm.csv', sep=',')
#df_learn= pd.read_csv('data_training.csv', sep=',')
learning_df = df_learn.values.tolist()
rstate=63
train_tfidf_vector_matrix = transformation_pipeline(learning_df, x, y,rstate)
X_train, log_reg=statsmod_logreg(train_tfidf_vector_matrix)

print(X_train)



# loading the testing dataset  
df_test_ = pd.read_csv('True_pred_2.csv')
print(df_test_) 
# defining the dependent and independent variables
Xtest = df_test_['Search Query']
ytest = df_test_['Category']
Xtest  = np.squeeze(np.asarray(Xtest))  
# performing predictions on the test datdaset
yhat = log_reg.predict(Xtest)
prediction = list(map(round, yhat))
  
# comparing original and predicted values of y
#print('Actual values', list(ytest.values))
print('Predictions :', prediction)