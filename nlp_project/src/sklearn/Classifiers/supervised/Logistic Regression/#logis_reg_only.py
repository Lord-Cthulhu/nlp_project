#logis_reg_only
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





#Run Logistic Regression Classifier     
def run_logis_reg(learning_df, x, y,rstate):
    #Split Data into Train Test Partitions
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=rstate)
    
    #Logistic Regression Pipeline
    text_clf = Pipeline([
                        ('vect', CountVectorizer(max_features=150, min_df = 5)),
                        ('tfidf', TfidfTransformer()), # Vect+TfidfTransformer = TfidfVectorizer
                        ('clf', LogisticRegression(multi_class='ovr',  random_state=rstate, n_jobs=-2))
                         ])
    
    
    #Fit Data & Make predictions
    text_clf.fit(x_train, y_train)  
    print("sdfsdfsdf")
    print(x_train,y_train)
    
    #model=text_clf.fit(x_train, y_train) 
    lr_predictions = text_clf.predict(x_test)
    #print(model.summary())
    #Classification Performance
    lr_confusionmatrix = metrics.confusion_matrix(y_test,lr_predictions)
    lr_classificationreport = metrics.classification_report(y_test,lr_predictions)
    lr_accuracyscore = metrics.accuracy_score(y_test,lr_predictions)
    #pos_LR, neg_LR = metrics.class_likelihood_ratios(y_test, lr_predictions)
    
    
    #print("log_loss")
    #print(log_loss(y_test,lr_predictions))
    #Predictions
    lr_learning_predictions=[]
    
    for keywords in learning_df:
        learn_predictions = text_clf.predict(keywords)
        lr_learning_predictions.append(learn_predictions)
    #feature = text_clf.named_steps['clf'].get_feature_name()
    
    #XX = text_clf.named_steps['vect'].toarray()
    feature= text_clf.named_steps['vect'].get_feature_names()
    print("count vectorizer array")
    #print(XX)
    print(feature)
    
    
    
    intercept=text_clf.named_steps['clf'].intercept_
    coef=text_clf.named_steps['clf'].coef_
    nb_features=text_clf.named_steps['clf'].n_features_in_
    #feature_names=text_clf.named_steps['clf'].features_names_in_
    nb_iter=text_clf.named_steps['clf'].n_iter_
    #estimators=text_clf.named_steps['clf'].best_estimator_
    
    print("nb_features")
    print(nb_features)
    print("feature_names")
    #print(estimators)
    print("nb_iter")
    print(nb_iter)
    
    import math
    import scipy.stats as st
    from scipy.stats import normaltest
    # chi-squared test with similar proportions
    from scipy.stats import chi2_contingency
    from scipy.stats import chi2
    
    #mse = metrics.mean_squared_error(y_test,lr_predictions, multioutput='raw_values')
    #print(mse)
    for x in range(len(intercept)):
        print(x)
        coef_table = pd.DataFrame()
        coef_table['variable'] = feature
        coef_table['coef (B)'] = coef[x]
        #coef_table['std error'] = [st.sem(i) for i in coef[x]]
        hhh = [coef_table['coef (B)'].sem()]
        coef_table['exp(B)'] = [math.exp(i) for i in coef[x]]
        #coef_table['sig.'] = [normaltest(i,coef[x]) for i in coef[x]]
        #probability=text_clf.named_steps['clf'].predict_proba(coef[x])
        ggg = [normaltest(coef[x]) ]
        fff=   st.t.interval(alpha=0.95, df=len(coef_table['exp(B)'])-1, loc=np.mean(coef_table['exp(B)']), scale=st.sem(coef_table['exp(B)']))
        print('Coefficient Table')
        print(intercept[x])
        print(coef_table)
        print(fff)
        print(ggg)
        print(hhh)
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
    print(intercept)
    print(coef)
    
    return lr_predictions, lr_confusionmatrix,lr_classificationreport,lr_accuracyscore, lr_learning_predictions


dataset = pd.read_csv('train_svm.csv', sep=',', encoding='ISO-8859-1')
# cv_df, cv_df2=test_classifier(dataset)
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
df_learn= pd.read_csv('test_svm.csv', sep=',')
#df_learn= pd.read_csv('data_training.csv', sep=',')
learning_df = df_learn.values.tolist()
print('Logistic Regression')
lr_predictions, lr_confusionmatrix,lr_classificationreport,lr_accuracyscore, lr_learning_predictions = run_logis_reg( learning_df, x, y,9001)
print(lr_predictions)
print(lr_confusionmatrix)
print(lr_classificationreport)
print(lr_accuracyscore)
#print(learning_predictions)
print(len(lr_learning_predictions))