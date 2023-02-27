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

#from sklearn.metrics import class_likelihood_ratios

#Test multiple models
def test_classifier(dataset):
    nltk.download('stopwords')
    corpus = []
    dataset.head()
    dataset.isnull().sum()

    print(dataset)
    x = dataset['ï»¿Search Query']
    y = dataset['Title']


    for i in range(0, len(dataset)):
        #text = re.sub('[^a-zA-Z]', '', dataset['Text'][i])
        text = re.sub('[^a-zA-Z]', '', x[i])
        text = text.lower()
        text = text.split()
        ps = PorterStemmer()
        text = ''.join(text)
        corpus.append(text)
    
    # creating bag of words model
    cv = CountVectorizer(max_features = 1500)
    
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values


    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                            ngram_range=(1, 2), 
                            stop_words='english')
    # We transform each complaint into a vector
    features = tfidf.fit_transform(x).toarray()
    labels = y

    models = [
        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        LinearSVC(random_state=42),
        LogisticRegression(multi_class='ovr', random_state=42)
    ]


    CV = 5
    entries_accuracy = []
    entries_b_accuracy = []
    entries_a_precision = []
    entries_precision = []
    entries_recall = []

    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring="accuracy", cv=CV)
        balanced_accuracies = cross_val_score(model, features, labels, scoring="balanced_accuracy", cv=CV)

    
    for fold_idx, accuracy in enumerate(accuracies):
        entries_accuracy.append((model_name, fold_idx, accuracy))
        
        
    for fold_idx,balanced_accuracy in enumerate(balanced_accuracies):
        entries_b_accuracy.append((model_name, fold_idx, balanced_accuracy))
        
    cv_df = pd.DataFrame(entries_accuracy, columns=['model_name', 'fold_idx', 'accuracy'])
    cv_df2 = pd.DataFrame(entries_b_accuracy, columns=['model_name', 'fold_idx', 'balanced_accuracy'])
    return cv_df, cv_df2

#Run Linear SVC 
def run_linearsvc(learning_df, x, y,rstate):

    #Split Data into Train Test Partitions
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=rstate)
    
    #LinearSVC Pipeline
    text_clf = Pipeline([
                        ('tfidf', TfidfVectorizer()),
                         ('clf', LinearSVC(C=1,  random_state=rstate)), #squared hinge loss by default 1/2||w||^2  pour transformer la fonction loss='hinge'
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
                        ("rfe" , RFECV(LogisticRegression(multi_class='ovr',  random_state=rstate,penalty="l2", n_jobs=-2 ), n_features_to_select=mx_features)),
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
    train_c_vector_matrix = pd.DataFrame(data=train_c_vector_array, columns=c_vector_features)
    print(y_train)
    train_c_vector_matrix_class=pd.get_dummies(y_train.values)
    train_count_vector_matrix = pd.concat([train_c_vector_matrix, train_c_vector_matrix_class], axis=1)
    train_count_vector_matrix.drop(columns=['info_g','info_p', 'nav', 'trans'])
    print("count vectorizer array")
    print(train_count_vector_matrix)
    
    #TFIDF Matrix
    train_tfidf_vector_array = text_clf.named_steps['tfidf'].fit_transform(train_c_vector_array).toarray()
    train_tfidf_vector_matrix = pd.DataFrame(data=train_tfidf_vector_array, columns=c_vector_features)
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
    
    x_rfe = train_tfidf_vector_matrix.iloc[: , :-2]
    y_rfe = train_tfidf_vector_matrix.iloc[:,-1:]
    print('x_rfe,y_rfe')
    print(x_rfe,y_rfe)
    model = text_clf.named_steps['clf']
    #rfe = RFE(model,verbose=1)
    rfe = RFECV(model,verbose=1)
    rfe = rfe.fit(x_rfe, y_rfe.values.ravel())
    print('rfe support')
    print(rfe.support_)
    print('rfe ranking')
    print(rfe.ranking_)
    print('rfe estimator')
    print(rfe.estimator_)
    print('rfe features')
    print(rfe.n_features_)
    print('rfe features in')
    print(rfe.n_features_in_)
    print('rfe importance getter')
    print(rfe.importance_getter)
    
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest
    chi2_fe_selection = SelectKBest(score_func=chi2, k='all')
    chi2_fit = chi2_fe_selection.fit(x_rfe,y_train)
    print('chi2 scores')
    print(chi2_fit.scores_[0])
    print(chi2_fit.pvalues_)
    print(chi2_fit.n_features_in_)
    
    feature_selection_results = pd.DataFrame()
    #feature_selection_results["Features"] = x_rfe
    feature_selection_results["Support"] = [rfe.support_]
    feature_selection_results["Ranking"] = [rfe.ranking_]
    feature_selection_results["Chi2"] = [chi2_fit.scores_[0]]
    
    print("feature_selection_results")
    print(feature_selection_results)
    
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


def run_check_vif(data, variables,y):
    print(data)
    df = pd.DataFrame(data=data,columns=variables)
    print(df)
    variables = df[variables]
    print(y)
    df2=pd.get_dummies(y, columns=[y[1]])
    df = pd.concat([df, df2], axis=1)
    df.drop(columns=['info_g','info_p', 'nav', 'trans'])
    print(df)
    print(df.corr())
    vif = pd.DataFrame() 
    vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
    error_vif = vif.loc[(vif>5).sum(axis=1)>0,:]
    print(max)
    print(error_vif)
    #series = [variance_inflation_factor(v.values,i) for i in range(v.shape[1])]
    return vif

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
    
    
    
#############################################
    #print(run_check_vif(X, xy,y))
#############################################

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


print('LinearSVC')
predictions, confusionmatrix,classificationreport,accuracyscore, learning_predictions = run_linearsvc( learning_df, x, y,42)
print(predictions)
print(confusionmatrix)
print(classificationreport)
print(accuracyscore)
#print(learning_predictions)
print(len(learning_predictions))


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
    print(columns)
    
    for col in columns:
        #y_cat = list(cat.iloc[:,i])
        y_cat = list(cat[col])
        print("y_cat")
        print(y_cat)
        #log_reg2  =sm.Logit(list(cat["info_p"]),x_train_col, sm.add_constant()).fit_regularized(maxiter = 150, L1_wt=0.0)
        log_reg2  =sm.Logit(list(cat["info_p"]),x_train_col).fit_regularized(maxiter = 500, L1_wt=0.0)
        print(log_reg2.summary())
        
        #pred =sm.Logit(list(cat["trans"]),x_train_col).predict()
        #print(pred)
        
        
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
    #print(llf)
    #print(llnull)
    #print(llr)
    #print(llpvalue)

    return X_train


print('Logistic Regression')
lr_predictions, lr_confusionmatrix,lr_classificationreport,lr_accuracyscore, lr_learning_predictions,train_tfidf_vector_matrix = run_logis_reg( learning_df, x, y,63)
print(lr_predictions)
print(lr_confusionmatrix)
print(lr_classificationreport)
print(lr_accuracyscore)
#print(learning_predictions)
print(len(lr_learning_predictions))


#print(statsmod_logreg(train_tfidf_vector_matrix))

