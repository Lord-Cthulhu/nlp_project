#SVC_Model_No_Porter_Visuals

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


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

#Nightly Build V1.2 
from sklearn_dev.inspection._plot.decision_boundary import DecisionBoundaryDisplay

print(nltk.__version__)
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
def run_linearsvc(learning_df, x, y,rstate, graph):

    #Split Data into Train Test Partitions
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=rstate)
    
    #LinearSVC Pipeline
    text_clf = Pipeline([
                        ('tfidf', TfidfVectorizer()), # Vect+TfidfTransformer = TfidfVectorizer
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
    
    #######VISUALS########    

    # plt.figure(figsize=(10, 5))
    # for i, C in enumerate([1, 100]):
    #     # "hinge" is the standard SVM loss
    #     clf = text_clf.fit(x_train, y_train)  
    #     # obtain the support vectors through the decision function
    #     decision_function = clf.decision_function(x_train)
    #     # we can also calculate the decision function manually
    #     # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
    #     # The support vectors are the samples that lie within the margin
    #     # boundaries, whose size is conventionally constrained to 1
    #     support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    #     support_vectors = x_train[support_vector_indices]

    #     plt.subplot(1, 2, i + 1)
    #     plt.scatter(x_train[:, 0], x_train[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    #     ax = plt.gca()
    #     DecisionBoundaryDisplay.from_estimator(
    #         clf,
    #         x_train,
    #         ax=ax,
    #         grid_resolution=50,
    #         plot_method="contour",
    #         colors="k",
    #         levels=[-1, 0, 1],
    #         alpha=0.5,
    #         linestyles=["--", "-", "--"],
    #     )
    #     plt.scatter(
    #         support_vectors[:, 0],
    #         support_vectors[:, 1],
    #         s=100,
    #         linewidth=1,
    #         facecolors="none",
    #         edgecolors="k",
    #     )
    #     plt.title("C=" + str(C))
    # plt.tight_layout()
    # plt.show()  
    
    #TFIDF Matrix
    
    train_tfidf_features = text_clf.named_steps['tfidf'].get_feature_names()
    train_tfidf_vector_array = text_clf.named_steps['tfidf'].fit_transform(x_train).toarray()
    train_tfidf_vector_matrix = pd.DataFrame(data=train_tfidf_vector_array, columns=train_tfidf_features)
    #train_tfidf_vector_matrix = pd.concat([train_tfidf_vector_matrix , train_tfidf_features], axis=1)
    print("tfidf array")
    #train_tfidf_vector_matrix = train_tfidf_vector_matrix.drop(columns=['info_g','info_p', 'nav', 'trans'])
    print(train_tfidf_vector_array)
    print(train_tfidf_vector_matrix)
    
    
    X=train_tfidf_vector_array
    
    #X, y = make_blobs(n_samples=8000, centers=2, random_state=0)
    #print(X)
    info_p_train = []
    info_g_train = []
    nav_train = []
    com_train = []
    trans_train = []
    for ii in y_train:
        if ii != "info_p":
            info_p_train.append(0)
        if ii == "info_p":
            info_p_train.append(1)
        if ii != "info_g":
            info_g_train.append(0)
        if ii == "info_g":
            info_g_train.append(1)
        if ii != "nav":
            nav_train.append(0)
        if ii == "nav":
            nav_train.append(1)
        if ii != "com":
            com_train.append(0)
        if ii == "com":
            com_train.append(1)
        if ii != "trans":
            trans_train.append(0)
        if ii == "trans":
            trans_train.append(1)
                
    y=com_train
    print(len(y))          
    info_p_train_=np.array(info_p_train)
    # plt.figure(figsize=(10, 5))
    # for i, C in enumerate([1, 100]):
    #     # "hinge" is the standard SVM loss
    #     clf = LinearSVC(C=C, loss="hinge", random_state=42).fit(X, y)
    #     # obtain the support vectors through the decision function
    #     decision_function = clf.decision_function(X)
    #     # we can also calculate the decision function manually
    #     # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
    #     # The support vectors are the samples that lie within the margin
    #     # boundaries, whose size is conventionally constrained to 1
    #     support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    #     support_vectors = X[support_vector_indices]
    #     plt.subplot(1, 2, i + 1)
    #     plt.scatter(X[:,0], X[ :,1], c=y, s=30, cmap=plt.cm.Paired)
    #     ax = plt.gca()
    #     DecisionBoundaryDisplay.from_estimator(
    #         clf,
    #         X[:,0],
    #         ax=ax,
    #         grid_resolution=50,
    #         plot_method="contour",
    #         colors="k",
    #         levels=[-1, 0, 1],
    #         alpha=0.5,
    #         linestyles=["--", "-", "--"],
    #     )
    #     plt.scatter(
    #         support_vectors[:, 0],
    #         support_vectors[:, 1],
    #         s=100,
    #         linewidth=1,
    #         facecolors="none",
    #         edgecolors="k",
    #     )
    #     plt.title("C=" + str(C))
    # plt.tight_layout()
    # plt.show()
    
    
    from sklearn.decomposition import PCA
    if graph == 1:
        pca = PCA(n_components=2)
        Xreduced = pca.fit_transform(X)

        def make_meshgrid(x, y, h=.02):
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            return xx, yy

        def plot_contours(ax, clf, xx, yy, **params):
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            out = ax.contourf(xx, yy, Z, **params)
            return out

        model = LinearSVC(C=1,  random_state=rstate)
        clf = model.fit(Xreduced, y)

        fig, ax = plt.subplots()
        # title for the plots
        title = ('Decision surface of linear SVC ')
        # Set-up grid for plotting.
        X0, X1 = Xreduced[:, 0], Xreduced[:, 1]
        xx, yy = make_meshgrid(X0, X1)

        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_ylabel('Component 2')
        ax.set_xlabel('Component 1')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title('Classification nav avec le modele SVC PCA*')
        plt.suptitle('Reduction de dimensions par PCA')
        ax.legend()
        plt.show()

    if graph == 2:
        #Equivalent de diagonaliser la matrice de covariance
        pca2 = PCA(n_components=2)
        Xreduced2 = pca2.fit_transform(X)

        model = LinearSVC(C=1,  random_state=rstate)
        clf = model.fit(Xreduced2, y)
        
        from mlxtend.plotting import plot_decision_regions
        
        y_array= np.array(y)
        
        plot_decision_regions(Xreduced2, y_array, clf=model, legend=2)
        
        plt.title('Reduction de dimensions par PCA')
        plt.suptitle('SVM pour la classification com')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()



    #######################  
            

    return predictions, confusionmatrix,classificationreport,accuracyscore, learning_predictions


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
predictions, confusionmatrix,classificationreport,accuracyscore, learning_predictions = run_linearsvc( learning_df, x, y,42, 2)
print(predictions)
print(confusionmatrix)
print(classificationreport)
print(accuracyscore)
#print(learning_predictions)
print(len(learning_predictions))


