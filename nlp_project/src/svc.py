
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics

def remove_whitespaces(df):
    blanks = [] 
    for i,kw,intent in df.itertuples():  
        if type(intent)==str:            
            if intent.isspace():         
                blanks.append(i)            
    len(blanks)
    return df


def run_linearsvc(learning_df, x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    satisfaction_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LinearSVC()), #squared hinge loss by default 1/2||w||^2  pour transformer la fonction loss='hinge'
                         ])
    satisfaction_clf.fit(x_train, y_train)  
    
    predictions = satisfaction_clf.predict(x_test)
    
    confusionmatrix = metrics.confusion_matrix(y_test,predictions)
    classificationreport = metrics.classification_report(y_test,predictions)
    accuracyscore = metrics.accuracy_score(y_test,predictions)
    
    learning_predictions=[]
    
    for keywords in learning_df:
        learn_predictions = satisfaction_clf.predict(keywords)
        learning_predictions.append(learn_predictions)
    return predictions, confusionmatrix,classificationreport,accuracyscore, learning_predictions


def run_poly_svc(learning_df, x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    satisfaction_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', SVC(kernel='poly', degree=3, gamma=2)), #RBF Kernel #Basic hinge loss by default
                         ])
    satisfaction_clf.fit(x_train, y_train)  
    
    predictions = satisfaction_clf.predict(x_test)
    
    confusionmatrix = metrics.confusion_matrix(y_test,predictions)
    classificationreport = metrics.classification_report(y_test,predictions)
    accuracyscore = metrics.accuracy_score(y_test,predictions)
    
    learning_predictions=[]
    
    for keywords in learning_df:
        learn_predictions = satisfaction_clf.predict(keywords)
        learning_predictions.append(learn_predictions)
    return predictions, confusionmatrix,classificationreport,accuracyscore, learning_predictions


#Output CSV
csv_name = "prediction.csv"
csv_path = '/csv/'

f = csv.writer(open(csv_name, 'w', newline='',encoding='utf-8'))
f.writerow(['Keyword','category'])

#Training Set
df = pd.read_csv('category.tsv', sep='\t')
df.head()
df.isnull().sum()

#Preprocessing
training_df = remove_whitespaces(df)

#Learning Set 
learn_df= pd.read_csv('input.tsv', sep=',')
learning_df = learn_df.values.tolist()

x = training_df['keyword']
y = training_df['category']

p_predictions,p_confusionmatrix,p_classificationreport,p_accuracyscore, p_learning_predictions= run_poly_svc(learning_df, x, y)

print(p_predictions)
print(p_confusionmatrix)
print(p_classificationreport)
print(p_accuracyscore)
print(p_learning_predictions)

predictions, confusionmatrix,classificationreport,accuracyscore, learning_predictions = run_linearsvc( learning_df, x, y)

print(predictions)
print(confusionmatrix)
print(classificationreport)
print(accuracyscore)
print(learning_predictions)



    
