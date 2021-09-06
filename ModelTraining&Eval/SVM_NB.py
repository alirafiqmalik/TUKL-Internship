# **TFID SVM + Naive Bayes Classifer**

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn import model_selection, svm
from dataset import *


def get_vectorizeddataset(path,max_features = 5000):
  X,y=dataloader(path,mode="read")
  predtxt,predlabel=X[0],y[0]
  X=X[1:]
  y=y[1:]
  Vectorizer = TfidfVectorizer(max_features = max_features )
  X = Vectorizer.fit_transform(X).toarray()
  y=np.array(y)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)

  return Vectorizer,X_train, X_test, y_train, y_test

def init_train_NB(X_train, y_train):
  classifier = MultinomialNB()
  classifier.fit(X_train, y_train)
  return classifier

def init_train_SVM(X_train, y_train):
  SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
  SVM.fit(X_train, y_train)
  return SVM

def eval(y_test, y_pred_NB,predictions_SVM):
  # Classification metrics for Naive Bayes Classifer
  reportNB = classification_report(y_test, y_pred_NB,output_dict=True)
  print('Naive Bayes Classifer Accuracy: ', accuracy_score(y_test, y_pred_NB))
  print('\nClassification Report')
  print('======================================================')
  eval_NB=pd.DataFrame(reportNB).transpose().head()
  print(eval_NB)

  # Classification metrics for SVM Classifer
  reportSVM = classification_report(y_test, predictions_SVM,output_dict=True)
  print("SVM Accuracy Score -> ",accuracy_score(y_test,predictions_SVM)*100)
  print('\nClassification Report')
  print('======================================================')
  eval_SVM=pd.DataFrame(reportSVM).transpose().head()
  print(eval_SVM)

  return eval_NB,eval_SVM

def getprediction(Vectorizer,classifier,txt):
  return classifier.predict(Vectorizer.transform([txt]).toarray())[0]

def main(path,max_features = 5000):
  Vectorizer,X_train, X_test, y_train, y_test=get_vectorizeddataset(path,max_features =max_features )
  classifier=init_train_NB(X_train, y_train)
  SVM=init_train_SVM(X_train, y_train)
  y_pred_NB = classifier.predict(X_test)
  predictions_SVM = SVM.predict(X_test)
  print(predtxt,predlabel)
  print(getprediction(Vectorizer,classifier,predtxt))
  print(getprediction(Vectorizer,SVM,predtxt))