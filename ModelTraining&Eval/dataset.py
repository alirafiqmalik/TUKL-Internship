# **Preprocessing**

import pandas as pd    
import numpy as np
import re
import string
import nltk
import os.path
nltk.download('stopwords')
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

def readjson(datasetpath):
  return pd.read_json(path_or_buf=datasetpath, lines=True)

def readcsv(datasetpath):
  return pd.read_csv(datasetpath)


def get_col(dataframe,col,start=0,end=0):
  if end==0:
    end=len(dataframe) 
  return dataframe[col].tolist()[start:end]


def getextremelen(facts):
  max=[]
  for i in facts:
    max.append(len(i.split()))
  minlength=max[np.argmin(max)]
  maxlength=max[np.argmax(max)]
  return minlength,maxlength

def cleantext(datasetlist):
  print("Min/Max string length before cleaning text:")
  minb,maxb=getextremelen(datasetlist)
  print(minb,maxb)
    
  cleanedtxt=[]
  for i in datasetlist:
    temp=re.sub("^[0-9]+[.]", " ", i )
    temp=re.sub("[.]|['’].", " ", temp )
    temp=temp.translate(str.maketrans('', '', string.punctuation))
    temp=re.sub("\s+"," ",temp)
    temp=" ".join([word for word in str(temp).split() if word not in STOPWORDS])
    cleanedtxt.append(temp)
    
  print("Min/Max string length after cleaning text:")
  mina,maxa=getextremelen(cleanedtxt)
  print(mina,maxa)
  return cleanedtxt

def dataloader(path,trainratio=0.8,mode="split"):
  dataset=pd.read_csv(path).dropna()
  X=cleantext(get_col(dataset,'facts'))
  y=get_col(dataset,'label')
  if mode=="split":
    train_facts,test_facts,train_labels,test_labels = train_test_split(X, y,train_size = trainratio, random_state = 0)
    print("Total Dataset=",len(dataset),"Train Dataset=",len(train_facts),"Test Dataset=",len(test_facts))
    return (train_facts,train_labels),(test_facts,test_labels)
  elif mode=="read":
    return X,y
