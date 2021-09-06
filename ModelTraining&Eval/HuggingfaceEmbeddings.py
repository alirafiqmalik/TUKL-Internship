#**Embedding Model #**


#**HuggingFace Embedding Model #**
from dataset import *
from torch.cuda import is_available as getdevice
device='cuda' if getdevice() else 'cpu'

import torch

from sentence_transformers import SentenceTransformer,InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import pickle

def listtostr(x):
  y=""
  for i in x:
    y=y+i+" "
  return y.strip()

def texttochunks(x,windowsize=512):
  inputtxt=x.split()
  chunks=[]
  start=0
  while True:
    end=start+windowsize
    if end>=len(inputtxt):
      chunks.append(listtostr(inputtxt[start:]))
      break
    chunks.append(listtostr(inputtxt[start:end]))
    start=end
  return chunks



def get_exp_factlist(x,windowsize):
  expset=[]
  record=[]
  count=0

  for i in x:
    if len(i.split())<windowsize:
      expset.append(i)
      record.append([count])
      count+=1
    else:
      temp=texttochunks(i)
      for j in temp:
        expset.append(j)
  
      temp2=[]
      for j in range(len(temp)):
        temp2.append(count+j)
  
      record.append(temp2)
      count+=len(temp)

  return expset,record

def get_exp_labels(train_labels,record):
  exp_labels=[]
  for id,i in enumerate(record):
    for j in i:
      exp_labels.append(train_labels[id])
  return exp_labels


def separate_embeddings(embeddings,record):
  sep_embeddings=[]
  for i in record:
    temp=[]
    for j in i:
      temp.append(embeddings[j])
    #print(len(temp),type(temp),len(temp[0]),type(temp[0]))
    sep_embeddings.append(temp)
  return sep_embeddings 


def combinechunkembeddings(chunkembeddings,mode="mean"):
  chunkembeddings=torch.tensor(chunkembeddings)
  finalchunkembedding=[]
  if mode=="mean":
    poolfn=torch.mean
  elif mode=="max":
    poolfn=torch.amax
  for i in range(len(chunkembeddings[0])):
    tmp=[]
    for j in range(len(chunkembeddings)):
      tmp.append(chunkembeddings[j][i])
    finalchunkembedding.append(poolfn(torch.stack(tmp),0))
  return np.array(finalchunkembedding)



def get_dataloader(factlist,factlist_labels):
  posfacts=[]
  negfacts=[]
  for i in range(len(factlist)):
    label=factlist_labels[i]
    facts=factlist[i]
    if label:
      posfacts.append(facts)
    else:
      negfacts.append(facts)
  print("Total Samples=",len(factlist_labels),"Positive Labels=",len(negfacts),"Negative Labels=",len(posfacts))
  if len(posfacts)<len(negfacts):
    ni=len(posfacts)
  else:
    ni=len(negfacts)  
  
  return [InputExample(texts=posfacts[0:ni], label=1.),InputExample(texts=negfacts[0:ni], label=0.)]


def truncate_string(x,windowsize,mode="head"):
  txtlist=x.split(" ")
  if mode=="head":
    return " ".join(txtlist[0:windowsize])
  elif mode=="tail":
    return " ".join(txtlist[windowsize:])
  elif mode=="head+tail":
    return " ".join(txtlist[:windowsize-60]+txtlist[windowsize+60:])


def truncate_string_list(x,windowsize,mode="head"):
  truncate_list_string=[]
  for i in x:
    truncate_list_string.append(truncate_string(i,windowsize,mode))
  return truncate_list_string


class WordEmbedding:
  def __init__(self,device,modelname='paraphrase-MiniLM-L6-v2'):
    self.modelname=modelname
    self.device=device
    self.Model=SentenceTransformer(modelname,device=self.device)
 
  def ModelEmbeddings(self,modelfactlist,mode):
    if mode=="truncate_head":
      self.sentence_embeddings=self.getembeddings(modelfactlist,mode="head")
    elif mode=="truncate_tail":
      self.sentence_embeddings=self.getembeddings(modelfactlist,mode="tail")
    elif mode=="truncate_head+tail":
      self.sentence_embeddings=self.getembeddings(modelfactlist,mode="head+tail")
    elif mode=="mean_pooling":
      self.sentence_embeddings=self.EncodeTextEmbeddings(modelfactlist,mode="mean")
    elif mode=="max_pooling":
      self.sentence_embeddings=self.EncodeTextEmbeddings(modelfactlist,mode="max")
    
    return self.sentence_embeddings

  def getembeddings(self,factlist,mode="head"):
    if not mode=="head":
      windowsize=self.Model.max_seq_length
      factlist=truncate_string_list(factlist,windowsize,mode=mode)
    sentence_embeddings=self.Model.encode(factlist,show_progress_bar=True,device=self.device)
    return sentence_embeddings

  def EncodeTextEmbeddings(self,factlist,mode):
    exp_factlist,record=get_exp_factlist(factlist,self.Model.max_seq_length)

    embeddings=self.Model.encode(exp_factlist,show_progress_bar=True,device=self.device)

    hybrid_sentence_embeddings=separate_embeddings(embeddings,record)

    sentence_embeddings=[]
    for i in hybrid_sentence_embeddings:
      if len(i)!=1:
        sentence_embeddings.append(combinechunkembeddings(i,mode=mode))
      else:
        sentence_embeddings.append(i[0])

    return np.array(sentence_embeddings)


  
  def save_embeddings(self,path='/content/'):
    #Store sentences & embeddings on disc
    with open(path+'embeddings.pkl', "wb") as fOut:
      pickle.dump({'sentences': self.factlist, 'embeddings': self.sentence_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
  
  def load_saved_embeddings(self,path='/content/'):
    #Load sentences & embeddings from disc
    with open(path+'embeddings.pkl', "rb") as fIn:
      stored_data = pickle.load(fIn)
      self.factlist = stored_data['sentences']
      self.sentence_embeddings = stored_data['embeddings']

  def gets_max_similiarity(self,k,testtxt,show_progress_bar=True):
    testtxt=self.Model.encode(testtxt,show_progress_bar=show_progress_bar,device=self.device)
    testsim=cosine_similarity(self.sentence_embeddings,testtxt.reshape(1, -1))
    testsim=testsim.reshape(len(testsim))
    sortedsimarray=np.sort(testsim,axis=0)[::-1][0:k]      
    testlist=testsim.tolist()
    maxindex=[]
    for i in sortedsimarray:
      maxindex.append(testlist.index(i))
    return maxindex

  def getsimilari(self,k,txt):
    max_sim_index=self.gets_max_similiarity(k,[txt])
    return max_sim_index
  
  
  def ext_train(self,train_facts,train_labels,epoch=1,batch_size=16,warmup_steps=100):


    exp_factlist,record=get_exp_factlist(train_facts,model.max_seq_length)
    exp_labels=get_exp_labels(train_labels,record)

    train_examples=get_dataloader(exp_factlist,exp_labels)

    #Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(self.Model)

    #Tune the model
    self.Model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epoch, warmup_steps=warmup_steps)
  
  
  def getlabel(self,k,testtxt,train_labels,show_progress_bar=False):
    index=self.gets_max_similiarity(k,testtxt,show_progress_bar=show_progress_bar)
    k_labels=[]
    for i in index:
      k_labels.append(train_labels[i])
    poscount=0
    for i in k_labels:
      if i==1:
        poscount+=1
      if poscount>(len(k_labels)/2):
        return 1
    return 0
  

  def evaluate_embeddings(self,test_facts,train_labels,k=3):
    correct=0
    for id in range(len(test_facts)):
      testtxt=test_facts[id]
      label=test_labels[id]
      predlabel=self.getlabel(k,testtxt,train_labels)
      if predlabel==label:
        correct+=1
    accuracy=correct*100/len(test_facts)
    print("For k=",k,"\nCorrect Predictions=",correct,"\nTotal=",len(test_facts),"\nAccuracy=%.2f" % accuracy)
    return accuracy


def main(path):
  (train_facts,train_labels),(test_facts,test_labels)=dataloader(path,trainratio=0.8)
  WordEmbeddingModel=WordEmbedding(device=device)

  model=WordEmbeddingModel.Model
  print("Max Sequence Length:", model.max_seq_length)
  model.max_seq_length=512
  print("Max Sequence Length:", model.max_seq_length)
  xval=range(1,6)

  sentence_embeddings=WordEmbeddingModel.ModelEmbeddings(train_facts,mode="truncate_head")
  #**Trunacate Head #**

  xval=range(1,6)
  headt_yval=[]
  for i in xval:
    print("#############################################")
    headt_yval.append(WordEmbeddingModel.evaluate_embeddings(test_facts,train_labels,k=i))

  sentence_embeddings=WordEmbeddingModel.ModelEmbeddings(train_facts,mode="truncate_tail")
  #**Trunacate Tail #**

  tailt_yval=[]
  for i in xval:
    print("#############################################")
    tailt_yval.append(WordEmbeddingModel.evaluate_embeddings(test_facts,train_labels,k=i))

  sentence_embeddings=WordEmbeddingModel.ModelEmbeddings(train_facts,mode="truncate_head+tail")
  #**Trunacate Head+Tail #**

  headtailt_yval=[]
  for i in xval:
    print("#############################################")
    headtailt_yval.append(WordEmbeddingModel.evaluate_embeddings(test_facts,train_labels,k=i))

  sentence_embeddings=WordEmbeddingModel.ModelEmbeddings(train_facts,mode="max_pooling")
  
  
   #**Max Pooling #**
  maxp_yval=[]
  for i in xval:
    print("#############################################")
    maxp_yval.append(WordEmbeddingModel.evaluate_embeddings(test_facts,train_labels,k=i))

  #**Mean Pooling #**

  sentence_embeddings=WordEmbeddingModel.ModelEmbeddings(train_facts,mode="mean_pooling")

  meanp_yval=[]
  for i in xval:
    print("#############################################")
    meanp_yval.append(WordEmbeddingModel.evaluate_embeddings(test_facts,train_labels,k=i))

  #**Tables #**

  summary=pd.DataFrame({"head+tail":headtailt_yval,
                        "head only":headt_yval, 
                        "tail only":tailt_yval, 
                        "mean pooling":meanp_yval, 
                        "max pooling":maxp_yval}, 
                        index=["Accuracy for k="+str(i) for i in xval])
  summary.columns.name = 'Pre-Trained Model'
  print(summary)