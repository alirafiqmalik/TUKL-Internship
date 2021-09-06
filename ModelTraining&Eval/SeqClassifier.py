# **HuggingFace Neural Text Classification Models**

from dataset import *

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries
from torchtext.legacy.data import Field,TabularDataset, BucketIterator, Iterator

# Models
import torch.nn as nn
from torch.cuda import is_available as gtdev
device="cuda" if gtdev() else "cpu"
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Training
import torch.optim as optim



# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

import pandas as pd
from sklearn.model_selection import train_test_split
import os

from google.colab import output


**Dataset**


def gen_dataset(path,
                dst,
                train_test_ratio = 0.9,
                train_valid_ratio = 0.8,
                label="label",
                facts="facts"):
  df_raw = pd.read_csv(path,usecols=[label,facts])
  df_raw=df_raw.dropna()
  df_raw = df_raw.reindex(columns=[label, facts])

  
  # Split according to label
  df_pos = df_raw[df_raw[label] == 0]
  df_neg = df_raw[df_raw[label] == 1]

  # Train-test split
  df_pos_full_train, df_pos_test = train_test_split(df_pos, train_size = train_test_ratio, random_state = 1)
  df_neg_full_train, df_neg_test = train_test_split(df_neg, train_size = train_test_ratio, random_state = 1)

  # Train-valid split
  df_pos_train, df_pos_valid = train_test_split(df_pos_full_train, train_size = train_valid_ratio, random_state = 1)
  df_neg_train, df_neg_valid = train_test_split(df_neg_full_train, train_size = train_valid_ratio, random_state = 1)

  # Concatenate splits of different labels
  df_train = pd.concat([df_pos_train, df_neg_train], ignore_index=True, sort=False)
  df_valid = pd.concat([df_neg_valid, df_neg_valid], ignore_index=True, sort=False)
  df_test = pd.concat([df_pos_test, df_neg_test], ignore_index=True, sort=False)

  # Write preprocessed data
  df_train.to_csv(os.path.join(dst,'train.csv'), index=False)
  df_valid.to_csv(os.path.join(dst,'valid.csv'), index=False)
  df_test.to_csv(os.path.join(dst,'test.csv'), index=False)
  

def tokenizeDataset(tokenizer, source_folder,
                    MAX_SEQ_LEN = 128,batch_size=16):
  
  PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
  UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

  # Fields

  label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
  text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
  fields = [('label', label_field),('facts', text_field)]


  train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv',
                                           test='test.csv', format='CSV', fields=fields, skip_header=True)

  # Iterators

  train_iter = BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.facts),
                            device=device, train=True, sort=True, sort_within_batch=True)
  valid_iter = BucketIterator(valid, batch_size=batch_size, sort_key=lambda x: len(x.facts),
                            device=device, train=True, sort=True, sort_within_batch=True)
  test_iter = Iterator(test, batch_size=batch_size, device=device, train=False, shuffle=False, sort=False)

  return train_iter,valid_iter,test_iter

**Model**

class Classifier(nn.Module):

    def __init__(self,model_name):
        super(Classifier, self).__init__()
        self.model_name=model_name
        self.encoder = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea

**Training&Eval Functions/Classes**

# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']



def train(model,
          train_loader,
          valid_loader,
          optimizer,
          criterion = nn.BCELoss(),
          num_epochs = 1,
          file_path = ".",
          best_valid_loss = float("Inf")):
  
    eval_every = len(train_loader)
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for i in train_loader:
            labels = i.label.type(torch.LongTensor)           
            labels = labels.to(device)
            facts = i.facts.type(torch.LongTensor) 
            facts = facts.to(device)
            output = model(facts, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    

                    # validation loop
                    for j in valid_loader:
                      
                        val_labels = j.label.type(torch.LongTensor)        
                        val_labels = val_labels.to(device)
                        val_facts = j.facts.type(torch.LongTensor)
                        val_facts = val_facts.to(device)
                        output = model(val_facts, labels)
                        loss, _ = output
                        
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                

    save_checkpoint(file_path + '/' +model.model_name.split("/")[-1] +"_"+ 'model.pt', model, best_valid_loss)
    save_metrics(file_path + '/' +model.model_name.split("/")[-1] +"_"+ 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
    return file_path + '/' +model.model_name.split("/")[-1] +"_"

# Evaluation Function

def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for i in test_loader:

                labels = i.label.type(torch.LongTensor)          
                labels = labels.to(device)
                facts = i.facts.type(torch.LongTensor)  
                facts = facts.to(device)
                output = model(facts, labels)

                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(labels.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])

def TrainSaveEval(modelname,
                  batch_size,
                  modelfolder,
                  source_folder="/content/sample_data/",
                  lr=2e-5,
                  epochs=1):
  tokenizer = AutoTokenizer.from_pretrained(modelname)
  output.clear()
  train_iter,valid_iter,test_iter=tokenizeDataset(tokenizer,source_folder=source_folder,batch_size=batch_size)
  model = Classifier(model_name=modelname).to(device)

  output.clear()
  print("Training and Evaluating Model=",modelname,"for epoch=",epochs)

  optimizer = optim.Adam(model.parameters(), lr=lr)

  result_path=train(model=model, optimizer=optimizer,train_loader = train_iter,valid_loader = valid_iter,num_epochs = epochs)
  print("/b")
  best_model = Classifier(model_name=modelname).to(device)
  load_checkpoint(result_path+'model.pt', best_model)
  evaluate(best_model, test_iter)

# ***HuggingFace Models Train,Save&Eval***
