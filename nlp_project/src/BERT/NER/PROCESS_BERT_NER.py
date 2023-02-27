           
import numpy as np
import joblib
import torch
import transformers
import torch.nn as nn 
# import tokenization
#Config
#https://www.youtube.com/watch?v=MqQ7rqRllIc
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCH = 10
BASE_MODEL_PATH = "bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "ner_pos2.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case = True
)
    # TOKENIZER2 = tokenization.FullTokenizer(
    #     BASE_MODEL_PATH,
    #     do_lower_case = True
    # )


class EntityDataset:
    def __init__(self, text, pos, tags):
        #text = [["kw1","kw1"],["kw2"]]
        #pos = [[1 4 3],[2 3 5]]
        #tags = [[1 4 3],[2 3 5]]
        self.text = text
        self.pos = pos
        self.tags = tags
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text = self.text[item]
        pos = self.pos[item]
        tags = self.tags[item]
        
        ids = []
        target_pos = []
        target_tags = []
        
        for i, s in enumerate(text):
            inputs = TOKENIZER.encode(
                s,
                add_special_tokens = False
            )
            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend([pos[i]] * input_len)
            target_tags.extend([tags[i]] * input_len)
            
            
        ids = ids[:MAX_LEN - 2]
        target_pos = target_pos[:MAX_LEN - 2]
        target_tags = target_tags[:MAX_LEN - 2]
        
        ids = [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tags = [0] + target_tags + [0]
        
        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = MAX_LEN - len(ids)
        
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        target_tags = target_tags + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype = torch.long),
            "mask": torch.tensor(mask, dtype = torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype = torch.long),
            "target_pos": torch.tensor(target_pos, dtype = torch.long),
            "target_tags": torch.tensor(target_tags, dtype = torch.long), 
        }      

class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos):
        #super():  to call methods of the superclass in your subclass
        super(EntityModel, self).__init__()
        
        #
        self.num_tag = num_tag
        self.num_pos = num_pos

        #
        self.bert = transformers.BertModel.from_pretrained(BASE_MODEL_PATH)
        
        #During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3)
        
        #Applies a linear transformation to the incoming data: y=xAT+by = xA^T + by=xAT+b.
        self.out_tag = nn.Linear(768, self.num_tag)
        self.out_pos = nn.Linear(768, self.num_pos)
    
    def forward(self,ids, mask, token_type_ids, target_pos, target_tags ):
        #Defines the computation performed at every call.
        #Should be overridden by all subclasses.
        ol, _ = self.bert(ids, attention_mask = mask, token_type_ids=token_type_ids, return_dict=False)
        
        bo_tag = self.bert_drop_1(ol)
        bo_pos = self.bert_drop_2(ol)
        
        tag = self.out_tag(bo_tag)
        pos = self.out_pos(bo_pos)
         
        loss_tag = cross_entropy_loss(tag, target_tags, mask, self.num_tag)
        loss_pos = cross_entropy_loss(pos, target_tags, mask, self.num_pos)

        loss = (loss_tag + loss_pos)/2
        
        #metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
        #print(metrics.result().numpy())
        return tag, pos, loss

import tensorflow as tf 
import keras
from keras import metrics

def cross_entropy_loss(output, target, mask, num_labels):
    #Cross Entropy Loss
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1), 
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    
    return loss

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# def compute_metrics(output, target, mask, num_labels):
#     active_loss = mask.view(-1) == 1
#     active_logits = output.view(-1, num_labels)

#     labels = pred.label_ids
#     preds = pred.preditions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy':acc,
#         'f1':f1,
#         'precision':precision,
#         'recall':recall
#     }

def predict(sentence):
    meta_data = joblib.load("meta.bin")
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))
    
    tokenized_sentence = TOKENIZER.encode(sentence)

    sentence = sentence.split()

    test_dataset = EntityDataset(
        text=[sentence], 
        pos=[[0] * len(sentence)], 
        tags=[[0] * len(sentence)]
    )

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag, num_pos=num_pos)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)

    #Arret calcul des gradients
    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, pos, _= model(**data)
        print('Source Keyword')
        print(sentence)
        print('Bert Keyword')
        print(TOKENIZER.decode(tokenized_sentence))
        #print('Tokens')
        #print(tokenized_sentence)
        print('NER')
        pred_ner = enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[:len(tokenized_sentence)]
        print(enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1))
              [:len(tokenized_sentence)])
        print('POS')
        print(enc_pos.inverse_transform(
                pos.argmax(2).cpu().numpy().reshape(-1)) 
              [:len(tokenized_sentence)]) 
        return pred_ner         

data_test = ["google merch store", "google t-shirt", "youtube merch", "google dino game"]

len_pred_ner = []
predicted_ner = [] 
predicted_ner_cod = []
x=0      
for i in data_test:
    pre_ner = predict(i)
    #print(pre_ner)
    predicted_ner.append(pre_ner.tolist())
    predicted_ner_code = [w.replace('O', '0') for w in pre_ner.tolist()]
    predicted_ner_code = [w.replace('B-ART', '1') for w in predicted_ner_code]
    predicted_ner_code = [w.replace('I-BRAND', '2') for w in predicted_ner_code]
    predicted_ner_cod.append(predicted_ner_code)
    len_pred_ner.append(x)
    x=x+1
    
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

#data = {'ner':predicted_ner, "x":len_pred_ner }
print(predicted_ner_cod)

y_pred = [['B-ART', 'O', 'O', 'I-BRAND', 'O', 'B-ART'], ['B-ART', 'I-BRAND', 'O', 'O', 'O', 'B-ART'], ['B-ART', 'I-BRAND', 'O', 'O', 'B-ART'], ['B-ART', 'I-BRAND', 'O', 'O', 'B-ART']]

y_true = [x for xs in predicted_ner for x in xs]
y_pred = [x for xs in y_pred for x in xs]

print(y_true)
print(y_pred)

target_names = ['O', 'B-ART', 'I-BRAND']

print(classification_report(y_true, y_pred))


