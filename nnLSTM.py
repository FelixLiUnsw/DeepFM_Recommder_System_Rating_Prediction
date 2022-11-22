import os
import os.path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import wideAndDeep
start_lr = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 32
class LSTMRating(nn.Module):
    def __init__(self,input_dim,hidden_dim, num_output, num_layers, num_dropout = 0.1, batch_size = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_size= batch_size
        self.layers = num_layers
        self.output = num_output
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = num_layers, batch_first = True,dropout = num_dropout )
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, num_output)
        

    def forward(self, x):

        hidden = (torch.zeros(self.layers, x.size(0), self.hidden_dim).to(device=device),#(D*num_layer, N, H_out) 
                  torch.zeros(self.layers, x.size(0), self.hidden_dim).to(device=device))
        # x ori --> tensor([N,H_in]) --> tensor([32, 20])
        x = x.view(len(x), 1,  -1)        
        output, self.hidden = self.lstm(x,hidden)
        x = output.view(len(x), -1)
        x = self.relu(x)
        rating_scores = self.linear(x)
        return rating_scores



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def train(batch_size = 32):
    net = LSTMRating(input_dim = 20,hidden_dim= 128, num_output= 5, num_layers = 2, num_dropout = 0.2,batch_size= batch_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr = start_lr, weight_decay=1e-5)
    scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    for epoch in range(10):
        total_loss = 0
        total_correct = 0
        total = 0
        print(get_lr(optimizer))
        for batch in tqdm(training_data):
            train,labels = batch
            trains = train.to(device)
            labels = labels.to(device)
            #lstm
            preds = net(trains)
            labels = labels.type(torch.LongTensor).to(device)
            loss = criterion(preds, labels)

            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output = preds.argmax(dim = 1)
            total_loss += loss.item()
            total_correct = total_correct + torch.eq(labels, output).sum().item()
            total = total + labels.size(0)
        scheduler.step()
        acc = total_correct/total * 100
        print("Epoch {} -- loss {} -- acc{} ".format(epoch,total_loss,acc))
    return net

def test(model):
    predictions = []
    i = 0
    
    for batch in tqdm(testing_data):
        # i = i + len(batch[0])
        # print(i)
        test= batch[0]
        preds = model(test)

        #print(preds)
        output = preds.argmax(dim = 1)
        predictions.append(output)
    predictions = np.concatenate(predictions).ravel()
    predictions = predictions + 1
    submission = pd.read_csv("submission.csv")
    submission.Predicted = predictions
    submission.to_csv("submission.csv",index=False)
    print(np.amin(predictions), np.amax(predictions))

training_data, validation_data, testing_data = wideAndDeep(".",0.7, batch= batch_size, data_type="LSTM")

model = train(batch_size)
test(model)