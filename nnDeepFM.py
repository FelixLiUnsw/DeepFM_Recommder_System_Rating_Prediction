#########################################################################################
#  Author: Jiaxuan Li, zID:           Tel:               Email:                         #
#  If you have any question about this program, please contact me                       #   
#  This NNs method needs at least 20 epochs to train to have a higher accuarcy          #
#  DeepFM is a progressive version for Wide&Deep model by replacing Fully Connect layer #
#########################################################################################

import os.path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from preprocess import wideAndDeep
import warnings
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + 1e-6)
        return loss



# get learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class DeepFM(nn.Module):
    def __init__(self, sparse_feature, len_dense_features=0, emb_size=8,
                 hid_dims=[128, 64], num_classes=1, dropout=[0.2, 0.2]):
        super().__init__()
        """
        :Parameter
        --------------------------------------------------
        sparse_feature: nunique() of sparse features, sparse features has ['name', 'user_id', 'recipe_id', 'contributor_id']
        len_dense_features: how many dense features, dense features are all other features. such as totalFat, year, month etc. 16 in total
        emb_size: it is embedding size
        hidden_dims: it is hidden_dimension for DNN, first layer is 128, second layer is 64
        num_class: it is the output for final fc layer
        dropout: it is the droprate to avoid overfitting.
        --------------------------------------------------
        """
        
        self.sparse_feature_len = len(sparse_feature)
        self.len_dense_features = len_dense_features

        #FM
        ################################ FM Layer####################################
        #Order 1 interactions among features
        #dense
        self.order_1_dense = nn.Linear(self.len_dense_features, 1) 
        #sparse
        self.order_1_sparse_emb = nn.ModuleList([nn.Embedding(num_emb, 1) for num_emb in sparse_feature])
        
        #Order 2 pairwise eature interaction as inner product of respective features latent vectors
        #sparse
        self.order_2_sparse_emb = nn.ModuleList([nn.Embedding(num_emb, emb_size) for num_emb in sparse_feature])
        self.dense_linear = nn.Linear(self.len_dense_features, self.sparse_feature_len * emb_size)
        #############################################################################
        
        
        ################################ DNN Layer ##################################
        dnn_dims = [self.sparse_feature_len * emb_size] + hid_dims
        self.relu = nn.ReLU()
        self.dnn_module = DNNModule(dnn_dims=dnn_dims, num_classes=num_classes, dropout_rate=dropout)
        #############################################################################
        
    def forward(self, sparse, dense=None):
        ################################ FM Layer####################################
        # order 1 FM
        order_1_sparse = []
        for i, emb in enumerate(self.order_1_sparse_emb):
            #print(sparse[:,i].shape)
            embedding = emb(sparse[:, i])
            order_1_sparse.append(embedding)
        order_1_sparse = torch.cat(order_1_sparse, dim=-1)
        order_1_sparse = torch.sum(order_1_sparse, 1, keepdim=True)


        order_1s_dense = self.order_1_dense(dense)
        order_1 = order_1_sparse + order_1s_dense


        # order 2 FM
        order_2_sparse = []
        for i, emb in enumerate(self.order_2_sparse_emb ):
            embedding = emb(sparse[:, i])
            order_2_sparse.append(embedding)
        order_2_sparse = torch.stack(order_2_sparse, dim=1)

        #This is an advanced version. The idea is from this paper: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
        #simplied the pairwise interaction sum(sum(<v_i, v_j>x_i*x_j)) = 0.5*sum((sum_(v_i*x_i))^2 - sum(v_i^2*x_i^2))
        sum_embedding = torch.sum(order_2_sparse, 1) # sum_(v_i*x_i)
        square_sum = sum_embedding * sum_embedding # (sum_(v_i*x_i))^2
        sum_square = order_2_sparse * order_2_sparse  # v_i^2*x_i^2
        sum_square = torch.sum(sum_square, 1)  # sum(v_i^2*x_i^2)
        cross_term = square_sum - sum_square # sum((sum_(v_i*x_i))^2 - sum(v_i^2*x_i^2))
        cross_term  = 0.5*cross_term  # sum(sum(<v_i, v_j>x_i*x_j)) = 0.5*sum((sum_(v_i*x_i))^2 - sum(v_i^2*x_i^2))
        order_2 = torch.sum(cross_term , dim = 1, keepdim=True)
        ##############################################################################
        
        #################################### DNN #####################################
        dnns = torch.flatten(order_2_sparse, 1) + self.relu(self.dense_linear(dense))
        dnn_in = order_1 + order_2  + dnns
        dnn_out = self.dnn_module(dnn_in) 
        if self.training:
            return dnn_out
        else:
            pred_logits = F.softmax(dnn_out, dim=-1)
            pred_label = torch.argmax(dnn_out, dim=-1)
            pred_prob = torch.sum(torch.stack([i*pred_logits[:,i] for i in range(pred_logits.shape[-1])], dim=-1), dim=-1)
            return pred_prob, pred_label


class DNNModule(nn.Module):
    """
    :parameters
    ------------------------------------
    dnn_dims: [a, b] layer 1 is a inputs, layer 2 is b input
    num_class: output dim
    dropout_rate: [0.2,0.2]
    ------------------------------------
    """

    def __init__(self, dnn_dims: list, num_classes=1, dropout_rate=[0.4, 0.4]): # it was  [0.2,0.2]
        super(DNNModule, self).__init__()
        print('dropout_rate ', dropout_rate)
        self.dnn_module = nn.ModuleList()
        for i in range(1, len(dnn_dims)):
            self.dnn_module.append(
                nn.Sequential(
                    nn.Linear(dnn_dims[i - 1], dnn_dims[i]),
                    nn.BatchNorm1d(dnn_dims[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate[i - 1])
                )
            )

        self.dnn_dims = dnn_dims
        self.dnn_linear = nn.Linear(dnn_dims[-1], num_classes)

    def forward(self, inputs):
        dnn_out = inputs
        for i in range(len(self.dnn_module)):
            dnn_out = self.dnn_module[i](dnn_out)

        dnn_out = self.dnn_linear(dnn_out)
        return dnn_out
class Trainer:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(self.device)

        #  prepare dataset
        batch_size = 1024
        self.training_data, self.validation_data, self.testing_data, category_feature_unique, dense_features, ratings_unique = wideAndDeep(".", 1, batch_size)

        #  prepare train parameter
        self.start_lr = 0.1

        # prepare model
        num_classes = len(ratings_unique)
        print('num_classes ', num_classes)
        self.model = DeepFM(sparse_feature=category_feature_unique, len_dense_features=len(dense_features), num_classes=num_classes)
        self.model = self.model.to(self.device)

        # prepare criterion
        criterion = nn.CrossEntropyLoss()
        self.criterion = criterion.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.start_lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        self.eval_criterion = RMSELoss()

        # dst save path
        dst_dir = '.'
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        self.dst_dir = dst_dir

    def train(self):
        epochs = 45

        for epoch_i in range(epochs):
            """training"""
            self.model.train()
            print("Epoch {} Current lr : {}".format(epoch_i, self.optimizer.state_dict()['param_groups'][0]['lr']))
            total_loss = 0
            total_correct = 0
            total = 0
            for (sparse, dense, label) in tqdm(self.training_data):
                sparse, dense, label = sparse.to(self.device), dense.to(self.device), label.long().to(self.device)
                sparse = sparse.long()
                dense = dense.float()
                pred = self.model(sparse, dense)
                loss = self.criterion(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # record_result
                total_loss += loss.item()
                pred_label = torch.argmax(pred, dim=-1)
                total_correct = total_correct + torch.eq(label, pred_label).sum().item()
                total = total + label.size(0)

            self.scheduler.step()
            print('total_correct ', total_correct)
            acc = 100.0 * total_correct / total
            print("Epoch {} --lr{} -- loss {:.6f} -- acc{}\n\n".format(epoch_i, get_lr(self.optimizer), total_loss, acc))

            # save and evaluate model
            if epoch_i % 5 == 0 and epoch_i > 0:
                #self.eval_model(epoch_i)
                torch.save(self.model.state_dict(), 'DeepFM.pth')

        return self.model
    def eval_model(self, epoch):
        self.model.eval()
        total_compare = 0
        total_correct = 0

        total_eval1 = 0
        total_eval2 = 0
        for (cate_fea, nume_fea, label) in tqdm(self.validation_data):
            cate_fea, nume_fea, label = cate_fea.to(self.device), nume_fea.to(self.device), label.to(self.device)
            cate_fea = cate_fea.long()
            nume_fea = nume_fea.float()

            pred_prob, pred_label = self.model(cate_fea, nume_fea)
            temp_correct1 = self.eval_criterion(label, pred_prob)
            total_eval1 += temp_correct1
            temp_correct2 = self.eval_criterion(label, pred_label)
            total_eval2 += temp_correct2

            total_correct = total_correct + torch.eq(label, pred_label).sum().item()
            total_compare = total_compare + label.size(0)
        print(total_compare)
        print('total_eval1 = ',total_eval1,' total_eval2 = ',total_eval2)
        print('eval_model total_correct ', total_correct)
        acc = 100.0 * total_correct / total_compare
        print("Epoch {} -- acc={:.6f} -- {}/{}".format(epoch, acc, total_correct, total_compare))
        print('##'*50)

#         self.model.train()
    def test(self, model):
        predictions = []
        counter = 0
        dense_counter = 0
        for (sparse, dense) in tqdm(self.testing_data):
            sparse, dense = sparse.to(self.device), dense.to(self.device)
            counter = counter + len(sparse)
            dense_counter = dense_counter + len(dense)
            sparse = sparse.long()
            dense = dense.float()
            pred = model(sparse, dense)
            #print(pred)
            #print(torch.argmax(pred, dim=1))
            output = torch.argmax(pred, dim=1)
            #break
            #pred_logits = F.softmax(pred, dim=-1)
            #pred_label = torch.argmax(pred, dim=-1)
            #pred_prob = torch.sum(torch.stack([i*pred_logits[:,i] for i in range(pred_logits.shape[-1])], dim=-1), dim=-1)
            predictions.append(output.cpu().detach().numpy())
        predictions = np.concatenate(predictions).ravel()
        predictions = predictions + 1
        submission = pd.read_csv("submission.csv")
        submission.Predicted = predictions
        submission.to_csv("submission.csv",index=False)
        print(np.amin(predictions), np.amax(predictions))
        
def main():
    print("Deep FM")
    trainer = Trainer()
    model = trainer.train()
    trainer.test(model)

if __name__ == "__main__":
    main()
