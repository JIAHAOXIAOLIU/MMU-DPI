import os
import time
import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score,\
                            f1_score, accuracy_score
from tqdm import tqdm
from loguru import logger
import pandas as pd
from tdc.multi_pred import DTI

from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

from Utils import csv_record, check_dir, save_model, class_metrics

class DTIDataset(data.Dataset):
    def __init__(self, idx_map, df):
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        idx1 = self.idx_map[str(self.df.iloc[index].Graph_Drug)]
        idx2 = self.idx_map[self.df.iloc[index].Graph_Target]
        label = self.df.iloc[index].Graph_Label
        drug_encoding = self.df.iloc[index].drug_encoding
        target_encoding = self.df.iloc[index].target_encoding
        v_d = drug_encoding
        v_p = utils.protein_2_embed(target_encoding)
        y = self.df.iloc[index].Seq_Label
        return v_d, v_p, y, idx1, idx2, label


def get_model():
    drug_encoding, target_encoding = 'MPNN', 'CNN'
    config = utils.generate_config(drug_encoding=drug_encoding, 
                         target_encoding=target_encoding, 
                         cls_hidden_dims=[1024, 1024, 512], 
                         train_epoch=5, 
                         LR=0.001, 
                         batch_size=128,
                         hidden_dim_drug=128,
                         mpnn_hidden_size=128,
                         mpnn_depth=3, 
                         cnn_target_filters=[32, 64, 96],
                         cnn_target_kernels=[4, 8, 12]
                        )
    model = models.model_initialize(**config)
    return model

def evaluate_model(model, data_loader, batch_size):
    model.eval()
    batch_total = len(data_loader)
    y_pred = np.empty([batch_total, batch_size])
    y_label = np.empty([batch_total, batch_size])
    for i in tqdm(range(batch_total), 'metrics'):
        v_d, v_p, y, idx_1, idx_2, label = next(iter(data_loader))
        pred = model(v_d, v_p)
        label_ids = y.numpy()
        y_label[i] = label_ids
        y_pred[i] = pred.flatten().detach().numpy()
    y_pred = y_pred.flatten()
    y_label = y_label.flatten()
    threshold = 0.5
    y_pred_binary = np.empty(batch_total*batch_size)
    for i in range(len(y_pred_binary)):
        if y_pred[i] > threshold:
            y_pred_binary[i] = 1
        else:
            y_pred_binary[i] = 0
    auprc = average_precision_score(y_label, y_pred)
    auroc = roc_auc_score(y_label, y_pred)
    result = class_metrics(y_label, y_pred_binary)
    result['auprc'] = auprc
    result['auroc'] = auroc
    return result

def preprocess_df_id(df):
    df = df.dropna()
    df['Drug_ID'] = df['Drug_ID'].astype(str)
    df['Label'] = 1
    df['Label'][df.Y <= 30.0] = 0
    return df

def process_dti_data(df):
    df = df.dropna()
    seq_drug = df['Drug']
    seq_target = df['Target']
    seq_label = df['Label']
    graph_drug = df['Drug_ID']
    graph_target = df['Target_ID']
    graph_label = df['Label']
    df = pd.DataFrame(zip(seq_drug, seq_target, seq_label, 
                          graph_drug, graph_target, graph_label))
    df.rename(columns={0:'Seq_Drug',
                        1:'Seq_Target',
                        2:'Seq_Label',
                        3:'Graph_Drug',
                        4:'Graph_Target',
                        5:'Graph_Label'}, 
                        inplace=True)
    drug_encoding, target_encoding = 'MPNN', 'CNN'
    df = utils.encode_drug(df, drug_encoding, column_name='Seq_Drug')
    df = utils.encode_protein(df, target_encoding, column_name='Seq_Target')
    return df

def train_model(name):
    batch_size = 32
    epochs = 20
    learning_rate = 5e-4
    early_stopping = 10

    model_path = f"./src/model/"
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = f'./src/result/output/{name}/' + now + '/'
    csv_path = root_path
    log_path = root_path
    check_dir(root_path)
    check_dir(csv_path)
    check_dir(model_path)
    check_dir(log_path)
    log_fd = logger.add(log_path+"/train.log")
    
    data_dti = DTI(name=name)
    split = data_dti.get_split(method='random', seed=42, frac=[1.0, 0, 0])
    df = preprocess_df_id(split['train'])
    logger.info(f"{name}: \n{df.head(5)}")
    idx = np.concatenate((df['Drug_ID'].unique(), df['Target_ID'].unique()))
    idx_map = {j: i for i, j in enumerate(idx)}
    
    split = data_dti.get_split(method='random', seed=42, frac=[0.7, 0.1, 0.2])
    df_train = process_dti_data(preprocess_df_id(split['train']))
    df_valid = process_dti_data(preprocess_df_id(split['valid']))
    df_test = process_dti_data(preprocess_df_id(split['test']))
    logger.info(f'train: {df_train.shape}')
    logger.info(f'valid: {df_valid.shape}')
    logger.info(f'test: {df_test.shape}')
    logger.info(f'df_train: \n {df_train.head(5)}')
    
    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'drop_last': True,
                    'collate_fn': utils.mpnn_collate_func
                    }

    test_params = {'batch_size': batch_size,
                    'shuffle': False,
                    'drop_last': True,
                    'collate_fn': utils.mpnn_collate_func
                    }

    train_dataset = DTIDataset(idx_map, df_train)
    train_loader = data.DataLoader(train_dataset, **train_params)

    valid_dataset = DTIDataset(idx_map, df_valid)
    valid_loader = data.DataLoader(valid_dataset, **test_params)

    test_dataset = DTIDataset(idx_map, df_test)
    test_loader = data.DataLoader(test_dataset, **test_params)

    model = get_model().model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    logger.info('Start Training...')
    t_total = time.time()
    loss_list = []
    best_auc = 0
    best_epoch = -1
    
    for epoch in range(epochs):
        t = time.time()
        model.train()
        batch_total = len(train_loader)
        for i in tqdm(range(batch_total), f"train epoch{epoch+1}"):
            v_d, v_p, y, idx_1, idx_2, label = next(iter(train_loader))
            optimizer.zero_grad()
            pred = model(v_d, v_p)
            pred = pred.flatten()
            label = y.float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            csv_record(csv_path+"loss.csv", {'epoch':epoch+1, 'batch':i, 'loss':loss.item(), 'avg_loss':np.array(loss_list).mean()})

        save_model(model, model_path+f"/HDN_{name}_epoch{epoch+1}.pt")
        
        result = evaluate_model(model, valid_loader, batch_size)
        result['epoch'] = epoch+1
        result['epoch_loss'] = np.array(loss_list).mean()
        csv_record(csv_path+"val_metrics.csv", result)
        logger.info(f'Train: {result}')
        auroc_val, auprc_val, f1_val = result['auroc'], result['auprc'], result['f1']
        if auroc_val > best_auc:
            best_auc = auroc_val
            best_epoch = epoch+1
            logger.info(f'best_auc: {best_auc}, best_epoch: {best_epoch}')
        if (epoch - best_epoch) > early_stopping:
            logger.info(f"early stopping in epoch：{epoch+1}")
            break
        logger.info('epoch: {:04d}, '.format(epoch+1) +
                'auroc_val: {:.4f}, '.format(auroc_val) +
                'auprc_val: {:.4f}, '.format(auprc_val) +
                'f1_val: {:.4f}, '.format(f1_val) +
                'time: {:.4f}s'.format(time.time() - t))

    logger.info("Optimization Finished!")
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    model.load_state_dict(torch.load(model_path + f"/HDN_{name}_epoch{best_epoch}.pt"))
    model.eval()
    result = evaluate_model(model, test_loader, batch_size)
    csv_record(csv_path+"test_metrics.csv", result)
    logger.info(f'Test: {result}')
    logger.remove(log_fd)


if __name__ == '__main__':
    train_model('BindingDB_Kd')
    # train_model('BindingDB_IC50')
    # train_model('BindingDB_Ki')
    # train_model('DAVIS')
    # train_model('KIBA')
    