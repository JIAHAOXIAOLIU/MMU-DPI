import os
import time
import datetime
import random

import pandas as pd # 确保 pandas 已导入
from sklearn.model_selection import StratifiedKFold
from DeepPurpose import utils

import torch
from torch import nn
from torch.utils import data
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score,\
                            f1_score, accuracy_score, auc, roc_curve
                            
import pandas as pd
from tqdm import tqdm
from loguru import logger
import pandas as pd
from tdc.multi_pred import DTI

from DeepPurpose import utils
from BGAT import MixHopNetwork

from MPNN import get_model
from Utils import csv_record, check_dir, save_model, load_model, load_model_fine, class_metrics, features_to_sparse, create_propagator_matrix, setup_seed

neg_label = 1
pos_label = 0

class DualModelNetwork(nn.Module):
    def __init__(self, mpnn_model, bgat_model, propagation_matrix, features, alpha=0.9) -> None:
        super().__init__()
        self.view1_model = mpnn_model
        self.view2_model = bgat_model
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.propagation_matrix = propagation_matrix
        self.features = features
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, v_d, v_p, idx1, idx2):
        pred1 = self.view1_model(v_d, v_p)
        pred2, _ = self.view2_model(self.propagation_matrix, self.features, (idx1, idx2))
        output = self.alpha * pred1 + (1 - self.alpha) * pred2
        return output
 
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

def calculate_threshold(label, pred):
    df_pred = pd.DataFrame(pred, columns=['pred'])
    df_pred = df_pred.sort_values(by=['pred'])
    df_label = pd.DataFrame(label, columns=['label'])
    neg_num = df_label[df_label.label == neg_label].shape[0]
    pos_num = df_label[df_label.label == pos_label].shape[0]
    threshold_idx = int(neg_num / (neg_num + pos_num) * df_pred.shape[0])
    threshold = df_pred.at[threshold_idx, 'pred']
    logger.info(f'threshold:{threshold}')
    return threshold
    
def evaluate_model(model, data_loader, batch_size):
    with torch.no_grad():
        model.eval()
        batch_total = len(data_loader)
        y_pred = np.empty([batch_total, batch_size])
        y_label = np.empty([batch_total, batch_size])
        for i in tqdm(range(batch_total), 'metrics'):
            v_d, v_p, y, idx_1, idx_2, label = next(iter(data_loader))
            pred = model(v_d, v_p, idx_1, idx_2)
            label_ids = y.numpy()
            y_label[i] = label_ids
            y_pred[i] = pred.flatten().cpu().detach().numpy()
        y_pred = y_pred.flatten()
        y_label = y_label.flatten()
        threshold = 0.5
        y_pred_binary = np.empty(batch_total*batch_size)
        for i in range(len(y_pred_binary)):
            if y_pred[i] > threshold:
                y_pred_binary[i] = neg_label
            else:
                y_pred_binary[i] = pos_label
        auprc = average_precision_score(y_label, y_pred)
        auroc = roc_auc_score(y_label, y_pred)

        result = class_metrics(y_label, y_pred_binary)
        result['auprc'] = auprc
        result['auroc'] = auroc
        return result

def analyze_samples(df):
    neg_samples = df[df.Label == neg_label]
    pos_samples = df[df.Label == pos_label]
    neg_label_num = neg_samples.shape[0]
    pos_label_num = pos_samples.shape[0]
    logger.info(f'neg/pos:{neg_label_num}/{pos_label_num}, neg:{neg_label_num * 100 //(neg_label_num + pos_label_num)}%, pos:{pos_label_num * 100 //(neg_label_num + pos_label_num)}%')
    return neg_label_num, pos_label_num

def find_unobserved_pair(df, drug_ids, target_ids):
    while(1):
        drug_id = random.sample(drug_ids, 1)[0]
        target_id = random.sample(target_ids, 1)[0]
        dfA = df[df.Drug_ID == drug_id]
        if target_id not in dfA["Target_ID"].values:
            break
    return drug_id, target_id

def generate_negative_samples(df):
    neg_samples = df[df.Label == neg_label]
    pos_samples = df[df.Label == pos_label]
    neg_label_num = neg_samples.shape[0]
    pos_label_num = pos_samples.shape[0]
    delta = pos_label_num - neg_label_num
    drug_dict = {}
    target_dict = {}
    drug_ids = list(df['Drug_ID'].unique())
    target_ids = list(df['Target_ID'].unique())
    if len(drug_ids)*len(target_ids) < delta + pos_label_num + neg_label_num:
        iter_num = pos_label_num // neg_label_num
        for _ in range(iter_num):
            df = df.append(neg_samples, ignore_index=True)
    else:
        for id in tqdm(drug_ids, "drug dict"):
            drug = df[df.Drug_ID == id].Drug.values[0]
            drug_dict[id] = drug
        for id in tqdm(target_ids, "target dict"):
            target = df[df.Target_ID == id].Target.values[0]
            target_dict[id] = target
        for _ in tqdm(range(delta), "oversampling"):
            drug_id, target_id = find_unobserved_pair(df, drug_ids, target_ids)
            row = [drug_id, drug_dict[drug_id], target_id, target_dict[target_id], neg_label]
            df = df.append(pd.Series(row, index=df.columns), ignore_index=True)
    analyze_samples(df)
    return df

def preprocess_data(df, oversampling=False, undersampling=True):
    df = df.dropna()
    df['Drug_ID'] = df['Drug_ID'].astype(str)
    df = df.rename(columns={"Y": "Label"})
    neg_label_num, pos_label_num = analyze_samples(df)
    if oversampling:
        logger.info('oversampling')
        pos_samples = df[df.Label == pos_label]
        for _ in range(1):
            df = df.append(pos_samples, ignore_index=True)
    if undersampling:
        logger.info('undersampling')
        neg_samples = df[df.Label == neg_label][:pos_label_num]
        pos_samples = df[df.Label == pos_label]
        df = pos_samples._append(neg_samples, ignore_index=True)
    analyze_samples(df)
    return df

def split_data(df, frac=[0.8, 0.1, 0.1]):
    df = df.sample(frac=1, replace=True, random_state=1)
    total = df.shape[0]
    train_idx = int(total*frac[0])
    valid_idx = int(total*(frac[0]+frac[1]))
    df_train = df.iloc[:train_idx]
    df_valid = df.iloc[train_idx:valid_idx]
    df_test = df.iloc[valid_idx:total-1]
    analyze_samples(df_train)
    analyze_samples(df_valid)
    analyze_samples(df_test)
    return df_train, df_valid, df_test

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


def execute_workflow(name, phase="train", batch_size=32, epochs=5, learning_rate=5e-4, lr_step_size=10,
                     early_stopping=10, device=torch.device('cpu'), seed_id=10, mixup=True):
    setup_seed(seed_id)

    # 路径和日志设置 (与你原来的一致)
    model_path = f"./src/model/"
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = r'./src/result/output/' + now + '/'
    csv_path = root_path
    log_path = root_path
    check_dir(root_path)
    check_dir(csv_path)
    check_dir(model_path)
    check_dir(log_path)
    log_fd = logger.add(log_path + "/train.log")


    data_dti = DTI(name=name)

    if name in "DAVIS":
        data_dti.convert_to_log(form='binding')
        data_dti.binarize(threshold=7, order='descending')
    elif name == "BindingDB_Kd":
        data_dti.convert_to_log(form='binding')
        data_dti.binarize(threshold=7.6, order='descending')
    elif name == "KIBA":
        data_dti.binarize(threshold=12.1, order='descending')
    else:
        logger.error(f"dataset {name} is not supported")
        return

    df = data_dti.get_data()
    df = preprocess_data(df)  # 确保 'Label' 列在这里被正确处理
    logger.info(f"{name} (Full Data): \n{df.head(5)}")

    idx = np.concatenate((df['Drug_ID'].unique(), df['Target_ID'].unique()))
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = df[['Drug_ID', 'Target_ID']].values
    idx_total = len(idx)
    features = np.eye(idx_total)
    features = features_to_sparse(features, device)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = create_propagator_matrix(adj, device)
    propagation_matrix = adj  # 这个传播矩阵在所有折中是共享的

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_id)

    X = df.index  # 我们可以只分割索引
    y = df['Label']  # 用标签进行分层

    all_fold_results = []  # 存储每一折的最终评估结果

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        logger.info(f"--- 启动第 {fold + 1}/{n_splits} 折 ---")

        df_train = df.iloc[train_index].reset_index(drop=True)
        df_val = df.iloc[val_index].reset_index(drop=True)

        df_train_processed = process_dti_data(df_train)
        df_val_processed = process_dti_data(df_val)

        logger.info(f'Fold {fold + 1}: train samples {len(df_train_processed)}, val samples {len(df_val_processed)}')

        # 4.3. 创建 DataLoaders
        train_params = {'batch_size': batch_size,
                        'shuffle': True,
                        'drop_last': True,
                        'collate_fn': utils.mpnn_collate_func
                        }

        # 验证集的 loader 使用 test_params (shuffle=False)
        test_params = {'batch_size': batch_size,
                       'shuffle': False,
                       'drop_last': True,
                       'collate_fn': utils.mpnn_collate_func
                       }

        train_dataset = DTIDataset(idx_map, df_train_processed)
        train_loader = data.DataLoader(train_dataset, **train_params)

        # 这一折的“验证集”就是论文中提到的“测试集”
        valid_dataset = DTIDataset(idx_map, df_val_processed)
        valid_loader = data.DataLoader(valid_dataset, **test_params)

        mpnn_model = get_model().model.to(device)
        feature_number = features["dimensions"][1]
        bgat_model = MixHopNetwork(feature_number).to(device)
        model = DualModelNetwork(mpnn_model, bgat_model, propagation_matrix, features).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1)

        if phase == 'train':
            logger.info(f'Start Training Fold {fold + 1}...')
            t_total = time.time()
            for epoch in range(epochs):
                alpha = 2
                lam = np.random.beta(alpha, alpha)
                t = time.time()
                epoch_loss = 0
                batch_total = len(train_loader)
                y_pred_train = np.empty([batch_total, batch_size])
                y_label_train = np.empty([batch_total, batch_size])

                for i in tqdm(range(batch_total), f"Fold {fold + 1}, Epoch {epoch + 1}"):
                    data_iter = iter(train_loader)
                    batch_current = next(data_iter)
                    v_d, v_p, y, idx_1, idx_2, label = batch_current
                    batch_next = next(data_iter)
                    v_d2, v_p2, y2, idx_12, idx_22, label2 = batch_next

                    optimizer.zero_grad()
                    pred = model(v_d, v_p, idx_1, idx_2).to(device)
                    if mixup == True and epoch < 7:
                        y = lam * y + (1 - lam) * y2
                    else:
                        y = y
                    pred = pred.flatten().to(device)
                    label = y.float().to(device)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    csv_record(csv_path + "loss.csv",
                               {'fold': fold + 1, 'epoch': epoch + 1, 'batch': i, 'loss': loss.item(),
                                'avg_loss': epoch_loss / (i + 1)})
                    y_label_train[i] = label.flatten().cpu().numpy()
                    y_pred_train[i] = pred.detach().flatten().cpu().numpy()

                save_model(model, model_path + f"train_{name}_fold{fold + 1}_epoch{epoch + 1}.pt")
                scheduler.step()

                logger.info(f"Fold {fold + 1}, Epoch {epoch + 1} Training loss: {loss.cpu().detach().numpy()}")

                result = evaluate_model(model, valid_loader, batch_size)
                result['fold'] = fold + 1
                result['epoch'] = epoch + 1
                result['epoch_loss'] = epoch_loss / batch_total
                result['lr'] = optimizer.state_dict()['param_groups'][0]['lr']
                csv_record(csv_path + "train_val_metrics.csv", result)
                logger.info(f'Fold {fold + 1} Epoch {epoch + 1} Val Metrics: {result}')

            logger.info(f"Fold {fold + 1} Optimization Finished!")
            logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        final_fold_result = evaluate_model(model, valid_loader, batch_size)
        final_fold_result['fold'] = fold + 1
        logger.info(f"--- Fold {fold + 1} Final Result: {final_fold_result} ---")
        all_fold_results.append(final_fold_result)
        csv_record(csv_path + "fold_final_metrics.csv", final_fold_result)

    logger.info("--- 10-Fold Cross-Validation Finished ---")
    df_results = pd.DataFrame(all_fold_results)

    avg_results = df_results.mean().to_dict()
    std_results = df_results.std().to_dict()

    logger.info(f"Average CV Results: {avg_results}")
    logger.info(f"Std Dev CV Results: {std_results}")

    csv_record(csv_path + "final_avg_metrics.csv", avg_results)
    csv_record(csv_path + "final_std_metrics.csv", std_results)

    logger.remove(log_fd)

if __name__ == '__main__':
    execute_workflow('DAVIS')
    