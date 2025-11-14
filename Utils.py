import csv
import os
import random

import torch
import scipy.sparse as sp

import numpy as np

from loguru import logger

neg_label = 1
pos_label = 0

def class_metrics(y_label, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    total = len(y_label)
    for i in range(total):
        if y_pred[i] == pos_label and y_label[i] == pos_label:
            TP = TP + 1
        elif y_pred[i] == pos_label and y_label[i] == neg_label:
            FP = FP + 1
        elif y_pred[i] == neg_label and y_label[i] == neg_label:
            TN = TN + 1
        elif y_pred[i] == neg_label and y_label[i] == pos_label:
            FN = FN + 1
        else:
            print(f"error,y_pred:{y_pred[i]},y_label:{y_label[i]}")
    logger.info(f"TP:{TP},TN:{TN},FP:{FP},FN:{FN}")
    if (TP + FN) == 0:
        sensitivity = 1.0
    else:
        sensitivity = TP / (TP + FN)
    if (FP + TN) == 0:
        specificity = 1.0
    else:
        specificity = TN / (FP + TN)
    recall = sensitivity
    if (TP + FP) == 0:
        precision = 1.0
    else:
        precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if (2*TP + FN + FP) == 0:
        f1 = 1.0
    else:
        f1 = 2*TP / (2*TP + FN + FP)
    return {"sensitivity": sensitivity, "specificity": specificity,
            "recall": recall, "precision": precision, "accuracy": accuracy, 
            "f1": f1}

def csv_record(path, data):
    all_header = ['epoch', 'batch', 'lr', 'loss', 'avg_loss', 'epoch_loss', 
                 'auprc', 'auroc', 'sensitivity', 'specificity',
                 'recall', 'precision', 'cindex', 'accuracy', 'f1']  
    row = []
    header = []
    for name in all_header:
        if name in data.keys():
            row.append(data[name])
            header.append(name)
            
    if os.path.exists(path):
        with open(path, 'a', newline='') as f:
            write = csv.writer(f)
            write.writerow(row)
    else:
        with open(path, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)
            
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"create dir: {path}")
    else:
        logger.info(f"dir exists, {path}")
        
def save_model(model, path):
    torch.save(model.state_dict(), path)
    logger.info(f"save {path} model parameters done")

def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f"load {path} model parameters done")

def load_model_fine(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        logger.info(f"load {path} model parameters done")

def normalize_adjacency_matrix(A, I):
    """
    创建带有自环的标准化邻接矩阵
    :param A: 稀疏邻接矩阵
    :param I: 单位矩阵
    :return A_tile_hat: 标准化邻接矩阵
    """
    A_tilde = A + 2 * I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sp.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat

def create_propagator_matrix(A, device):
    """
    创建传播矩阵
    :param A: 稀疏邻接矩阵
    :param device: 计算设备
    :return propagator: 矩阵索引和值的字典
    """
    I = sp.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    propagator = dict()
    A_tilde_hat = sp.coo_matrix(A_tilde_hat)
    ind = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1)
    propagator["indices"] = torch.LongTensor(ind.T).to(device)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data).to(device)
    return propagator

def features_to_sparse(features, device):
    """
    将特征矩阵转换为稀疏格式
    :param features: 特征矩阵
    :param device: 计算设备
    :return out_features: 包含索引和值张量的字典
    """
    index_1, index_2 = features.nonzero()
    values = [1.0] * len(index_1)
    node_count = features.shape[0]
    feature_count = features.shape[1]
    features = sp.coo_matrix((values, (index_1, index_2)),
                             shape=(node_count, feature_count),
                             dtype=np.float32)
    out_features = dict()
    ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
    out_features["indices"] = torch.LongTensor(ind.T).to(device)
    out_features["values"] = torch.FloatTensor(features.data).to(device)
    out_features["dimensions"] = features.shape
    return out_features

def setup_seed(seed):
    """
    设置随机种子以确保结果可重现
    :param seed: 随机种子值
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    logger.info("seed: %d, random:%.4f, torch random:%.4f, np random:%.4f" % (seed, random.random(), torch.rand(1), np.random.rand(1)))