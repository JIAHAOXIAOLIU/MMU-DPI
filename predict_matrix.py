import os
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
from tdc.multi_pred import DTI
from loguru import logger
from torch.utils import data

from DeepPurpose import utils
from model import DualModelNetwork, process_dti_data, preprocess_data, DTIDataset
from MPNN import get_model
from BGAT import MixHopNetwork
from Utils import features_to_sparse, create_propagator_matrix, load_model, check_dir

def generate_matrix(name, model_path, output_path, device=torch.device('cpu')):
    """
    生成药物-蛋白质相互作用矩阵并保存为CSV文件
    
    参数:
    name: 数据集名称 ('DAVIS', 'KIBA', 'BindingDB_Kd')
    model_path: 模型路径
    output_path: 输出路径
    device: 计算设备
    """
    logger.info(f"开始处理数据集: {name}")
    
    # 确保输出路径存在
    check_dir(output_path)
    
    # 加载数据集
    data_dti = DTI(name=name)
    if name == "DAVIS":
        data_dti.convert_to_log(form='binding')
        data_dti.binarize(threshold=7, order='descending')
    elif name == "BindingDB_Kd":
        data_dti.convert_to_log(form='binding')
        data_dti.binarize(threshold=7.6, order='descending')
    elif name == "KIBA":
        data_dti.binarize(threshold=12.1, order='descending')
    else:
        logger.error(f"数据集 {name} 不支持")
        return
    
    # 获取数据
    df = data_dti.get_data()
    df = preprocess_data(df)
    
    # 获取所有唯一的药物ID和蛋白质ID
    drug_ids = df['Drug_ID'].unique()
    target_ids = df['Target_ID'].unique()
    

    
    # 创建索引映射
    idx = np.concatenate((drug_ids, target_ids))
    idx_map = {j: i for i, j in enumerate(idx)}
    
    # 处理所有数据用于测试
    df_test = process_dti_data(df)
    
    # 创建特征和邻接矩阵
    edges_unordered = df[['Drug_ID', 'Target_ID']].values
    idx_total = len(idx)
    features = np.eye(idx_total)
    features = features_to_sparse(features, device)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    propagation_matrix = create_propagator_matrix(adj, device)
    
    # 加载预训练模型
    mpnn_model = get_model().model.to(device)
    feature_number = features["dimensions"][1]
    bgat_model = MixHopNetwork(feature_number).to(device)
    model = DualModelNetwork(mpnn_model, bgat_model, propagation_matrix, features).to(device)
    
    model_file = os.path.join(model_path, f"train_{name}_epoch2.pt")
    if not os.path.exists(model_file):
        logger.error(f"模型文件不存在: {model_file}")
        return
    
    load_model(model, model_file)
    model.eval()
    
    # 创建结果矩阵
    prediction_matrix = np.zeros((len(drug_ids), len(target_ids)))
    label_matrix = np.full((len(drug_ids), len(target_ids)), "Unknown", dtype=object)
    
    # 填充已知的标签
    for index, row in df.iterrows():
        drug_idx = np.where(drug_ids == row['Drug_ID'])[0][0]
        target_idx = np.where(target_ids == row['Target_ID'])[0][0]
        label_matrix[drug_idx, target_idx] = row['Label']
    
    # 为所有药物-蛋白质对创建测试数据集
    logger.info("正在构建测试数据集...")
    
    # 创建包含所有药物-蛋白质对的大数据帧
    all_pairs_data = {
        'Drug': [],
        'Target': [],
        'Label': [],
        'Drug_ID': [],
        'Target_ID': [],
        'i': [],
        'j': []
    }
    
    for i, drug_id in enumerate(tqdm(drug_ids, desc="准备药物-蛋白质对")):
        for j, target_id in enumerate(target_ids):
            try:
                drug = df[df['Drug_ID'] == drug_id]['Drug'].values[0]
                target = df[df['Target_ID'] == target_id]['Target'].values[0]
                
                all_pairs_data['Drug'].append(drug)
                all_pairs_data['Target'].append(target)
                all_pairs_data['Label'].append(0)  # 临时标签，实际预测不使用
                all_pairs_data['Drug_ID'].append(drug_id)
                all_pairs_data['Target_ID'].append(target_id)
                all_pairs_data['i'].append(i)
                all_pairs_data['j'].append(j)
            except:
                logger.warning(f"跳过药物ID:{drug_id}或蛋白质ID:{target_id}，无法找到对应的数据")
    
    # 创建数据帧
    all_pairs_df = pd.DataFrame(all_pairs_data)
    logger.info(f"总共创建了 {len(all_pairs_df)} 个药物-蛋白质对")
    
    # 处理数据
    processed_df = process_dti_data(all_pairs_df)
    
    # 批量预测
    logger.info("开始预测所有药物-蛋白质对的相互作用...")
    
    # 设置批处理参数，使用更大的batch_size
    batch_size = 64  # 可根据GPU内存调整
    test_params = {
        'batch_size': batch_size,
        'shuffle': False,
        'drop_last': False,
        'collate_fn': utils.mpnn_collate_func
    }
    
    # 创建数据集和加载器
    test_dataset = DTIDataset(idx_map, processed_df)
    test_loader = data.DataLoader(test_dataset, **test_params)
    
    # 记录索引信息
    i_indices = all_pairs_df['i'].values
    j_indices = all_pairs_df['j'].values
    
    with torch.no_grad():
        batch_idx = 0
        for batch_data in tqdm(test_loader, desc="批量处理预测"):
            try:
                # 获取批次数据
                v_d, v_p, y, idx_1, idx_2, label = batch_data
                
                # 预测
                pred = model(v_d, v_p, idx_1, idx_2)
                pred_values = torch.sigmoid(pred).cpu().numpy()
                
                # 确定当前批次的大小
                current_batch_size = len(pred_values)
                
                # 计算当前批次在all_pairs_df中的起始索引
                start_idx = batch_idx * batch_size
                end_idx = start_idx + current_batch_size
                
                # 将预测结果写入预测矩阵
                for k in range(current_batch_size):
                    i = i_indices[start_idx + k]
                    j = j_indices[start_idx + k]
                    prediction_matrix[i, j] = pred_values[k]
                
                batch_idx += 1
            except Exception as e:
                logger.error(f"批处理预测错误: {e}")
                continue
    
    # 将结果保存为CSV文件
    prediction_df = pd.DataFrame(prediction_matrix, index=drug_ids, columns=target_ids)
    label_df = pd.DataFrame(label_matrix, index=drug_ids, columns=target_ids)
    
    prediction_path = os.path.join(output_path, f"{name}_prediction_matrix.csv")
    label_path = os.path.join(output_path, f"{name}_label_matrix.csv")
    
    prediction_df.to_csv(prediction_path)
    label_df.to_csv(label_path)
    
    logger.info(f"预测矩阵已保存至: {prediction_path}")
    logger.info(f"标签矩阵已保存至: {label_path}")

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 设置模型路径和输出路径
    model_path = "./src/model/"
    output_path = "./result/matrix/"
    
    # 执行推理
    generate_matrix('DAVIS', model_path, output_path, device) 