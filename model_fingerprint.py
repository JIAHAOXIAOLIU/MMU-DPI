import datetime
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.optim.lr_scheduler as lr_scheduler
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils import data
from tqdm import tqdm
from tdc.multi_pred import DTI

from DeepPurpose import utils

from BGAT import MixHopNetwork
from GraphUtils import build_dti_graph_state, build_dynamic_graph_for_sample, resolve_pair_indices
from MPNN import get_model
from Utils import check_dir, class_metrics, csv_record, save_model, setup_seed


neg_label = 1
pos_label = 0


class DualModelNetwork(nn.Module):
    def __init__(self, mpnn_model, bgat_model, propagation_matrix=None, features=None, alpha=0.9):
        super().__init__()
        self.view1_model = mpnn_model
        self.view2_model = bgat_model
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.propagation_matrix = propagation_matrix
        self.features = features

    def set_graph_state(self, propagation_matrix, features):
        self.propagation_matrix = propagation_matrix
        self.features = features

    def forward(self, v_d, v_p, idx1, idx2, propagation_matrix=None, features=None):
        propagation_matrix = propagation_matrix or self.propagation_matrix
        features = features or self.features
        if propagation_matrix is None or features is None:
            raise ValueError("Graph state is required before BGAT inference.")

        pred1 = self.view1_model(v_d, v_p)
        pred2, _ = self.view2_model(propagation_matrix, features, (idx1, idx2))
        return self.alpha * pred1 + (1 - self.alpha) * pred2


class DTIDataset(data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        v_d = row.drug_encoding
        v_p = utils.protein_2_embed(row.target_encoding)
        y = float(row.Seq_Label)
        drug_id = str(row.Graph_Drug)
        target_id = row.Graph_Target
        drug_smiles = row.Seq_Drug
        target_sequence = row.Seq_Target
        return v_d, v_p, y, drug_id, target_id, drug_smiles, target_sequence


def dti_collate_fn(batch):
    mpnn_batch = [(item[0], item[1], item[2]) for item in batch]
    v_d, v_p, y = utils.mpnn_collate_func(mpnn_batch)
    drug_ids = [item[3] for item in batch]
    target_ids = [item[4] for item in batch]
    drug_smiles = [item[5] for item in batch]
    target_sequences = [item[6] for item in batch]
    return v_d, v_p, y, drug_ids, target_ids, drug_smiles, target_sequences


def calculate_threshold(label, pred):
    df_pred = pd.DataFrame(pred, columns=["pred"]).sort_values(by=["pred"])
    df_label = pd.DataFrame(label, columns=["label"])
    neg_num = df_label[df_label.label == neg_label].shape[0]
    pos_num = df_label[df_label.label == pos_label].shape[0]
    threshold_idx = int(neg_num / (neg_num + pos_num) * df_pred.shape[0])
    threshold = df_pred.iloc[threshold_idx]["pred"]
    logger.info(f"threshold:{threshold}")
    return threshold


def _safe_auc(metric_fn, y_true, y_score):
    try:
        return metric_fn(y_true, y_score)
    except ValueError:
        return float("nan")


def evaluate_model(model, data_loader, reference_df, device):
    model.eval()
    known_drugs = set(reference_df["Drug_ID"].astype(str))
    known_targets = set(reference_df["Target_ID"])
    y_pred = []
    y_label = []

    with torch.no_grad():
        for batch in tqdm(data_loader, "metrics"):
            (
                v_d,
                v_p,
                y,
                drug_ids,
                target_ids,
                drug_smiles,
                target_sequences,
            ) = batch

            graph_state = build_dynamic_graph_for_sample(
                reference_df=reference_df,
                known_drugs=known_drugs,
                known_targets=known_targets,
                drug_id=drug_ids[0],
                drug_smiles=drug_smiles[0],
                target_id=target_ids[0],
                target_sequence=target_sequences[0],
                device=device,
            )
            idx_1, idx_2 = resolve_pair_indices(drug_ids, target_ids, graph_state["idx_map"], device)
            pred = model(
                v_d,
                v_p,
                idx_1,
                idx_2,
                propagation_matrix=graph_state["propagation_matrix"],
                features=graph_state["features"],
            )
            pred = torch.sigmoid(pred)
            y_label.extend(y.detach().cpu().numpy().flatten().tolist())
            y_pred.extend(pred.detach().cpu().flatten().tolist())

    y_pred = np.asarray(y_pred)
    y_label = np.asarray(y_label)
    y_pred_binary = np.where(y_pred > 0.5, neg_label, pos_label)

    result = class_metrics(y_label, y_pred_binary)
    result["auprc"] = _safe_auc(average_precision_score, y_label, y_pred)
    result["auroc"] = _safe_auc(roc_auc_score, y_label, y_pred)
    return result


def analyze_samples(df):
    neg_samples = df[df.Label == neg_label]
    pos_samples = df[df.Label == pos_label]
    neg_label_num = neg_samples.shape[0]
    pos_label_num = pos_samples.shape[0]
    logger.info(
        f"neg/pos:{neg_label_num}/{pos_label_num}, neg:{neg_label_num * 100 // (neg_label_num + pos_label_num)}%, pos:{pos_label_num * 100 // (neg_label_num + pos_label_num)}%"
    )
    return neg_label_num, pos_label_num


def find_unobserved_pair(df, drug_ids, target_ids):
    while True:
        drug_id = random.sample(drug_ids, 1)[0]
        target_id = random.sample(target_ids, 1)[0]
        df_a = df[df.Drug_ID == drug_id]
        if target_id not in df_a["Target_ID"].values:
            return drug_id, target_id


def generate_negative_samples(df):
    neg_samples = df[df.Label == neg_label]
    pos_samples = df[df.Label == pos_label]
    neg_label_num = neg_samples.shape[0]
    pos_label_num = pos_samples.shape[0]
    delta = pos_label_num - neg_label_num
    drug_dict = {}
    target_dict = {}
    drug_ids = list(df["Drug_ID"].unique())
    target_ids = list(df["Target_ID"].unique())
    if len(drug_ids) * len(target_ids) < delta + pos_label_num + neg_label_num:
        iter_num = max(pos_label_num // max(neg_label_num, 1), 1)
        for _ in range(iter_num):
            df = df._append(neg_samples, ignore_index=True)
    else:
        for drug_id in tqdm(drug_ids, "drug dict"):
            drug_dict[drug_id] = df[df.Drug_ID == drug_id].Drug.values[0]
        for target_id in tqdm(target_ids, "target dict"):
            target_dict[target_id] = df[df.Target_ID == target_id].Target.values[0]
        for _ in tqdm(range(delta), "oversampling"):
            drug_id, target_id = find_unobserved_pair(df, drug_ids, target_ids)
            row = [drug_id, drug_dict[drug_id], target_id, target_dict[target_id], neg_label]
            df = df._append(pd.Series(row, index=df.columns), ignore_index=True)
    analyze_samples(df)
    return df


def preprocess_data(df, oversampling=False, undersampling=True):
    df = df.dropna().copy()
    df["Drug_ID"] = df["Drug_ID"].astype(str)
    df = df.rename(columns={"Y": "Label"})
    _, pos_label_num = analyze_samples(df)
    if oversampling:
        logger.info("oversampling")
        pos_samples = df[df.Label == pos_label]
        df = df._append(pos_samples, ignore_index=True)
    if undersampling:
        logger.info("undersampling")
        neg_samples = df[df.Label == neg_label][:pos_label_num]
        pos_samples = df[df.Label == pos_label]
        df = pos_samples._append(neg_samples, ignore_index=True)
    analyze_samples(df)
    return df


def process_dti_data(df):
    df = df.dropna().copy()
    seq_drug = df["Drug"]
    seq_target = df["Target"]
    seq_label = df["Label"]
    graph_drug = df["Drug_ID"].astype(str)
    graph_target = df["Target_ID"]
    graph_label = df["Label"]
    df = pd.DataFrame(
        zip(seq_drug, seq_target, seq_label, graph_drug, graph_target, graph_label)
    )
    df.rename(
        columns={
            0: "Seq_Drug",
            1: "Seq_Target",
            2: "Seq_Label",
            3: "Graph_Drug",
            4: "Graph_Target",
            5: "Graph_Label",
        },
        inplace=True,
    )
    drug_encoding, target_encoding = "MPNN", "CNN"
    df = utils.encode_drug(df, drug_encoding, column_name="Seq_Drug")
    df = utils.encode_protein(df, target_encoding, column_name="Seq_Target")
    return df


def execute_workflow(
    name,
    phase="train",
    batch_size=32,
    epochs=5,
    learning_rate=5e-4,
    lr_step_size=10,
    early_stopping=10,
    device=torch.device("cpu"),
    seed_id=10,
    mixup=True,
):
    setup_seed(seed_id)

    model_path = "./src/model/"
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = f"./src/result/output/{now}/"
    csv_path = root_path
    log_path = root_path
    check_dir(root_path)
    check_dir(csv_path)
    check_dir(model_path)
    check_dir(log_path)
    log_fd = logger.add(log_path + "/train.log")

    data_dti = DTI(name=name)
    if name == "DAVIS":
        data_dti.convert_to_log(form="binding")
        data_dti.binarize(threshold=7, order="descending")
    elif name == "BindingDB_Kd":
        data_dti.convert_to_log(form="binding")
        data_dti.binarize(threshold=7.6, order="descending")
    elif name == "KIBA":
        data_dti.binarize(threshold=12.1, order="descending")
    else:
        logger.error(f"dataset {name} is not supported")
        logger.remove(log_fd)
        return

    df = preprocess_data(data_dti.get_data())
    logger.info(f"{name} (Full Data): \n{df.head(5)}")

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_id)
    X = df.index
    y = df["Label"]
    all_fold_results = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        logger.info(f"--- Start Fold {fold + 1}/{n_splits} ---")

        df_train = df.iloc[train_index].reset_index(drop=True)
        df_val = df.iloc[val_index].reset_index(drop=True)

        train_graph_state = build_dti_graph_state(df_train, device=device)
        df_train_processed = process_dti_data(df_train)
        df_val_processed = process_dti_data(df_val)
        logger.info(
            f"Fold {fold + 1}: train samples {len(df_train_processed)}, val samples {len(df_val_processed)}"
        )

        train_loader = data.DataLoader(
            DTIDataset(df_train_processed),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=dti_collate_fn,
        )
        valid_loader = data.DataLoader(
            DTIDataset(df_val_processed),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=dti_collate_fn,
        )

        mpnn_model = get_model().model.to(device)
        bgat_model = MixHopNetwork(train_graph_state["feature_number"], device=device).to(device)
        model = DualModelNetwork(
            mpnn_model,
            bgat_model,
            train_graph_state["propagation_matrix"],
            train_graph_state["features"],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1)

        best_auroc = -np.inf
        patience = 0

        if phase == "train":
            logger.info(f"Start Training Fold {fold + 1}...")
            t_total = time.time()
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                batch_total = max(len(train_loader), 1)

                for batch_idx, batch in enumerate(tqdm(train_loader, f"Fold {fold + 1}, Epoch {epoch + 1}")):
                    (
                        v_d,
                        v_p,
                        y_batch,
                        drug_ids,
                        target_ids,
                        _,
                        _,
                    ) = batch

                    idx_1, idx_2 = resolve_pair_indices(
                        drug_ids, target_ids, train_graph_state["idx_map"], device
                    )
                    optimizer.zero_grad()
                    pred = model(v_d, v_p, idx_1, idx_2).flatten()
                    label = y_batch.float().to(device)

                    if mixup and epoch < 7 and label.shape[0] > 1:
                        lam = np.random.beta(2, 2)
                        perm = torch.randperm(label.shape[0], device=label.device)
                        label = lam * label + (1 - lam) * label[perm]

                    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    csv_record(
                        csv_path + "loss.csv",
                        {
                            "fold": fold + 1,
                            "epoch": epoch + 1,
                            "batch": batch_idx,
                            "loss": loss.item(),
                            "avg_loss": epoch_loss / (batch_idx + 1),
                        },
                    )

                save_model(model, model_path + f"train_{name}_fold{fold + 1}_epoch{epoch + 1}.pt")
                scheduler.step()

                result = evaluate_model(model, valid_loader, df_train, device)
                result["fold"] = fold + 1
                result["epoch"] = epoch + 1
                result["epoch_loss"] = epoch_loss / batch_total
                result["lr"] = optimizer.state_dict()["param_groups"][0]["lr"]
                csv_record(csv_path + "train_val_metrics.csv", result)
                logger.info(f"Fold {fold + 1} Epoch {epoch + 1} Val Metrics: {result}")

                current_auroc = result.get("auroc", float("nan"))
                if np.isnan(current_auroc) or current_auroc <= best_auroc:
                    patience += 1
                else:
                    best_auroc = current_auroc
                    patience = 0

                if patience > early_stopping:
                    logger.info(f"Early stopping at fold {fold + 1}, epoch {epoch + 1}")
                    break

            logger.info(f"Fold {fold + 1} Optimization Finished!")
            logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        final_fold_result = evaluate_model(model, valid_loader, df_train, device)
        final_fold_result["fold"] = fold + 1
        logger.info(f"--- Fold {fold + 1} Final Result: {final_fold_result} ---")
        all_fold_results.append(final_fold_result)
        csv_record(csv_path + "fold_final_metrics.csv", final_fold_result)

    logger.info("--- 10-Fold Cross-Validation Finished ---")
    df_results = pd.DataFrame(all_fold_results)
    avg_results = df_results.mean(numeric_only=True).to_dict()
    std_results = df_results.std(numeric_only=True).to_dict()
    logger.info(f"Average CV Results: {avg_results}")
    logger.info(f"Std Dev CV Results: {std_results}")
    csv_record(csv_path + "final_avg_metrics.csv", avg_results)
    csv_record(csv_path + "final_std_metrics.csv", std_results)
    logger.remove(log_fd)


if __name__ == "__main__":
    execute_workflow("DAVIS")
