import os

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils import data
from tqdm import tqdm
from tdc.multi_pred import DTI

from BGAT import MixHopNetwork
from GraphUtils import build_dti_graph_state, build_dynamic_graph_for_sample, resolve_pair_indices
from MPNN import get_model
from Utils import check_dir, load_model
from model_fingerprint import DTIDataset, DualModelNetwork, dti_collate_fn, preprocess_data, process_dti_data


def generate_matrix(name, model_path, output_path, device=torch.device("cpu")):
    logger.info(f"Start processing dataset: {name}")
    check_dir(output_path)

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
        return

    df = preprocess_data(data_dti.get_data())
    graph_state = build_dti_graph_state(df, device=device)

    drug_ids = np.array(graph_state["drug_ids"])
    target_ids = np.array(graph_state["target_ids"])

    mpnn_model = get_model().model.to(device)
    bgat_model = MixHopNetwork(graph_state["feature_number"], device=device).to(device)
    model = DualModelNetwork(
        mpnn_model,
        bgat_model,
        graph_state["propagation_matrix"],
        graph_state["features"],
    ).to(device)

    model_file = os.path.join(model_path, f"train_{name}_epoch2.pt")
    if not os.path.exists(model_file):
        logger.error(f"model file does not exist: {model_file}")
        return
    load_model(model, model_file)
    model = model.to(device)
    model.eval()

    prediction_matrix = np.zeros((len(drug_ids), len(target_ids)))
    label_matrix = np.full((len(drug_ids), len(target_ids)), "Unknown", dtype=object)
    for _, row in df.iterrows():
        i = np.where(drug_ids == str(row["Drug_ID"]))[0][0]
        j = np.where(target_ids == row["Target_ID"])[0][0]
        label_matrix[i, j] = row["Label"]

    all_pairs = []
    for i, drug_id in enumerate(tqdm(drug_ids, desc="prepare pairs")):
        drug_smiles = df[df["Drug_ID"].astype(str) == str(drug_id)]["Drug"].values[0]
        for j, target_id in enumerate(target_ids):
            target_sequence = df[df["Target_ID"] == target_id]["Target"].values[0]
            all_pairs.append(
                {
                    "Drug": drug_smiles,
                    "Target": target_sequence,
                    "Label": 0,
                    "Drug_ID": str(drug_id),
                    "Target_ID": target_id,
                    "i": i,
                    "j": j,
                }
            )

    all_pairs_df = pd.DataFrame(all_pairs)
    processed_df = process_dti_data(all_pairs_df)
    loader = data.DataLoader(
        DTIDataset(processed_df),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=dti_collate_fn,
    )

    known_drugs = set(df["Drug_ID"].astype(str))
    known_targets = set(df["Target_ID"])

    with torch.no_grad():
        for row_idx, batch in enumerate(tqdm(loader, desc="dynamic graph inference")):
            (
                v_d,
                v_p,
                _,
                drug_id_batch,
                target_id_batch,
                drug_smiles_batch,
                target_sequence_batch,
            ) = batch

            graph_state = build_dynamic_graph_for_sample(
                reference_df=df,
                known_drugs=known_drugs,
                known_targets=known_targets,
                drug_id=drug_id_batch[0],
                drug_smiles=drug_smiles_batch[0],
                target_id=target_id_batch[0],
                target_sequence=target_sequence_batch[0],
                device=device,
            )
            idx_1, idx_2 = resolve_pair_indices(
                drug_id_batch,
                target_id_batch,
                graph_state["idx_map"],
                device,
            )
            pred = model(
                v_d,
                v_p,
                idx_1,
                idx_2,
                propagation_matrix=graph_state["propagation_matrix"],
                features=graph_state["features"],
            )
            pred_value = torch.sigmoid(pred).cpu().item()
            prediction_matrix[all_pairs_df.iloc[row_idx]["i"], all_pairs_df.iloc[row_idx]["j"]] = pred_value

    prediction_df = pd.DataFrame(prediction_matrix, index=drug_ids, columns=target_ids)
    label_df = pd.DataFrame(label_matrix, index=drug_ids, columns=target_ids)
    prediction_path = os.path.join(output_path, f"{name}_prediction_matrix.csv")
    label_path = os.path.join(output_path, f"{name}_label_matrix.csv")
    prediction_df.to_csv(prediction_path)
    label_df.to_csv(label_path)
    logger.info(f"prediction matrix saved to: {prediction_path}")
    logger.info(f"label matrix saved to: {label_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"using device: {device}")
    generate_matrix("DAVIS", "./src/model/", "./result/matrix/", device)
