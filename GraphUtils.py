import hashlib

import numpy as np
import scipy.sparse as sp
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
except ImportError:  # pragma: no cover
    Chem = None
    AllChem = None
    DataStructs = None

from Utils import create_propagator_matrix, features_to_sparse


DRUG_FP_DIM = 2048
PROTEIN_FP_DIM = 1024
PROTEIN_KMER = 3
MORGAN_RADIUS = 2

_DRUG_FP_CACHE = {}
_PROTEIN_FP_CACHE = {}


def drug_node_key(drug_id):
    return f"drug::{str(drug_id)}"


def target_node_key(target_id):
    return f"target::{target_id}"


def _stable_bucket(token, dim):
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


def _set_hashed_bits(tokens, dim):
    fp = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        if token:
            fp[_stable_bucket(token, dim)] = 1.0
    return fp


def _smiles_fallback_tokens(smiles):
    tokens = [smiles]
    for n in (2, 3, 4):
        upper = max(len(smiles) - n + 1, 0)
        tokens.extend(smiles[i:i + n] for i in range(upper))
    return tokens


def smiles_to_fingerprint(smiles, radius=MORGAN_RADIUS, dim=DRUG_FP_DIM):
    smiles = str(smiles)
    if smiles in _DRUG_FP_CACHE:
        return _DRUG_FP_CACHE[smiles]

    fp = np.zeros(dim, dtype=np.float32)
    if Chem is not None and AllChem is not None and DataStructs is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=dim)
            DataStructs.ConvertToNumpyArray(bit_vector, fp)
        else:
            fp = _set_hashed_bits(_smiles_fallback_tokens(smiles), dim)
    else:
        fp = _set_hashed_bits(_smiles_fallback_tokens(smiles), dim)

    _DRUG_FP_CACHE[smiles] = fp
    return fp


def protein_to_kmer_fingerprint(sequence, dim=PROTEIN_FP_DIM, k=PROTEIN_KMER):
    sequence = str(sequence)
    if sequence in _PROTEIN_FP_CACHE:
        return _PROTEIN_FP_CACHE[sequence]

    tokens = [sequence[i:i + k] for i in range(max(len(sequence) - k + 1, 0))]
    if not tokens:
        tokens = [sequence]
    fp = _set_hashed_bits(tokens, dim)
    _PROTEIN_FP_CACHE[sequence] = fp
    return fp


def _ordered_unique(values, extra_values=None):
    ordered = []
    seen = set()
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    if extra_values is not None:
        for value in extra_values:
            if value not in seen:
                ordered.append(value)
                seen.add(value)
    return ordered


def build_dti_graph_state(
    df,
    device,
    extra_drug_records=None,
    extra_target_records=None,
    drug_fp_dim=DRUG_FP_DIM,
    protein_fp_dim=PROTEIN_FP_DIM,
):
    graph_df = df.dropna(subset=["Drug_ID", "Drug", "Target_ID", "Target"]).copy()
    graph_df["Drug_ID"] = graph_df["Drug_ID"].astype(str)

    drug_records = (
        graph_df[["Drug_ID", "Drug"]]
        .drop_duplicates(subset=["Drug_ID"])
        .set_index("Drug_ID")["Drug"]
        .to_dict()
    )
    target_records = (
        graph_df[["Target_ID", "Target"]]
        .drop_duplicates(subset=["Target_ID"])
        .set_index("Target_ID")["Target"]
        .to_dict()
    )

    if extra_drug_records:
        drug_records.update({str(k): v for k, v in extra_drug_records.items()})
    if extra_target_records:
        target_records.update(extra_target_records)

    drug_ids = _ordered_unique(graph_df["Drug_ID"].tolist(), drug_records.keys())
    target_ids = _ordered_unique(graph_df["Target_ID"].tolist(), target_records.keys())
    node_ids = [drug_node_key(drug_id) for drug_id in drug_ids] + [
        target_node_key(target_id) for target_id in target_ids
    ]
    idx_map = {node_id: i for i, node_id in enumerate(node_ids)}

    feature_dim = drug_fp_dim + protein_fp_dim
    features = np.zeros((len(node_ids), feature_dim), dtype=np.float32)
    for drug_id in drug_ids:
        features[idx_map[drug_node_key(drug_id)], :drug_fp_dim] = smiles_to_fingerprint(
            drug_records[drug_id], dim=drug_fp_dim
        )
    for target_id in target_ids:
        features[idx_map[target_node_key(target_id)], drug_fp_dim:] = protein_to_kmer_fingerprint(
            target_records[target_id], dim=protein_fp_dim
        )

    edge_df = graph_df[["Drug_ID", "Target_ID"]].drop_duplicates()
    edge_rows = []
    edge_cols = []
    for drug_id, target_id in edge_df.itertuples(index=False):
        drug_id = str(drug_id)
        drug_key = drug_node_key(drug_id)
        target_key = target_node_key(target_id)
        if drug_key in idx_map and target_key in idx_map:
            edge_rows.append(idx_map[drug_key])
            edge_cols.append(idx_map[target_key])

    adj = sp.coo_matrix(
        (np.ones(len(edge_rows), dtype=np.float32), (edge_rows, edge_cols)),
        shape=(len(node_ids), len(node_ids)),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return {
        "propagation_matrix": create_propagator_matrix(adj, device),
        "features": features_to_sparse(features, device),
        "idx_map": idx_map,
        "feature_number": feature_dim,
        "drug_ids": drug_ids,
        "target_ids": target_ids,
    }


def resolve_pair_indices(drug_ids, target_ids, idx_map, device):
    idx_1 = torch.tensor(
        [idx_map[drug_node_key(drug_id)] for drug_id in drug_ids],
        dtype=torch.long,
        device=device,
    )
    idx_2 = torch.tensor(
        [idx_map[target_node_key(target_id)] for target_id in target_ids],
        dtype=torch.long,
        device=device,
    )
    return idx_1, idx_2


def build_dynamic_graph_for_sample(
    reference_df,
    known_drugs,
    known_targets,
    drug_id,
    drug_smiles,
    target_id,
    target_sequence,
    device,
):
    extra_drug_records = None
    extra_target_records = None
    if str(drug_id) not in known_drugs:
        extra_drug_records = {str(drug_id): drug_smiles}
    if target_id not in known_targets:
        extra_target_records = {target_id: target_sequence}
    return build_dti_graph_state(
        reference_df,
        device=device,
        extra_drug_records=extra_drug_records,
        extra_target_records=extra_target_records,
    )
