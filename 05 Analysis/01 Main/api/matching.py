# ------------------------------
# Imports
# ------------------------------
import numpy as np
import pandas as pd
import cudf
import cupy as cp # Ensure cupy is installed with CUDA support
import torch
import torch.nn.functional as F
from tqdm import tqdm
import dill as pickle

# ------------------------------
# 1. Loading functions
# ------------------------------

def load_pickle(filename):
    """Load a pickle file using dill."""
    
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_data(citations_path = "/content/drive/MyDrive/PhD Data/08 Citations/03 Patent citations - raw, filing.pickle", 
              treated_path = "/content/drive/MyDrive/PhD Data/10 Sample - pre final/acquired_patents.pkl", 
              control_path = "/content/drive/MyDrive/PhD Data/10 Sample - pre final/potential_controls.pkl", 
              clean_ids_path = "/content/drive/MyDrive/PhD Data/10 Sample - pre final/clean_potential_control_ids.csv"):
    """Load citations, treated, control data, and clean ids."""
    citations, treated, control, clean_ids = (
        pd.read_pickle(citations_path),
        pd.read_pickle(treated_path),
        pd.read_pickle(control_path),
        pd.read_csv(clean_ids_path)
    )
    return citations, treated, control, clean_ids


# ------------------------------
# 2. Citation preprocessing
# ------------------------------

def preprocess_citations(citations, treated, control, clean_ids):
    """Preprocess citation data and remove unwanted patents."""
    citations = cudf.DataFrame.from_pandas(citations)
    citations['patent_id'] = citations['patent_id'].astype(str)
    citations['citation_patent_id'] = citations['citation_patent_id'].astype(str)
    citations['citation_date'] = cudf.to_datetime(citations['citation_date'])
    citations['year'] = citations['citation_date'].dt.year
    citations['month'] = citations['citation_date'].dt.month
    citations['qtr'] = ((citations['month'] - 1) // 3 + 1).astype(str)
    citations['citation_quarter'] = citations['year'].astype(str) + 'Q' + citations['qtr']

    clean_set = set(clean_ids['patent_id'].astype(str))
    treated['patent_id'] = treated['patent_id'].astype(str)
    control['patent_id'] = control['patent_id'].astype(str)
    control = control[control['patent_id'].isin(clean_set)]

    valid_ids = set(treated['patent_id']).union(control['patent_id'])
    citations = citations[citations['patent_id'].isin(valid_ids)]
    return citations, treated, control


def compute_quarterly_citation_counts(citations):
    """Compute quarterly citation counts for each patent."""
    grouped = citations.groupby(['patent_id', 'citation_quarter']).size()
    quarterly_counts_pd = grouped.to_pandas().unstack(fill_value=0)
    return quarterly_counts_pd


def build_citation_counts_dict(quarterly_counts_pd):
    """Build a dictionary mapping patent_id to citation counts."""
    citation_counts_dict = {}
    for pid, row in quarterly_counts_pd.iterrows():
        citation_counts_dict[pid] = row.to_dict()
    return citation_counts_dict


# ------------------------------
# 3. Treated patents vector preparation
# ------------------------------

def compute_treated_vectors(treated, citation_counts_dict):
    """Compute pre-treatment citation vectors for treated patents."""
    treated['year'] = treated['acq_date'].dt.year
    treated['month'] = treated['acq_date'].dt.month
    treated['qtr'] = ((treated['month'] - 1) // 3 + 1).astype(str)
    treated['acq_quarter'] = treated['year'].astype(str) + 'Q' + treated['qtr']

    treated_counts_dict = {}
    for i, row in treated.iterrows():
        treated_id = row['patent_id']
        acq_period = pd.Period(row['acq_date'], freq='Q')
        pre_quarters = [str(acq_period - j) for j in range(8, 4, -1)]
        vec = [citation_counts_dict.get(treated_id, {}).get(q, 0) for q in pre_quarters]
        treated_counts_dict[treated_id] = {'pre_quarters': pre_quarters, 'vector': np.array(vec, dtype=float)}
    return treated_counts_dict


# ------------------------------
# 4. Distance calculation functions
# ------------------------------

def compute_cosine_distances(treated, control):
    """Compute cosine distances between treated and control embeddings."""
    cosine_distance_by_treated = {}
    group_cols = ['grant_year', 'cpc_subclass']
    treated_groups = treated.groupby(group_cols)

    for group_key, group in tqdm(treated_groups, total=len(treated_groups), desc="Precompute Cosine Distances"):
        grant_year_val, cpc_subclass_val = group_key
        candidates = control[(control['grant_year'] == grant_year_val) & (control['cpc_subclass'] == cpc_subclass_val)]
        if candidates.empty:
            continue
        candidate_embeddings = np.stack(candidates['embedding'].values)
        cand_emb = torch.tensor(candidate_embeddings, dtype=torch.float16, device='cuda')

        for _, treated_row in group.iterrows():
            tid = treated_row['patent_id']
            t_embed_np = treated_row['embedding']
            t_embed = torch.tensor(t_embed_np, dtype=torch.float16, device='cuda').view(1, -1)
            cos_sim = F.cosine_similarity(t_embed, cand_emb, dim=1)
            d_e = 1 - cos_sim.cpu().numpy()
            cosine_distance_by_treated[tid] = d_e

    return cosine_distance_by_treated


def compute_hybrid_distance(d_c, d_e, lam, alpha):
    """Compute the hybrid distance matrix using Mahalanobis and cosine distances."""
    d_c_scaled = 1 - np.exp(-alpha * d_c)
    d_e_scaled = d_e / 2
    d_h = lam * d_c_scaled + (1 - lam) * d_e_scaled
    return d_h


# ------------------------------
# 5. Matching
# ------------------------------

def hybrid_matching_for_lambda(lam, treated_df, control_df, treated_counts_dict, citation_counts_dict, alpha, cosine_distance_by_treated):
    """Perform hybrid matching between treated and control patents for a given lambda."""
    control_group_dict = {key: group for key, group in control_df.groupby(['grant_year', 'cpc_subclass'])}
    matches = []

    treated_groups = list(treated_df.groupby(['acq_quarter', 'grant_year', 'cpc_subclass']))

    for group_key, group in tqdm(treated_groups, total=len(treated_groups), desc="Hybrid Matching Groups"):
        acq_quarter, grant_year, cpc_subclass = group_key
        acq_period = pd.Period(acq_quarter, freq='Q')
        pre_quarters = [str(acq_period - i) for i in range(8, 4, -1)]

        candidates = control_group_dict.get((grant_year, cpc_subclass), pd.DataFrame())
        if candidates.empty:
            continue

        candidate_ids = candidates['patent_id'].tolist()
        candidate_vectors = [
            [citation_counts_dict.get(cid, {}).get(q, 0) for q in pre_quarters]
            for cid in candidate_ids
        ]

        candidate_matrix = cp.array(candidate_vectors, dtype=cp.float64)
        if candidate_matrix.shape[0] > 1:
            cov_matrix = cp.cov(candidate_matrix, rowvar=False)
        else:
            cov_matrix = cp.eye(4, dtype=cp.float64)
        inv_cov = cp.linalg.pinv(cov_matrix)

        treated_ids = []
        treated_vectors = []
        cosine_list = []

        for _, row in group.iterrows():
            tid = row['patent_id']
            d_e = cosine_distance_by_treated.get(tid)
            if d_e is None:
                continue
            treated_ids.append(tid)
            treated_info = treated_counts_dict.get(tid, {'pre_quarters': pre_quarters, 'vector': np.zeros(4)})
            treated_vectors.append(treated_info['vector'])
            cosine_list.append(d_e)

        if len(treated_vectors) == 0:
            continue

        T = cp.array(treated_vectors, dtype=cp.float64)
        diff = candidate_matrix[None, :, :] - T[:, None, :]
        d_c_sq = cp.sum((diff @ inv_cov) * diff, axis=2)
        d_c = cp.sqrt(d_c_sq)
        cp.cuda.Stream.null.synchronize()
        d_c_np = cp.asnumpy(d_c)

        d_h = compute_hybrid_distance(d_c_np, np.stack(cosine_list, axis=0), lam, alpha)

        best_indices = np.argmin(d_h, axis=1)

        for i, tid in enumerate(treated_ids):
            best_idx = best_indices[i]
            matches.append({
                'treated_id': tid,
                'control_id': candidate_ids[best_idx],
                'treated_vector': treated_vectors[i],
                'control_vector': candidate_vectors[best_idx],
                'mahalanobis_distance': float(d_c_np[i, best_idx]),
                'cosine_distance': float(cosine_list[i][best_idx]),
                'hybrid_distance': float(d_h[i, best_idx]),
                'pre_quarters': pre_quarters
            })

    return pd.DataFrame(matches)
