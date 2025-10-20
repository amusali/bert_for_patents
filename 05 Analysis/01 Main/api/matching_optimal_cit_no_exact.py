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

def load_data(citations_path = "/content/drive/MyDrive/PhD Data/08 Citations/03 Patent citations - raw, filing.pickle", 
              treated_path = "/content/drive/MyDrive/PhD Data/10 Sample - pre final/acquired_patents_noresold.pkl", 
              control_path = "/content/drive/MyDrive/PhD Data/10 Sample - pre final/potential_controls_no_exact_match_grantyear_cpc.pkl", 
              clean_ids_path = "/content/drive/MyDrive/PhD Data/10 Sample - pre final/clean_potential_control_ids_no_exact_match_grantyear.csv",
              acq_type_path = "/content/drive/MyDrive/PhD Data/09 Acquired patents/04 All patents.dta",
              ):
    
    """Load citations, treated, control data, clean ids, and acquistion type data."""

    # Always reload treated, as it's the only one that must be reloaded
    treated = pd.read_pickle(treated_path)
    assert len(treated) == 19649
    treated['patent_id'] = treated['patent_id'].astype(str)
    
    # Check if the other data is already loaded, if not, load it
    if 'citations' not in locals():
        citations = pd.read_pickle(citations_path)

    if 'control' not in locals():
        control = pd.read_pickle(control_path)
        control['patent_id'] = control['patent_id'].astype(str)

    if 'clean_ids' not in locals():
        clean_ids = pd.read_csv(clean_ids_path)
        clean_ids['patent_id'] = clean_ids['patent_id'].astype(str)

    if 'acq_types' not in locals():
        acq_types = pd.read_stata(acq_type_path)
        acq_types['patent_id'] = acq_types['patent_id'].astype(str)

    # Merge treated with acq_types to get acquisition type
    treated = treated.merge(acq_types[['patent_id', 'acq_type', 'deal_id']], on='patent_id', how='left')

    return citations, treated, control, clean_ids

## Adjust treated sample based on type and whether it is top tech or not
def adjust_treated_sample(treated, acq_type, treated_counts_dict, top_tech=False, top_tech_threshold = 90):
    """Adjust treated sample based on acquisition type and top tech status."""
    # Filter treated patents based on acquisition type and top tech status
    if acq_type == 'M&A' or acq_type == 'Off deal':
        treated_sample = treated[treated['acq_type'] == acq_type].copy()
    else:
        raise ValueError("Invalid acquisition type. Choose 'M&A' or 'Off deal'.")

    # Further filter based on top tech status, keep if total pretreatment citations are in the top 10% of treated patents in the last 4 quarters
    if top_tech:
        treated_sample['pre_treatment_total'] = treated_sample['patent_id'].apply(
            lambda pid: treated_counts_dict[pid]['vector'].sum() if pid in treated_counts_dict else 0
        )
        threshold = np.percentile(treated_sample['pre_treatment_total'], top_tech_threshold)
        treated_sample = treated_sample[treated_sample['pre_treatment_total'] >= threshold]
    
    return treated_sample


# ------------------------------
# 2. Citation preprocessing
# ------------------------------

def preprocess_citations(citations, treated, control, clean_ids):
    """Preprocess citation data and remove unwanted patents."""
    citations = cudf.DataFrame.from_pandas(citations)
    citations['patent_id'] = citations['patent_id'].astype(str)
    citations['citation_patent_id'] = citations['citation_patent_id'].astype(str)

    # Rename columns for clarity
    citations = citations.rename(columns={'patent_id': 'citedby_patent_id', 'citation_patent_id':'patent_id', 'filing_date':'citation_date'}, inplace = True)
   
    # Convert citation_date to datetime
    # and extract year, month, and quarter
    citations['citation_date'] = cudf.to_datetime(citations['citation_date'])
    citations['year'] = citations['citation_date'].dt.year
    citations['month'] = citations['citation_date'].dt.month
    citations['qtr'] = ((citations['month'] - 1) // 3 + 1).astype(str)
    citations['citation_quarter'] = citations['year'].astype(str) + 'Q' + citations['qtr']

    # Filter out patents that are not in the clean set
    # and ensure treated and control patents are in the clean set
    clean_set = set(clean_ids['patent_id'].astype(str))
    treated['patent_id'] = treated['patent_id'].astype(str)
    control['patent_id'] = control['patent_id'].astype(str)
    control = control[control['patent_id'].isin(clean_set)]
    assert len(control) == len(clean_set), "Control patents do not match clean ids"

    # Remove patents that are not in the clean set
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

def compute_treated_vectors(treated, citation_counts_dict,  baseline_begin_period = 13):


    """Compute pre-treatment citation vectors for treated patents."""

    # Calculate baseline end period
    baseline_end_period = int((baseline_begin_period - 1) / 2 + 2)

    # Ensure treated DataFrame has the necessary columns
    treated['year'] = treated['acq_date'].dt.year
    treated['month'] = treated['acq_date'].dt.month
    treated['qtr'] = ((treated['month'] - 1) // 3 + 1).astype(str)
    treated['acq_quarter'] = treated['year'].astype(str) + 'Q' + treated['qtr']

    treated_counts_dict = {}
    for i, row in treated.iterrows():
        treated_id = row['patent_id']
        acq_period = pd.Period(row['acq_date'], freq='Q')
        pre_quarters = [str(acq_period - j) for j in range(baseline_begin_period, baseline_end_period - 1, -1)]
        vec = [citation_counts_dict.get(treated_id, {}).get(q, 0) for q in pre_quarters]
        treated_counts_dict[treated_id] = {'pre_quarters': pre_quarters, 'vector': np.array(vec, dtype=float)}
    return treated_counts_dict


# ------------------------------
# 4. Distance calculation functions
# ------------------------------

def compute_cosine_distances(treated, control):
    """Compute cosine distances between treated and relevant control embeddings."""
    cosine_distance_by_treated = {}
    group_cols = ['cpc_subclass']
    treated_groups = treated.groupby(group_cols)

    for group_key, group in tqdm(treated_groups, total=len(treated_groups), desc="Precompute Cosine Distances"):
        # Extract group key values
        # and filter control patents based on cpc_subclass only - exact matching
        cpc_subclass_val = group_key
        candidates = control[(control['cpc_subclass'] == cpc_subclass_val)]
        
        if candidates.empty:
            print(f"No candidates found for group {group_key}. Skipping this group.")
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

    # Save
    with open("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/cosine_distance_by_treated.pkl", "wb") as f:
        pickle.dump(cosine_distance_by_treated, f)

    return cosine_distance_by_treated


def compute_hybrid_distance(d_mah, d_cos, lam):
    """Compute the hybrid distance matrix using Min-Max scaled Mahalanobis and normalized cosine distances."""

    # Min-Max scale Mahalanobis distances to [0, 1]
    d_mah_min = np.min(d_mah, axis=1, keepdims=True)
    d_mah_max = np.max(d_mah, axis=1, keepdims=True)
    d_mah_scaled = (d_mah - d_mah_min) / (d_mah_max - d_mah_min + 1e-16)  # Add epsilon to avoid divide-by-zero

    # Min-Max scale cosine distances to [0, 1]
    d_cos_min = np.min(d_cos, axis=1, keepdims=True)
    d_cos_max = np.max(d_cos, axis=1, keepdims=True)
    d_cos_scaled = (d_cos - d_cos_min) / (d_cos_max - d_cos_min + 1e-16)  # Add epsilon to avoid divide-by-zero

    # --- Explicit checks ---
    if not (0 <= d_mah_scaled.min() <= d_mah_scaled.max() <= 1):
        raise ValueError("Scaled Mahalanobis distances are not within [0, 1].")
    if not (0 <= d_cos_scaled.min() <= d_cos_scaled.max() <= 1):
        raise ValueError("Scaled cosine distances are not within [0, 1].")
    # -----------------------

    # Ensure that the shapes of d_mah and d_cos are compatible
    if d_mah_scaled.shape != d_cos_scaled.shape:
        raise ValueError("Scaled Mahalanobis and cosine distance matrices must have the same shape.")
    
    # Compute hybrid distance as convex combination
    d_h = lam * d_mah_scaled + (1 - lam) * d_cos_scaled
    return d_h, d_mah_scaled, d_cos_scaled

def precompute_mahalanobis(treated_df, control_df, citation_counts_dict, treated_counts_dict, baseline_begin_period=13):

    # Calculate baseline end period
    baseline_end_period = int((baseline_begin_period - 1) / 2 + 2)

    # Ensure treated_df has the necessary columns
    grouped = treated_df.groupby(['acq_quarter', 'cpc_subclass'])
    precomputed = []
    control_group_dict = {key: group for key, group in control_df.groupby(['cpc_subclass'])}

    for group_key, group in tqdm(grouped, total=len(grouped), desc="Precomputing Mahalanobis"):

        # --- Group prep ---
        acq_quarter, cpc_subclass = group_key
        acq_period = pd.Period(acq_quarter, freq='Q')
        pre_quarters = [str(acq_period - i) for i in range(baseline_begin_period, baseline_end_period - 1, -1)]


        # --- Get controls and candidate vectors ---
        candidates = control_group_dict.get((cpc_subclass), pd.DataFrame())
        if candidates.empty:
            continue

        candidate_ids = candidates['patent_id'].tolist()

        candidate_vectors = np.array([
            [citation_counts_dict.get(cid, {}).get(q, 0) for q in pre_quarters]
            for cid in candidate_ids
        ], dtype=np.float64)

        candidate_matrix = cp.asarray(candidate_vectors)
        cov_matrix = cp.cov(candidate_matrix, rowvar=False)
        inv_cov = cp.linalg.pinv(cov_matrix)


        treated_vectors = np.array([
            treated_counts_dict.get(row['patent_id'], {'vector': np.zeros(len(pre_quarters))})['vector']
            for _, row in group.iterrows()
        ], dtype=np.float64)

        if treated_vectors.size == 0:
            continue

        T = cp.asarray(treated_vectors)
        diff = candidate_matrix[None, :, :] - T[:, None, :]
        d_c_sq = cp.sum((diff @ inv_cov) * diff, axis=2)
        d_c = cp.sqrt(d_c_sq)
        cp.cuda.Stream.null.synchronize()

        # --- Append results ---
        precomputed.append({
            'treated_ids': group['patent_id'].tolist(),
            'candidate_ids': candidate_ids,
            'treated_vectors': treated_vectors,
            'candidate_vectors': candidate_vectors,
            'd_c_np': cp.asnumpy(d_c),
            'pre_quarters': pre_quarters,
            'group_key': group_key
        })

    return precomputed

def _save_mahalanobis(treated_sample, control, citation_counts_dict, treated_counts_dict,
                      baseline_begin_period, acq_type, threshold=None, top_tech=True):
    import dill as pickle
    import os

    suffix = f"{baseline_begin_period}q"
    threshold_str = f"_top_tech_{threshold}" if top_tech and threshold is not None else "_bl"
    filename = f"precomputed_mahalanobis_{acq_type}{threshold_str}_{suffix}.pkl"
    path = f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/{filename}"

    if os.path.exists(path):
        print(f"✔️ Mahalanobis already saved: {filename}")
        return

    print(f"⚙️ Computing Mahalanobis: {filename}")

    # Filter treated sample for valid vectors (if needed)
    treated_sample = treated_sample.copy()
    treated_sample['acq_period'] = treated_sample['acq_date'].apply(lambda d: pd.Period(d, freq='Q'))
    treated_sample['grant_period'] = treated_sample['grant_date'].apply(lambda d: pd.Period(d, freq='Q'))
    treated_sample['quarters_between'] = treated_sample.apply(lambda row: (row['acq_period'] - row['grant_period']).n, axis=1)
    filtered_treated = treated_sample[treated_sample['quarters_between'] >= baseline_begin_period]

    # Precompute
    precomputed = precompute_mahalanobis(filtered_treated, control, citation_counts_dict, treated_counts_dict, baseline_begin_period)

    # Save
    with open(path, 'wb') as f:
        pickle.dump(precomputed, f)

    print(f"✅ Saved: {filename}")


# ------------------------------
# 5. Matching
# ------------------------------
def hybrid_matching_for_lambda(lam, precomputed_mahalanobis, cosine_distance_by_treated, caliper=0.05, K = 1):
    matches = []
    dropped_patents_count = 0

    for group in precomputed_mahalanobis:
        treated_ids = group['treated_ids']
        candidate_ids = group['candidate_ids']
        d_c_np = group['d_c_np']
        pre_quarters = group['pre_quarters']

        cosine_matrix = np.stack(
            [cosine_distance_by_treated[tid] for tid in treated_ids if tid in cosine_distance_by_treated],
            axis=0
        )

        d_h, d_mah_scaled, d_cos_scaled = compute_hybrid_distance(d_c_np, cosine_matrix, lam)

        for i, tid in enumerate(treated_ids):
            distances = d_h[i]
            sorted_indices = np.argsort(distances)
            
            count = 0
            for idx in sorted_indices:
                if distances[idx] <= caliper and count < K:
                    matches.append({
                        'treated_id': tid,
                        'control_id': candidate_ids[idx],
                        'treated_vector': group['treated_vectors'][i],
                        'control_vector': group['candidate_vectors'][idx],
                        'mahalanobis_distance': float(d_c_np[i, idx]),
                        'mahalanobis_distance_scaled': float(d_mah_scaled[i, idx]),
                        'cosine_distance': float(cosine_matrix[i, idx]),
                        'cosine_distance_scaled': float(d_cos_scaled[i, idx]),
                        'hybrid_distance': float(distances[idx]),
                        'pre_quarters': pre_quarters
                    })
                    count += 1
                elif count >= K:
                    break

            if count == 0:
                dropped_patents_count += 1

    return pd.DataFrame(matches), dropped_patents_count


# -------------------------------
# 6. Placebo Effect Estimation Function
# -------------------------------
def estimate_placebo_effect(matched_df, citation_counts_dict, treated, placebo_periods=[7, 6, 5, 4, 3, 2]):
    """
    Estimate placebo effects for each matched treated-control pair by computing the difference in citations
    for each placebo period (t-5 to t-2). Assumes no treatment effect before acquisition, so true effect should be 0.

    Args:
        matched_df: DataFrame with matched treated and control patent pairs.
        citation_counts_dict: Dictionary with patent_id -> {quarter -> citation count}.
        treated: DataFrame with 'patent_id' and 'acq_date' columns.
        placebo_periods: List of integers indicating quarters before acquisition (e.g., [5, 4, 3, 2]).

    Returns:
        placebo_matrix: 2D np.array of shape (n_pairs, n_placebo_periods) with difference (treated - control)
                        in citations per placebo period.
    """
    treated_dates = treated.set_index('patent_id')['acq_date'].to_dict()
    placebo_matrix = []

    for _, row in matched_df.iterrows():
        tid = row['treated_id']
        cid = row['control_id']
        acq_date = treated_dates.get(tid)

        if acq_date is None:
            placebo_matrix.append([np.nan] * len(placebo_periods))
            continue

        acq_period = pd.Period(acq_date, freq='Q')

        diffs = []
        for p in placebo_periods:
            q = str(acq_period - p)
            t_cit = citation_counts_dict.get(tid, {}).get(q, 0)
            c_cit = citation_counts_dict.get(cid, {}).get(q, 0)
            diffs.append(t_cit - c_cit)

        placebo_matrix.append(diffs)

    return np.array(placebo_matrix)

# -------------------------------
# 7. Grid Search Over Lambda and Placebo Estimation
# -------------------------------

def prepare(
    placebo_periods=[4, 6, 8],
    acq_types=["M&A", "Off deal"],
    top_tech_flags=[False, True],
    top_tech_thresholds=[80, 90]
    ):
    import dill as pickle

    # Load and preprocess once
    citations, treated, control, clean_ids = load_data()
    print(f"Loaded data: {len(citations)} citations, {len(treated)} treated patents, {len(control)} control patents, {len(clean_ids)} clean ids.")
    # Preprocess citations and ensure treated and control are in the clean set
    citations, treated, control = preprocess_citations(citations, treated, control, clean_ids)
    print(f"Preprocessed data: {len(citations)} citations, {len(treated)} treated patents, {len(control)} control patents.")

    # Save static components
    quarterly_counts_pd = compute_quarterly_citation_counts(citations)
    citation_counts_dict = build_citation_counts_dict(quarterly_counts_pd)
    compute_cosine_distances(treated, control)

    if not os.path.exists("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/citation_counts_dict.pkl"):
        with open("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/citation_counts_dict.pkl", "wb") as f:
            pickle.dump(citation_counts_dict, f)
        print("Saved citation_counts_dict.")

    if not os.path.exists("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/control.pkl"):
        control_path = "/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/control.pkl"
        control.to_pickle(control_path)
        print(f"Saved control DataFrame.")

    for placebo_p in placebo_periods:
        baseline_begin_period = placebo_p * 2 + 1
        suffix = f"{baseline_begin_period}q"

        # Compute treated_vectors
        treated_counts_dict = compute_treated_vectors(treated, citation_counts_dict, baseline_begin_period)
        with open(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/treated_counts_dict_{suffix}.pkl", "wb") as f:
            pickle.dump(treated_counts_dict, f)
        print(f"Saved treated_counts_dict for baseline_begin_period={baseline_begin_period} quarters.")

        for acq_type in acq_types:
            for top_tech_flag in top_tech_flags:
                if top_tech_flag:
                    for threshold in top_tech_thresholds:
                        treated_sample = prepare_sample(
                            treated,
                            acq_type,
                            treated_counts_dict,
                            top_tech=True,
                            top_tech_threshold=threshold,
                            baseline_begin_period=baseline_begin_period
                        )
                        # Precompute and save mahalanobis
                        _save_mahalanobis(
                            treated_sample,
                            control,
                            citation_counts_dict,
                            treated_counts_dict,
                            baseline_begin_period,
                            acq_type,
                            threshold,
                            top_tech=True
                        )
                else:
                    treated_sample = prepare_sample(
                        treated,
                        acq_type,
                        treated_counts_dict,
                        top_tech=False,
                        baseline_begin_period=baseline_begin_period
                    )
                    _save_mahalanobis(
                        treated_sample,
                        control,
                        citation_counts_dict,
                        treated_counts_dict,
                        baseline_begin_period,
                        acq_type,
                        threshold=None,
                        top_tech=False
                    )
                # report progress
                print(f"Prepared and saved treated sample for acq_type={acq_type}, top_tech={top_tech_flag}, "
                      f"baseline_begin_period={baseline_begin_period} quarters, threshold={threshold if top_tech_flag else None}.")


def prepare_sample(treated, acq_type, treated_counts_dict,  top_tech = False, top_tech_threshold=90, baseline_begin_period = 13):

    """Prepare treated sample based on acquisition type and top tech status."""
    import os 

    # If the file exists, just load it
    treated_path = f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/treated_{acq_type}_top_tech_{top_tech_threshold}_{baseline_begin_period}q.pkl" if top_tech else f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/treated_{acq_type}_bl.pkl"
    if os.path.exists(treated_path): return pd.read_pickle(treated_path)

    # Adjust treated sample based on acquisition type and top tech status
    treated_sample = adjust_treated_sample(treated, acq_type, treated_counts_dict, top_tech, top_tech_threshold)

    # Save treated and control DataFrames
    if top_tech:
        treated_sample.to_pickle(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/treated_{acq_type}_top_tech_{top_tech_threshold}_{baseline_begin_period}q.pkl")
    else:
        treated_sample.to_pickle(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/treated_{acq_type}_bl.pkl")

    return treated_sample

def load_aux_data(acq_type, top_tech=False, top_tech_threshold=90, baseline_begin_period=13):

    # Suffix
    suffix = f"{baseline_begin_period}q"

    # Check if already the variable exists and skip loading if True
    if 'citation_counts_dict' not in locals():
        with open(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/citation_counts_dict.pkl", "rb") as f:
            citation_counts_dict = pickle.load(f)

    if 'cosine_distance_by_treated' not in locals():
        with open(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/cosine_distance_by_treated.pkl", "rb") as f:
            cosine_distance_by_treated = pickle.load(f)
    
    if 'control' not in locals():
        control = pd.read_pickle("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/control.pkl")

    # Load treated counts dictionary anyways
    with open(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/treated_counts_dict_{suffix}.pkl", "rb") as f:
        treated_counts_dict = pickle.load(f)

    # Load treated sample
    if top_tech:
        treated = pd.read_pickle(
            f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/treated_{acq_type}_top_tech_{top_tech_threshold}_{baseline_begin_period}q.pkl"
        )
    else:
        treated = pd.read_pickle(
            f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/treated_{acq_type}_bl.pkl"
        )

    return treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated

def load_precomputed_mahalanobis(acq_type, top_tech, baseline_begin_period, top_tech_threshold=None):
    suffix = f"{baseline_begin_period}q"
    threshold_str = f"_top_tech_{top_tech_threshold}" if top_tech and top_tech_threshold is not None else "_bl"
    filename = f"precomputed_mahalanobis_{acq_type}{threshold_str}_{suffix}.pkl"
    path = f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/_aux/{filename}"

    import dill as pickle
    with open(path, "rb") as f:
        return pickle.load(f)

import os
import pandas as pd
import csv

def log_grid_result(
    result_df,
    acq_type,
    top_tech,
    top_tech_threshold,
    baseline_begin_period,
    caliper,
    log_path="/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/grid_results_log.csv", 
    K = 1
):
    # Add identifying info
    result_df['acq_type'] = acq_type
    result_df['top_tech'] = top_tech
    result_df['top_tech_threshold'] = top_tech_threshold if top_tech else None
    result_df['baseline_begin_period'] = baseline_begin_period
    result_df['caliper'] = caliper
    result_df['number_of_matches'] = K

    # Reorder
    cols = [
        'acq_type', 'top_tech', 'top_tech_threshold',
        'baseline_begin_period', 'caliper', 'number_of_matches',
        'lambda', 'mse_diff', 'total_num_patents', 'num_dropped'
    ]
    result_df = result_df[cols]

    # Append
    file_exists = os.path.isfile(log_path)
    result_df.to_csv(log_path, mode='a', header=not file_exists, index=False)



def grid_runner(
    ## Not used
    placebo_periods=[4, 6, 8],
    calipers=[0.025, 0.05, 0.075, 0.1],
    acq_types=["M&A", "Off deal"],
    top_tech_flags=[False, True],
    top_tech_thresholds=[80, 90],
    K = 1
):
    for placebo_p in placebo_periods:
        baseline_begin_period = placebo_p * 2 + 1
        print(f"\nRunning grid for baseline_begin_period = {baseline_begin_period} quarters")
        for acq_type in acq_types:
            for top_tech_flag in top_tech_flags:
                thresholds = top_tech_thresholds if top_tech_flag else [None]

                for threshold in thresholds:
                    print(f"\nRunning grid for acq_type={acq_type}, top_tech={top_tech_flag}, "
                          f"threshold={threshold}, baseline={baseline_begin_period}q")

                    # Load all data
                    treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated = load_aux_data(
                        acq_type=acq_type,
                        top_tech=top_tech_flag,
                        top_tech_threshold=threshold if top_tech_flag else 90,
                        baseline_begin_period=baseline_begin_period
                    )

                    precomputed_mahalanobis = load_precomputed_mahalanobis(
                        acq_type=acq_type,
                        top_tech=top_tech_flag,
                        baseline_begin_period=baseline_begin_period,
                        top_tech_threshold=threshold
                    )

                    for caliper in calipers:
                        print(f"Matching for caliper = {caliper:.3f}")
                        results_df, matched_df_dict = run_routine(
                            treated,
                            control,
                            citation_counts_dict,
                            treated_counts_dict,
                            cosine_distance_by_treated,
                            caliper=caliper,
                            lambda_start=0,
                            lambda_end=1,
                            delta=0.25,
                            baseline_begin_period=baseline_begin_period,
                            placebo_periods=[i for i in range(baseline_begin_period - 1, 1, -1)],
                            precomputed_mahalanobis=precomputed_mahalanobis,
                            K = K  # Number of matches per treated patent
                        )

                        save_results(
                            results_df,
                            matched_df_dict,
                            acq_type=acq_type,
                            caliper=caliper,
                            top_tech=top_tech_flag,
                            top_tech_threshold=threshold or 90,                            
                            baseline_begin_period=baseline_begin_period
                        )

                        log_grid_result(
                            results_df,
                            acq_type=acq_type,
                            top_tech=top_tech_flag,
                            top_tech_threshold=threshold if top_tech_flag else None,
                            baseline_begin_period=baseline_begin_period,
                            caliper=caliper
                        )

                        # Visualize
                        visualize_mse(results_df)


def run_grid_point_K(args):
    (placebo_p, acq_type, top_tech_flag, threshold, caliper, K) = args
    baseline_begin_period = placebo_p * 2 + 1

    print(f"[K={K} | {acq_type}, top_tech={top_tech_flag}, threshold={threshold}, caliper={caliper}, baseline={baseline_begin_period}q]")

    treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated = load_aux_data(
        acq_type=acq_type,
        top_tech=top_tech_flag,
        top_tech_threshold=threshold if top_tech_flag else 90,
        baseline_begin_period=baseline_begin_period
    )

    precomputed_mahalanobis = load_precomputed_mahalanobis(
        acq_type=acq_type,
        top_tech=top_tech_flag,
        baseline_begin_period=baseline_begin_period,
        top_tech_threshold=threshold
    )

    results_df, matched_df_dict = run_routine(
        treated,
        control,
        citation_counts_dict,
        treated_counts_dict,
        cosine_distance_by_treated,
        caliper=caliper,
        delta=0.25,
        baseline_begin_period=baseline_begin_period,
        precomputed_mahalanobis=precomputed_mahalanobis,
        K=K
    )

    save_results(
        results_df,
        matched_df_dict,
        acq_type=acq_type,
        caliper=caliper,
        top_tech=top_tech_flag,
        top_tech_threshold=threshold if top_tech_flag else 90,
        baseline_begin_period=baseline_begin_period,
        K=K
    )

    log_grid_result(
        results_df,
        acq_type=acq_type,
        top_tech=top_tech_flag,
        top_tech_threshold=threshold if top_tech_flag else None,
        baseline_begin_period=baseline_begin_period,
        caliper=caliper,
        K=K
    )

    return f"✅ Done: K={K}, {acq_type}, top_tech={top_tech_flag}, threshold={threshold}, caliper={caliper}, baseline={baseline_begin_period}q"



def grid_runner_parallel_K(
    placebo_periods=[4, 6, 8],
    calipers=[0.05, 0.1],
    acq_types=["M&A", "Off deal"],
    top_tech_flags=[False, True],
    top_tech_thresholds=[80],
    Ks=[10],
    max_workers=4
):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = []
    import os

    prefix = "/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/"

    for placebo_p in placebo_periods:
        baseline_begin_period = placebo_p * 2 + 1
        for acq_type in acq_types:
            for top_tech_flag in top_tech_flags:
                thresholds = top_tech_thresholds if top_tech_flag else [None]
                for threshold in thresholds:
                    for caliper in calipers:
                        for K in Ks:
                            # Generate the file suffix
                            if top_tech_flag:
                                suffix = f"{acq_type}, top-tech, {threshold}, {baseline_begin_period}q, caliper_{caliper:.4f}, {K}matches"
                            else:
                                suffix = f"{acq_type}, baseline, {baseline_begin_period}q, caliper_{caliper:.4f}, {K}matches"

                            # Define the 4 expected files
                            files_to_check = [
                                f"{prefix}01 Hybrid matching results - {suffix}.pkl",
                                f"{prefix}01 Hybrid matching results - {suffix}.csv",
                                f"{prefix}01 Hybrid matches - {suffix}.pkl",
                                f"{prefix}01 Hybrid matches - {suffix}.csv"
                            ]

                            if all(os.path.exists(f) for f in files_to_check):
                                print(f"⏭️ Skipping: {suffix} (results already exist)")
                                continue

                            tasks.append((placebo_p, acq_type, top_tech_flag, threshold, caliper, K))


    print(f"\n🔁 Launching {len(tasks)} grid runs in parallel with {max_workers} workers...\n")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_grid_point_K, t) for t in tasks]
        for f in as_completed(futures):
            try:
                result = f.result()
                print(result)
                results.append(result)
            except Exception as e:
                print(f"❌ Error in a task: {e}")

    print("\n🎉 All grid points completed.")
    return results



def run_routine(treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated,
                caliper=0.05, lambda_start=0, lambda_end=1, delta=0.25, baseline_begin_period=13, precomputed_mahalanobis=None, K = 1):
    """
    Run hybrid matching over a grid of lambda values, compute placebo effects for t-5 to t-2,
    and return MSE results. Matching is done on CPC, and then based on hybrid distance
    using Mahalanobis (citations) and cosine (embeddings). Placebo effects are computed for each of
    four pre-treatment quarters, and overall MSE is used to select optimal lambda.

    Returns:
        results_df: DataFrame with lambda, MSE (true).
        matched_df_dict: Dictionary mapping lambda to the matched DataFrame.
    """
    lambda_values = np.arange(lambda_start, lambda_end + delta, delta)
    mse_diff_list = []
    patents_counts = []
    dropped_counts = []
    matched_df_dict = {}

    # Calculate baseline end period - we force equal number of estimation and placebo periods
    baseline_end_period = int((baseline_begin_period - 1)/2 + 2)

    # Prepare treated with periods
    treated = treated.copy()
    treated['acq_period'] = treated['acq_date'].apply(lambda d: pd.Period(d, freq='Q'))
    treated['grant_period'] = treated['grant_date'].apply(lambda d: pd.Period(d, freq='Q'))
    treated['quarters_between'] = treated.apply(lambda row: (row['acq_period'] - row['grant_period']).n, axis=1)
    filtered_treated = treated[treated['quarters_between'] >= baseline_begin_period]

    if precomputed_mahalanobis is None:

        precomputed_mahalanobis = precompute_mahalanobis(filtered_treated, control, citation_counts_dict, treated_counts_dict, baseline_begin_period)

    for lam in tqdm(lambda_values, total=len(lambda_values), desc="Grid Search over Lambda"):
        #print(f"Running hybrid matching for lambda = {lam:.2f}")
        matched_df, dropped_count  = hybrid_matching_for_lambda(lam, precomputed_mahalanobis, cosine_distance_by_treated, caliper, K)
        matched_df_dict[lam] = matched_df.copy()

        # Estimate placebo effects for each period (returns shape: [n_pairs, 4])
        placebo_periods = [i for i in range(baseline_end_period - 1, 1, -1)]
        placebo_matrix = estimate_placebo_effect(matched_df, citation_counts_dict, treated, placebo_periods)

        # Remove any rows with NaN
        valid_rows = ~np.isnan(placebo_matrix).any(axis=1)
        placebo_matrix_clean = placebo_matrix[valid_rows]

        # Compute overall MSE (true MSE): average of all squared differences
        mse_diff = np.mean(placebo_matrix_clean ** 2)
        mse_diff_list.append(mse_diff)
        patents_counts.append(len(filtered_treated))
        dropped_counts.append(dropped_count)

        #print(f"Lambda {lam:.2f}: MSE = {mse_diff:.3f}")

    results_df = pd.DataFrame({
        'lambda': lambda_values,
        'mse_diff': mse_diff_list,
        'total_num_patents': patents_counts,
        'num_dropped': dropped_counts
    })

    print("Results of grid search:")
    print(results_df)

    # Visualize 
    visualize_mse(results_df)

    return results_df, matched_df_dict


# -------------------------------
# 8. Visualize MSE's 
# -------------------------------

def visualize_mse(results_df):
    """Visualize MSE results"""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot MSE (Diff) on the left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Lambda')
    ax1.set_ylabel('MSE', color=color1)
    ax1.plot(results_df['lambda'], results_df['mse_diff'], marker='o', color=color1, label='MSE')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Title and grid
    fig.suptitle('MSE across Lambda')
    fig.tight_layout()
    plt.grid(True)
    plt.show()


# -------------------------------
# 9. Save Results
# -------------------------------

def save_results(results_df, matched_df_dict, acq_type, caliper = 0.05, top_tech = False, top_tech_threshold = 90, baseline_begin_period = 13, K = 1):

    # Define suffix and prefix
    suffix = f"{acq_type}, top-tech, {top_tech_threshold}, {baseline_begin_period}q, caliper_{caliper:.4f}, {K}matches" if top_tech else f"{acq_type}, baseline, {baseline_begin_period}q, caliper_{caliper:.4f}, {K}matches"
    prefix = "/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation_no_exact_match_on_grantyear/"
    
    # Save matching results
    results_df.to_pickle(f"{prefix}01 Hybrid matching results - {suffix}.pkl")
    results_df.to_csv(f"{prefix}01 Hybrid matching results - {suffix}.csv", index=False)

    # Save matched dictionary where keys are lambdas and values are matched DFs
    with open(f"{prefix}01 Hybrid matches - {suffix}.pkl", "wb") as f:
            pickle.dump(matched_df_dict, f)

    # Convert into Pandas DF and save as CSV
    import pandas as pd

    combined_df = pd.concat(
        [df.assign(lambda_val=lam) for lam, df in matched_df_dict.items()],
        ignore_index=True
    )
    combined_df.to_csv(f"{prefix}01 Hybrid matches - {suffix}.csv", index=False)


# -------------------------------
# 10. Main Routine
# -------------------------------

def main():
    # Prepare data
    treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated = prepare()

    # Run the matching routine and get results
    results_df, matched_df_dict = run_routine(treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated)

    return results_df, matched_df_dict