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
              treated_path = "/content/drive/MyDrive/PhD Data/10 Sample - pre final/acquired_patents.pkl", 
              control_path = "/content/drive/MyDrive/PhD Data/10 Sample - pre final/potential_controls.pkl", 
              clean_ids_path = "/content/drive/MyDrive/PhD Data/10 Sample - pre final/clean_potential_control_ids.csv",
              acq_type_path = "/content/drive/MyDrive/PhD Data/09 Acquired patents/04 All patents.dta",
              ):
    
    """Load citations, treated, control data, clean ids, and acquistion type data."""

    # Always reload treated, as it's the only one that must be reloaded
    treated = pd.read_pickle(treated_path)
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

def compute_treated_vectors(treated, citation_counts_dict,  baseline_begin_period = 13, baseline_end_period = 8):
    """Compute pre-treatment citation vectors for treated patents."""
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
    group_cols = ['grant_year', 'cpc_subclass']
    treated_groups = treated.groupby(group_cols)

    for group_key, group in tqdm(treated_groups, total=len(treated_groups), desc="Precompute Cosine Distances"):
        # Extract group key values
        # and filter control patents based on grant_year and cpc_subclass - exact matching
        grant_year_val, cpc_subclass_val = group_key
        candidates = control[(control['grant_year'] == grant_year_val) & (control['cpc_subclass'] == cpc_subclass_val)]
        
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
    with open("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/cosine_distance_by_treated.pkl", "wb") as f:
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



# ------------------------------
# 5. Matching
# ------------------------------

def hybrid_matching_for_lambda(lam, treated_df, control_df, treated_counts_dict, citation_counts_dict, cosine_distance_by_treated, caliper = 0.05, baseline_begin_period = 13, baseline_end_period = 8):
    """Perform hybrid matching between treated and control patents for a given lambda."""

    # Group control patents by (grant_year, cpc_subclass)
    control_group_dict = {key: group for key, group in control_df.groupby(['grant_year', 'cpc_subclass'])}
    matches = []

    # Group treated patents by (acq_quarter, grant_year, cpc_subclass)
    treated_groups = list(treated_df.groupby(['acq_quarter', 'grant_year', 'cpc_subclass']))

    dropped_patents_count = 0

    # Iterate over each treated group
    for group_key, group in tqdm(treated_groups, total=len(treated_groups), desc="Hybrid Matching Groups"):
        acq_quarter, grant_year, cpc_subclass = group_key
        acq_period = pd.Period(acq_quarter, freq='Q')

        # Select pre-treatment quarters (5th to 8th quarters before acquisition)
        pre_quarters = [str(acq_period - i) for i in range(baseline_begin_period, baseline_end_period - 1, -1)]

        # Get candidate control patents with matching (grant_year, cpc_subclass)
        candidates = control_group_dict.get((grant_year, cpc_subclass), pd.DataFrame())
        if candidates.empty:
            continue

        # Build citation vectors for each candidate control patent
        candidate_ids = candidates['patent_id'].tolist()
        candidate_vectors = [
            [citation_counts_dict.get(cid, {}).get(q, 0) for q in pre_quarters]
            for cid in candidate_ids
        ]

        # Convert to CuPy array and compute inverse covariance matrix
        candidate_matrix = cp.array(candidate_vectors, dtype=cp.float64)
        if candidate_matrix.shape[0] > 1:
            cov_matrix = cp.cov(candidate_matrix, rowvar=False)
        else:
            cov_matrix = cp.eye(4, dtype=cp.float64)
        inv_cov = cp.linalg.pinv(cov_matrix)

        treated_ids = []
        treated_vectors = []
        cosine_list = []

        # Collect treated patent IDs, their citation vectors, and cosine distances
        for _, row in group.iterrows():
            tid = row['patent_id']
            d_e = cosine_distance_by_treated.get(tid)
            if d_e is None:
                continue
            treated_ids.append(tid)
            treated_info = treated_counts_dict.get(tid, {'pre_quarters': pre_quarters, 'vector': np.zeros(len(pre_quarters))})
            treated_vectors.append(treated_info['vector'])
            cosine_list.append(d_e)

        # Skip if no valid treated vectors
        if len(treated_vectors) == 0:
            continue

        # Convert treated vectors to CuPy and compute Mahalanobis distance matrix
        T = cp.array(treated_vectors, dtype=cp.float64)
        diff = candidate_matrix[None, :, :] - T[:, None, :]
        d_c_sq = cp.sum((diff @ inv_cov) * diff, axis=2)
        d_c = cp.sqrt(d_c_sq)
        cp.cuda.Stream.null.synchronize()
        d_c_np = cp.asnumpy(d_c)

        # Compute hybrid distance using min-max scaled Mahalanobis and cosine distances
        d_h, d_mah_scaled, d_cos_scaled = compute_hybrid_distance(d_c_np, np.stack(cosine_list, axis=0), lam)

        # Identify best match (lowest hybrid distance) for each treated patent
        best_indices = np.argmin(d_h, axis=1)

        # Store match information
        for i, tid in enumerate(treated_ids):
            best_idx = best_indices[i]
            best_dist = d_h[i, best_idx]

            
            # Check if the best distance is within the caliper and drop if not
            if best_dist > caliper:
                dropped_patents_count += 1
                continue

            matches.append({
                'treated_id': tid,
                'control_id': candidate_ids[best_idx],
                'treated_vector': treated_vectors[i],
                'control_vector': candidate_vectors[best_idx],
                'mahalanobis_distance': float(d_c_np[i, best_idx]),
                'mahalanobis_distance_scaled': float(d_mah_scaled[i, best_idx]),
                'cosine_distance': float(cosine_list[i][best_idx]),
                'cosine_distance_scaled': float(d_cos_scaled[i, best_idx]),
                'hybrid_distance': float(d_h[i, best_idx]),
                'pre_quarters': pre_quarters
            })
    if dropped_patents_count >0:
        # Print the number of patents dropped due to caliper restriction
        print(f"Dropped {dropped_patents_count} patents due to caliper restriction.")

    # Return all matches as a DataFrame
    return pd.DataFrame(matches)


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

def prepare():
    # Load data
    citations, treated, control, clean_ids = load_data()

    # Preprocess citation data
    citations, treated, control = preprocess_citations(citations, treated, control, clean_ids)

    # Compute quarterly citation counts
    quarterly_counts_pd = compute_quarterly_citation_counts(citations)
    citation_counts_dict = build_citation_counts_dict(quarterly_counts_pd)

    # Compute treated vectors
    treated_counts_dict = compute_treated_vectors(treated, citation_counts_dict)

    # Compute cosine distances
    cosine_distance_by_treated = compute_cosine_distances(treated, control)

    # Save all 
    with open("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/citation_counts_dict.pkl", "wb") as f:
        pickle.dump(citation_counts_dict, f)
    with open("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/treated_counts_dict.pkl", "wb") as f:
        pickle.dump(treated_counts_dict, f)
    with open(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/cosine_distance_by_treated.pkl", "wb") as f:
        pickle.dump(cosine_distance_by_treated, f)

    # Save control DataFrame
    control.to_pickle("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/control.pkl")

    return treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated

def prepare_sample(treated, acq_type, treated_counts_dict,  top_tech = False, top_tech_threshold=90):

    # Adjust treated sample based on acquisition type and top tech status
    treated_sample = adjust_treated_sample(treated, acq_type, treated_counts_dict, top_tech, top_tech_threshold)

    # Save treated and control DataFrames
    if top_tech:
        treated_sample.to_pickle(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/treated_{acq_type}_top_tech_{top_tech_threshold}.pkl")
    else:
        treated_sample.to_pickle(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/treated_{acq_type}_bl.pkl")

    return treated_sample

def load_aux_data(acq_type, top_tech = False, top_tech_threshold=90):

    """Load auxiliary data from pickle files."""

    ## Check if already loaded
    if 'citation_counts_dict' not in locals():
        with open("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/citation_counts_dict.pkl", "rb") as f:
            citation_counts_dict = pickle.load(f)
    if 'treated_counts_dict' not in locals():
        with open("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/treated_counts_dict.pkl", "rb") as f:
            treated_counts_dict = pickle.load(f)
    if 'cosine_distance_by_treated' not in locals():
        with open("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/cosine_distance_by_treated.pkl", "rb") as f:
            cosine_distance_by_treated = pickle.load(f)
    if 'control' not in locals():
        control = pd.read_pickle("/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/control.pkl")

    ## Load treated based on type and top tech status
    if top_tech:
        treated = pd.read_pickle(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/treated_{acq_type}_top_tech_{top_tech_threshold}.pkl")
    else:
        treated = pd.read_pickle(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/_aux/treated_{acq_type}_bl.pkl")
        
    return treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated

def run_routine(treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated, caliper = 0.05, lambda_start = 0, lambda_end = 1, delta=0.2, baseline_begin_period=13):
    """
    Run hybrid matching over a grid of lambda values, compute placebo effects for t-5 to t-2,
    and return MSE results. Matching is done on grant year and CPC, and then based on hybrid distance
    using Mahalanobis (citations) and cosine (embeddings). Placebo effects are computed for each of
    four pre-treatment quarters, and overall MSE is used to select optimal lambda.

    Returns:
        results_df: DataFrame with lambda, MSE (true).
        matched_df_dict: Dictionary mapping lambda to the matched DataFrame.
    """
    lambda_values = np.arange(lambda_start, lambda_end + delta, delta)
    mse_diff_list = []
    matched_df_dict = {}

    # Prepare treated with periods
    treated = treated.copy()
    treated['acq_period'] = treated['acq_date'].apply(lambda d: pd.Period(d, freq='Q'))
    treated['grant_period'] = treated['grant_date'].apply(lambda d: pd.Period(d, freq='Q'))
    treated['quarters_between'] = treated.apply(lambda row: (row['acq_period'] - row['grant_period']).n, axis=1)
    filtered_treated = treated[treated['quarters_between'] >= baseline_begin_period]

    for lam in lambda_values:
        print(f"Running hybrid matching for lambda = {lam:.2f}")
        matched_df = hybrid_matching_for_lambda(lam, filtered_treated, control, treated_counts_dict, citation_counts_dict, cosine_distance_by_treated, caliper)
        matched_df_dict[lam] = matched_df.copy()

        # Estimate placebo effects for each period (returns shape: [n_pairs, 4])
        placebo_matrix = estimate_placebo_effect(matched_df, citation_counts_dict, treated)

        # Remove any rows with NaN
        valid_rows = ~np.isnan(placebo_matrix).any(axis=1)
        placebo_matrix_clean = placebo_matrix[valid_rows]

        # Compute overall MSE (true MSE): average of all squared differences
        mse_diff = np.mean(placebo_matrix_clean ** 2)
        mse_diff_list.append(mse_diff)

        print(f"Lambda {lam:.2f}: MSE = {mse_diff:.3f}")

    results_df = pd.DataFrame({
        'lambda': lambda_values,
        'mse_diff': mse_diff_list,
    })

    print("Results of grid search:")
    print(results_df)

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
    ax1.set_ylabel('MSE (Diff)', color=color1)
    ax1.plot(results_df['lambda'], results_df['mse_diff'], marker='o', color=color1, label='MSE (Diff)')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Title and grid
    fig.suptitle('MSE across Lambda')
    fig.tight_layout()
    plt.grid(True)
    plt.show()


# -------------------------------
# 9. Save Results
# -------------------------------

def save_results(results_df, matched_df_dict, acq_type, top_tech = False, top_tech_threshold = 90, baseline_begin_period = 13):
    if top_tech:
        results_df.to_pickle(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/01 Hybrid matching results - {acq_type}, top-tech, {top_tech_threshold}, {baseline_begin_period}q.pkl")
        results_df.to_csv(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/01 Hybrid matching results - {acq_type}, top-tech, {top_tech_threshold}, {baseline_begin_period}q.csv", index=False)

        with open(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/01 Hybrid matches - {acq_type}, top-tech, {top_tech_threshold}, {baseline_begin_period}q.pkl", "wb") as f:
            pickle.dump(matched_df_dict, f)

    else:
        results_df.to_pickle(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/01 Hybrid matching results - {acq_type}, baseline, {baseline_begin_period}q.pkl")
        results_df.to_csv(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/01 Hybrid matching results - {acq_type}, baseline, {baseline_begin_period}q.csv", index=False)

        with open(f"/content/drive/MyDrive/PhD Data/11 Matches/optimization results/citation/01 Hybrid matches - {acq_type}, baseline, {baseline_begin_period}q.pkl", "wb") as f:
            pickle.dump(matched_df_dict, f)


# -------------------------------
# 10. Main Routine
# -------------------------------

def main():
    # Prepare data
    treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated = prepare()

    # Run the matching routine and get results
    results_df, matched_df_dict = run_routine(treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated)

    return results_df, matched_df_dict