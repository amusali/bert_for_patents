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

def compute_treated_vectors(treated, citation_counts_dict,  baseline_begin_period = 9, baseline_end_period = 6):
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

    return cosine_distance_by_treated


def compute_hybrid_distance_alpha(d_c, d_e, lam, alpha):
    """Compute the hybrid distance matrix using Mahalanobis and cosine distances."""
    d_c_scaled = 1 - np.exp(-alpha * d_c)
    d_e_scaled = d_e / 2
    d_h = lam * d_c_scaled + (1 - lam) * d_e_scaled
    return d_h

def compute_hybrid_distance(d_mah, d_cos, lam):
    """Compute the hybrid distance matrix using Min-Max scaled Mahalanobis and normalized cosine distances."""

    # Min-Max scale Mahalanobis distances to [0, 1]
    d_mah_min = np.min(d_mah, axis=1, keepdims=True)
    d_mah_max = np.max(d_mah, axis=1, keepdims=True)
    d_mah_scaled = (d_mah - d_mah_min) / (d_mah_max - d_mah_min + 1e-16)  # Add epsilon to avoid divide-by-zero

    # Cosine distances should already be in [0, 2]; normalize to [0, 1]
    d_cos_scaled = d_cos / 2

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
    return d_h



# ------------------------------
# 5. Matching
# ------------------------------

def hybrid_matching_for_lambda(lam, treated_df, control_df, treated_counts_dict, citation_counts_dict, cosine_distance_by_treated):
    """Perform hybrid matching between treated and control patents for a given lambda."""

    # Group control patents by (grant_year, cpc_subclass)
    control_group_dict = {key: group for key, group in control_df.groupby(['grant_year', 'cpc_subclass'])}
    matches = []

    # Group treated patents by (acq_quarter, grant_year, cpc_subclass)
    treated_groups = list(treated_df.groupby(['acq_quarter', 'grant_year', 'cpc_subclass']))

    # Iterate over each treated group
    for group_key, group in tqdm(treated_groups, total=len(treated_groups), desc="Hybrid Matching Groups"):
        acq_quarter, grant_year, cpc_subclass = group_key
        acq_period = pd.Period(acq_quarter, freq='Q')

        # Select pre-treatment quarters (5th to 8th quarters before acquisition)
        pre_quarters = [str(acq_period - i) for i in range(8, 4, -1)]

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
            treated_info = treated_counts_dict.get(tid, {'pre_quarters': pre_quarters, 'vector': np.zeros(4)})
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
        d_h = compute_hybrid_distance(d_c_np, np.stack(cosine_list, axis=0), lam)

        # Identify best match (lowest hybrid distance) for each treated patent
        best_indices = np.argmin(d_h, axis=1)

        # Store match information
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

    # Return all matches as a DataFrame
    return pd.DataFrame(matches)


# -------------------------------
# 6. Placebo Effect Estimation Function
# -------------------------------
def estimate_placebo_effect(matched_df, citation_counts_dict, treated, baseline_end_period = 6, outcome_begin_period = 5, outcome_end_period = 2):
    """
    For each matched pair in matched_df, define the placebo effect as a difference-in-differences
    using a modified window:
      - Baseline: cumulative citations in the four quarters from Q_actual - 7 to Q_actual - 4,
      - Outcomes: citation count in Q_actual - 3 and Q_actual - 2.

    For each matched pair i, define: NEEDS TO BE MODIFIED
      e_{i,t-3} = [Y^T_i(Q_{actual}-3) - sum_{q=Q_actual-7}^{Q_actual-4} Y^T_i(q)]
                  - [Y^C_i(Q_{actual}-3) - sum_{q=Q_actual-7}^{Q_actual-4} Y^C_i(q)]
      e_{i,t-2} = [Y^T_i(Q_{actual}-2) - sum_{q=Q_actual-7}^{Q_actual-4} Y^T_i(q)]
                  - [Y^C_i(Q_{actual}-2) - sum_{q=Q_actual-7}^{Q_actual-4} Y^C_i(q)]

    Returns two arrays: one for e_{t-3} and one for e_{t-2} for all matched pairs.
    """
    # Assert that outcome period immediately follows baseline period
    assert outcome_begin_period == baseline_end_period - 1, "Outcome period must immediately follow baseline period"

    placebo_effects = []

    # Use the treated DataFrame (assumed to be a Pandas DataFrame with 'patent_id' and 'acq_date')
    treated_dates = treated.set_index('patent_id')['acq_date'].to_dict()
    for idx, row in matched_df.iterrows():
        tid = row['treated_id']
        cid = row['control_id']
        acq_date = treated_dates.get(tid)
        if acq_date is None:
            placebo_effects.append(np.nan)
            continue
        # Use the actual acquisition quarter.
        acq_period = pd.Period(acq_date, freq='Q')

        # Define the 4 outcome quarters: from Q_actual - 5 to Q_actual - 2.
        outcome_quarters = [str(acq_period - i) for i in range(outcome_begin_period, outcome_end_period - 1, -1)]

        treated_outcome = sum(citation_counts_dict.get(tid, {}).get(q, 0) for q in outcome_quarters)
        control_outcome = sum(citation_counts_dict.get(cid, {}).get(q, 0) for q in outcome_quarters)

        e = treated_outcome - control_outcome

        placebo_effects.append(e)


    return np.array(placebo_effects)


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

    return treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated

def run_routine(treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated, delta=0.2, baseline_begin_period = 9, baseline_end_period = 6, outcome_begin_period = 5, outcome_end_period = 2):
    """
    Run the matching routine for different lambda values and compute MSE.
    """
    # Define lambda values for grid search
    lambda_values = np.arange(0, 1 + delta, delta)   # 0, 0.2, ..., 1.0
    
    # Lists to store MSE's placebo outcomes.
    mse_diff_list = []  # Difference-based placebo effect MSE
    mse_reg_list = []   # Regression-based placebo effect MSE - real MSE

    # Create a dictionary to store matched DataFrames for each lambda.
    matched_df_dict = {}

    # Add acquisition and grant periods as Periods with quarterly frequency.
    treated['acq_period'] = treated['acq_date'].apply(lambda d: pd.Period(d, freq='Q'))
    treated['grant_period'] = treated['grant_date'].apply(lambda d: pd.Period(d, freq='Q'))

    # Compute number of quarters between grant and acquisition.
    treated['quarters_between'] = treated.apply(lambda row: (row['acq_period'] - row['grant_period']).n, axis=1)

    # Filter: only include treated patents with at least 9 quarters between grant and acq.
    filtered_treated = treated[treated['quarters_between'] >= baseline_begin_period]

    for lam in lambda_values:
        print(f"Running hybrid matching for lambda = {lam:.2f}")

        # Run matching
        matched_df = hybrid_matching_for_lambda(lam, filtered_treated, control, treated_counts_dict, citation_counts_dict, cosine_distance_by_treated)

        # Save this matched_df for later inspection.
        matched_df_dict[lam] = matched_df.copy()

        # Estimate placebo effects (using the difference-of-differences approach) for t+1 and t+2.
        e = estimate_placebo_effect(matched_df, citation_counts_dict, treated, baseline_end_period, outcome_begin_period, outcome_end_period)

        # Remove any NaN values.
        placebo_effects_clean = e[~np.isnan(e)]

        # Check that no NaN values
        assert len(placebo_effects_clean) == len(e), "NaN values found in placebo effects"

        # Difference-based MSE: average squared placebo effect.
        mse_diff = np.mean(placebo_effects_clean ** 2)

        # Regression-based MSE: average squared difference between placebo effect and mean placebo effect.
        mse_reg = (np.mean(placebo_effects_clean)) ** 2

        mse_diff_list.append(mse_diff)
        mse_reg_list.append(mse_reg)


        print(f"Lambda {lam:.2f}: Difference MSE = {mse_diff:.3f}, Regression MSE = {mse_reg:.3f}")

    # -------------------------------
    # Combine MSE Results into a DataFrame
    # -------------------------------
    results_df = pd.DataFrame({
        'lambda': lambda_values,
        'mse_diff': mse_diff_list,
        'mse_reg': mse_reg_list,

    })

    print("Results of grid search:")
    print(results_df)   

    return results_df, matched_df_dict

# -------------------------------
# 8. Visualize MSE's 
# -------------------------------

def visualize_mse(results_df):
    """Visualize MSE results using a dual-axis plot."""
    import matplotlib.pyplot as plt

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot MSE (Diff) and MSE (Reg)
    ax.plot(results_df['lambda'], results_df['mse_diff'], marker='o', label='MSE (Diff)', color='blue')
    ax.plot(results_df['lambda'], results_df['mse_reg'], marker='s', label='MSE (Reg)', color='red')

    # Set labels and title
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('MSE Comparison: Diff vs. Reg across Lambda')
    ax.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# 9. Save Results
# -------------------------------

def save_results(results_df, matched_df_dict, baseline_begin_period = 9):
    results_df.to_pickle(f"/content/drive/MyDrive/PhD Data/11 Matches/03 Hybrid matching results - {baseline_begin_period}q.pkl")

    import pickle
    with open(f"/content/drive/MyDrive/PhD Data/11 Matches/03 Hybrid matches - {baseline_begin_period}q.pkl", "wb") as f:
        pickle.dump(matched_df_dict, f)

    # Optionally, you can also save results_df as CSV:
    results_df.to_csv(f"/content/drive/MyDrive/PhD Data/11 Matches/03 Hybrid matching results - {baseline_begin_period}q.csv", index=False)

# -------------------------------
# 10. Main Routine
# -------------------------------

def main():
    # Prepare data
    treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated = prepare()

    # Run the matching routine and get results
    results_df, matched_df_dict = run_routine(treated, control, citation_counts_dict, treated_counts_dict, cosine_distance_by_treated)

    return results_df, matched_df_dict