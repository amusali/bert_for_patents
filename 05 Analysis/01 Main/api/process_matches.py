#!/usr/bin/env python3
"""
finalize_samples.py

Load matched samples, combine with citation and patent metadata,
add detailed timing for each step, and write final datasets for estimation.
Adjust the directory constants or pass via CLI to suit your environment.
"""
import os
import glob
import time
import dill as pickle
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

# ==== CONFIGURATION (adjust as needed) ====
# Input directory containing the old matched pickles
INPUT_DIR = "/content/drive/MyDrive/PhD Data/11 Matches/actual results/citation/"
# Output directory for final samples
OUTPUT_DIR = "/content/drive/MyDrive/PhD Data/12 Sample Final/actual results/citation/"
# Raw citations pickle (will be trimmed per sample)
CITATIONS_FILE = "/content/drive/MyDrive/PhD Data/08 Citations/03 Patent citations - raw, filing.pickle"
# All patents metadata (Stata .dta)
PATENTS_FILE = "/content/drive/MyDrive/PhD Data/09 Acquired patents/04 All patents.dta"
# Glob pattern to find matched samples
INPUT_PATTERN = "01 Hybrid matches - *10matches.pkl"
# Template for output filenames
OUTPUT_FILE_TEMPLATE = "Sample - {suffix}.{ext}"

# ---- Low-level I/O and parsing ----
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def parse_suffix(suffix: str) -> dict:
    parts = [p.strip() for p in suffix.split(',')]
    params = { 'acq_type': parts[0] }
    if parts[1].lower() == 'baseline':
        params.update({
            'top_tech': False,
            'top_tech_threshold': None,
            'baseline_period': int(parts[2].rstrip('q')),
            'caliper': float(parts[3].replace('caliper_', '')),
            'K': int(parts[4].replace('matches', ''))
        })
    else:
        params.update({
            'top_tech': True,
            'top_tech_threshold': int(parts[2]),
            'baseline_period': int(parts[3].rstrip('q')),
            'caliper': float(parts[4].replace('caliper_', '')),
            'K': int(parts[5].replace('matches', ''))
        })
    return params

# ---- Citation preprocessing ----
def get_unique_ids(sample: dict) -> np.ndarray:
    ids = []
    for lam, df in sample.items():
        df['treated_id'] = df['treated_id'].astype(str)
        df['control_id'] = df['control_id'].astype(str)
        ids.extend(df['treated_id'].unique())
        ids.extend(df['control_id'].unique())
    return np.unique(ids)

def trim_citations(citations: pd.DataFrame) -> pd.DataFrame:

    # Unique IDs
    patent_ids = retrieve_IDs()

    citations['patent_id'] = citations['patent_id'].astype(str)
    return citations[citations['patent_id'].isin(patent_ids)]


# Find all relevant IDs and trim citation file accordingly
def retrieve_IDs(input_dir: str = INPUT_DIR):
    """Retrieve unique IDs from all matched samples in the input directory."""
    all_ids = set()
    for filepath in tqdm(glob.glob(os.path.join(input_dir, INPUT_PATTERN)), desc = "Loading samples"):
        sample = load_pickle(filepath)
        ids = get_unique_ids(sample)
        all_ids.update(ids)
    return np.array(list(all_ids))

def collapse_citations() -> pd.DataFrame:
    """Collapse citation DataFrame to unique patent_id and citation_quarter."""
    # Load citations
    citations = load_pickle(CITATIONS_FILE)

    citations.rename(columns={'patent_id':'citedby_patent_id',
                         'citation_patent_id':'patent_id',
                         'filing_date':'citation_date'}, inplace=True)

    # Trim citations
    citations = trim_citations(citations)

    # Ensure 'patent_id' is string
    citations['patent_id'] = citations['patent_id'].astype(str)
    
    # Ensure 'citation_date' is datetime
    citations['citation_date'] = pd.to_datetime(citations['citation_date'])

    # Extract year, month, and quarter
    citations['year']  = citations['citation_date'].dt.year
    citations['month'] = citations['citation_date'].dt.month
    citations['qtr']   = ((citations['month'] - 1)//3 + 1).astype(str)
    citations['citation_quarter'] = citations['year'].astype(str) + 'Q' + citations['qtr']

    # Group by patent_id and citation_quarter, counting citations
    collapsed_citations = citations.groupby(['patent_id','citation_quarter']).size().reset_index(name='citation_count')
    return collapsed_citations

def save_collapsed_citations(citations: pd.DataFrame):
    """Save the collapsed citations DataFrame to a pickle file."""

    output_path = os.path.join(OUTPUT_DIR, 'collapsed_citations.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(citations, f)

def combine_with_citations(matched_dict: dict,
                           periods_before: int,
                           collapsed_citations: pd.DataFrame) -> dict:
    # assume periods_before is negative (e.g., -4); use a symmetric window by default
    n_pre = abs(periods_before)
    periods_after = n_pre

    out = {}
    # Prepare lookup frames once
    cc_t = collapsed_citations.rename(
        columns={'patent_id':'treated_id','citation_count':'citations_treated','citation_quarter':'quarter'}
    )
    cc_c = collapsed_citations.rename(
        columns={'patent_id':'control_id','citation_count':'citations_control','citation_quarter':'quarter'}
    )

    for lam, df in matched_dict.items():
        df = df.copy()
        df['match_id'] = df.index
        # t0 = last pre_quarter + 1
        df['t0'] = df['pre_quarters'].apply(lambda L: pd.Period(L[-1], freq='Q') + 1)

        rel_range = list(range(-n_pre, periods_after + 1))
        df['rel_q'] = [rel_range] * len(df)
        exp = df.explode('rel_q', ignore_index=True)
        exp['quarter'] = (exp['t0'] + exp['rel_q']).astype(str)

        # merge counts
        exp = exp.merge(cc_t, on=['treated_id','quarter'], how='left')
        exp = exp.merge(cc_c, on=['control_id','quarter'], how='left')

        # fill zeros only for pre-treatment (rel_q <= 0)
        pre_mask = exp['rel_q'] <= 0
        exp.loc[pre_mask, ['citations_treated','citations_control']] = \
            exp.loc[pre_mask, ['citations_treated','citations_control']].fillna(0)

        out[lam] = exp

    return out

def combine_with_citations(matched_dict: dict,
                           periods_before: int,
                           collapsed_citations: pd.DataFrame) -> dict:
    # assume periods_before is negative (e.g., -4); use a symmetric window by default
    n_pre = abs(periods_before)
    periods_after = n_pre

    out = {}
    # Prepare lookup frames once
    cc_t = collapsed_citations.rename(
        columns={'patent_id':'treated_id','citation_count':'citations_treated','citation_quarter':'quarter'}
    )
    cc_c = collapsed_citations.rename(
        columns={'patent_id':'control_id','citation_count':'citations_control','citation_quarter':'quarter'}
    )

    for lam, df in matched_dict.items():
        df = df.copy()
        df['match_id'] = df.index
        # t0 = last pre_quarter + 1
        df['t0'] = df['pre_quarters'].apply(lambda L: pd.Period(L[-1], freq='Q') + 1)

        rel_range = list(range(-n_pre, periods_after + 1))
        df['rel_q'] = [rel_range] * len(df)
        exp = df.explode('rel_q', ignore_index=True)
        exp['rel_q'] = exp['rel_q'].astype('int16')          # ensure numeric
        exp['quarter'] = (pd.PeriodIndex(exp['t0'], freq='Q') + exp['rel_q']).astype(str)


        # merge counts
        exp = exp.merge(cc_t, on=['treated_id','quarter'], how='left')
        exp = exp.merge(cc_c, on=['control_id','quarter'], how='left')

        # fill zeros only for pre-treatment (rel_q <= 0)
        pre_mask = exp['rel_q'] <= 0
        exp.loc[pre_mask, ['citations_treated','citations_control']] = \
            exp.loc[pre_mask, ['citations_treated','citations_control']].fillna(0)

        out[lam] = exp

    return out

##### FUNCTIONAL UP UNTIL HERE, NEED TO EDIT THE FUNCTIONS BELOW
"""def combine_with_citations(matched_dict: dict,
                           periods_before: int , 
                           collapsed_citations: pd.DataFrame) -> dict:
    
    # Periods
    periods_after = periods_before 
    
    # Flatten citation counts once
    n_pre = abs(periods_before)
    output = {}

    for lam, df in matched_dict.items():
        # Prepare DataFrame
        df = df.copy()
        df['match_id'] = df.index

        # Generate the sequence of quarters per row
        def make_quarters(pre_q_list):
            tp = pd.Period(pre_q_list[-1], freq='Q') + 1
            pre = [str(tp - i) for i in range(n_pre, 0, -1)]
            post = [str(tp + i) for i in range(1, periods_after+1)]
            return pre + [str(tp)] + post

        df['quarter_seq'] = df['pre_quarters'].apply(make_quarters)

        # Explode to long format
        exp = df.explode('quarter_seq').rename(columns={'quarter_seq':'quarter'})

        # Merge treated and control counts
        exp = exp.merge(
            collapsed_citations.rename(columns={'patent_id':'treated_id','citation_count':'treated_count', 'citation_quarter':'quarter'}),
            on=['treated_id','quarter'], how='left'
        )
        exp = exp.merge(
            collapsed_citations.rename(columns={'patent_id':'control_id','citation_count':'control_count', 'citation_quarter':'quarter'}),
            on=['control_id','quarter'], how='left'
        )

        # Drop any rows after 2024Q4, then fill missing counts with 0
        cutoff = pd.Period('2024Q4', freq='Q')

        # ensure 'quarter' is a valid quarterly string
        q_idx = pd.PeriodIndex(exp['quarter'].astype(str), freq='Q')

        # 1) remove rows beyond cutoff
        exp = exp.loc[q_idx <= cutoff].copy()

        # 2) fill missing/NA counts with 0 (treated & control)
        exp[['citations_treated', 'citations_control']] = (
            exp[['citations_treated', 'citations_control']].fillna(0)
        )

        # (optional) enforce integer dtype
        exp[['citations_treated', 'citations_control']] = (
            exp[['citations_treated', 'citations_control']].astype('int64')
        )


        output[lam] = exp

    return output"""

# ---- Reshape to long format ----
"""def get_long_data(sample: dict) -> pd.DataFrame:
    all_dfs = []
    for lam, df in sample.items():
        df2 = df.copy(); df2['lambda'] = lam
        idc = ['treated_id','control_id','match_id','lambda',
               'mahalanobis_distance','cosine_distance','hybrid_distance']
        tcols = [c for c in df2 if c.startswith('q_') and c.endswith('_treated')]
        ccols = [c for c in df2 if c.startswith('q_') and c.endswith('_control')]
        dt = df2.melt(id_vars=idc, value_vars=tcols,
                      var_name='quarter_treated',   value_name='citations_treated')
        dc = df2.melt(id_vars=idc, value_vars=ccols,
                      var_name='quarter_control',   value_name='citations_control')
        dt['quarter'] = dt['quarter_treated'].str.replace('_treated','')
        dc['quarter'] = dc['quarter_control'].str.replace('_control','')
        merged = pd.merge(dt[idc + ['quarter','citations_treated']],
                          dc[idc + ['quarter','citations_control']],
                          on=idc + ['quarter'])
        all_dfs.append(merged)
    return pd.concat(all_dfs, ignore_index=True).sort_values(['treated_id','lambda','quarter'])"""

# ---- Process one file with timing ----
"""def process_sample(filepath: str, citations: pd.DataFrame, baseline_period: int) -> pd.DataFrame:
    print(f"\nProcessing {os.path.basename(filepath)}")
    start_total = time.perf_counter()

    # Normalize baseline period to negative
    bp = -baseline_period if baseline_period > 0 else baseline_period

    # 1) Load sample
    t0 = time.perf_counter()
    sample = load_pickle(filepath)
    t1 = time.perf_counter()
    print(f" - Loaded sample in {t1-t0:.3f}s")

    # 3) Load collapsed quarterly citation counts
    t4 = time.perf_counter()
    counts = load_pickle(citations)
    t5 = time.perf_counter()
    print(f" - Computed quarterly citations in {t5-t4:.3f}s")

    # 4) Combine matches with citations
    t6 = time.perf_counter()
    combined = combine_with_citations(
        matched_dict=sample,
        periods_before=bp,
        collapsed_citations=counts
    )    
    t7 = time.perf_counter()
    print(f" - Combined with citations in {t7-t6:.3f}s")

    total = time.perf_counter() - start_total
    print(f" -> Total process_sample time: {total:.3f}s")

    
    return combined"""

def process_sample(filepath: str, collapsed_counts: pd.DataFrame, baseline_period: int) -> dict:
    print(f"\nProcessing {os.path.basename(filepath)}")
    start_total = time.perf_counter()
    bp = -baseline_period if baseline_period > 0 else baseline_period  # ensure negative

    t0 = time.perf_counter()
    sample = load_pickle(filepath)
    t1 = time.perf_counter()
    print(f" - Loaded sample in {t1-t0:.3f}s")

    t6 = time.perf_counter()
    combined_dict = combine_with_citations(sample, periods_before=bp, collapsed_citations=collapsed_counts)
    t7 = time.perf_counter()
    print(f" - Combined with citations in {t7-t6:.3f}s")

    # 4) Concatenate lambdas to a single long DF
    df_long = pd.concat(
        [v.assign(lambda_val=k) for k, v in combined_dict.items()],
        ignore_index=True
    )

    total = time.perf_counter() - start_total
    print(f" -> Total process_sample time: {total:.3f}s")
    return df_long



# ---- Main pipeline ----
def finalize_all(cite,
                 patents, 
                 input_dir: str = INPUT_DIR,
                 output_dir: str = OUTPUT_DIR):

    os.makedirs(output_dir, exist_ok=True)
    for filepath in tqdm(glob.glob(os.path.join(input_dir, INPUT_PATTERN)), desc="Processing samples"):
        # Get parameters and baseline period length
        suffix = os.path.basename(filepath)[len("01 Hybrid matches - "):-4]
        params = parse_suffix(suffix)
        periods_before = params['baseline_period']

        df_long = process_sample(filepath, cite, periods_before)
        df_long = df_long.dropna(subset=['citations_treated','citations_control'])
        merged = pd.merge(df_long, patents,
                          left_on='treated_id', right_on='patent_id', how='inner')
        #merged['quarter'] = merged['quarter'].astype(str).str.replace('q_','').astype(int)
        for k,v in params.items(): merged[k] = v
        for ext in ('pkl','csv'):
            out_name = OUTPUT_FILE_TEMPLATE.format(suffix=suffix, ext=ext)
            out_path = os.path.join(output_dir, out_name)
            if ext == 'pkl':
                merged.to_pickle(out_path)
            else:
                merged.to_csv(out_path, index=False)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Finalize matched samples for estimation")
    p.add_argument('--input_dir',     default=INPUT_DIR)
    p.add_argument('--output_dir',    default=OUTPUT_DIR)
    p.add_argument('--citations_file',default=CITATIONS_FILE)
    p.add_argument('--patents_file',  default=PATENTS_FILE)
    args = p.parse_args()
    finalize_all(
    cite=args.citations_file,
    patents=args.patents_file,
    input_dir=args.input_dir,
    output_dir=args.output_dir )

    # report
    print("All samples processed.")
    

