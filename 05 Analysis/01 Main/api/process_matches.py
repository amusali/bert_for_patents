#!/usr/bin/env python3
"""
finalize_samples.py

Load matched samples, combine with citation and patent metadata,
and write final datasets for estimation.

Adjust the directory constants or pass via CLI to suit your environment.
"""
import os
import glob
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
INPUT_PATTERN = "01 Hybrid matches - *.pkl"
# Template for output filenames
OUTPUT_FILE_TEMPLATE = "Sample - {suffix}.{ext}"

# ---- Low‑level I/O and parsing ----
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def parse_suffix(suffix: str) -> dict:
    """
    Given a suffix string of the form:
      "M&A, baseline, 4q, caliper_0.0250, 10matches"
    or
      "M&A, top-tech, 90, 4q, caliper_0.0250, 10matches"
    returns a dict with keys:
      acq_type, top_tech (bool), top_tech_threshold (int|None),
      baseline_period (int), caliper (float), K (int)
    """
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
        # "top-tech" branch
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

def trim_citations(citations: pd.DataFrame, patent_ids: np.ndarray) -> pd.DataFrame:
    citations['patent_id'] = citations['patent_id'].astype(str)
    return citations[citations['patent_id'].isin(patent_ids)]

def precompute_quarterly_citations(cite_df: pd.DataFrame) -> dict:
    cite_df['citation_date'] = pd.to_datetime(cite_df['citation_date'])
    cite_df['year']  = cite_df['citation_date'].dt.year
    cite_df['month'] = cite_df['citation_date'].dt.month
    cite_df['qtr']   = ((cite_df['month'] - 1)//3 + 1).astype(str)
    cite_df['citation_quarter'] = cite_df['year'].astype(str) + 'Q' + cite_df['qtr']
    grp = cite_df.groupby(['patent_id','citation_quarter']).size().unstack(fill_value=0)
    return { pid: row.to_dict() for pid, row in grp.iterrows() }

# ---- Merge matched samples with citations ----
def combine_with_citations(sample: dict, citation_counts: dict,
                           periods_before: int = -4, periods_after: int = 20) -> dict:
    for lam, df in sample.items():
        treated_cits, control_cits = [], []
        for _, row in df.iterrows():
            pre_q = row['pre_quarters']
            treat_period = pd.Period(pre_q[-1], freq='Q') + 1
            # build quarters t-4 ... t+20
            quarters = ([str(treat_period - i) for i in range(periods_before,0,-1)] +
                        [str(treat_period)] +
                        [str(treat_period + i) for i in range(1, periods_after+1)])
            tvec, cvec = [], []
            last_q = pd.Period('2024Q4', freq='Q')
            for q in quarters:
                qpd = pd.Period(q, freq='Q')
                tcount = citation_counts.get(row['treated_id'], {}).get(q,
                              np.nan if qpd>last_q else 0)
                ccount = citation_counts.get(row['control_id'], {}).get(q,
                              np.nan if qpd>last_q else 0)
                tvec.append(tcount)
                cvec.append(ccount)
            treated_cits.append(tvec)
            control_cits.append(cvec)
        labels = [f"q_{i}" for i in range(periods_before, periods_after+1)]
        for i, lab in enumerate(labels):
            df[lab + '_treated'] = [vec[i] for vec in treated_cits]
            df[lab + '_control'] = [vec[i] for vec in control_cits]
        df['match_id'] = df.index
    return sample

# ---- Reshape to long format ----
def get_long_data(sample: dict) -> pd.DataFrame:
    all_dfs = []
    for lam, df in sample.items():
        df2 = df.copy(); df2['lambda'] = lam
        idc = ['treated_id','control_id','treated_vector', 'control_vector', 'lambda',
               'mahalanobis_distance', 'mahalanobis_distance_scaled','cosine_distance', 'cosine_distance_scaled', 
               'hybrid_distance']
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
    return pd.concat(all_dfs, ignore_index=True).sort_values(['treated_id','lambda','quarter'])

# ---- Process one file ----
def process_sample(filepath: str, citations: pd.DataFrame, baseline_period) -> pd.DataFrame:
    sample = load_pickle(filepath)
    ids    = get_unique_ids(sample)
    trim   = trim_citations(citations, ids)
    counts = precompute_quarterly_citations(trim)
    combined = combine_with_citations(sample, counts, periods_before=baseline_period)
    return get_long_data(combined)

# ---- Main pipeline ----
def finalize_all(input_dir: str = INPUT_DIR,
                 output_dir: str = OUTPUT_DIR,
                 citations_file: str = CITATIONS_FILE,
                 patents_file: str = PATENTS_FILE):
    # load and rename citations
    cite = pd.read_pickle(citations_file)
    cite.rename(columns={'patent_id':'citedby_patent_id',
                         'citation_patent_id':'patent_id',
                         'filing_date':'citation_date'}, inplace=True)
    # load patents metadata
    patents = pd.read_stata(patents_file)
    patents['patent_id'] = patents['patent_id'].astype(str)

    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(input_dir, INPUT_PATTERN)

    for filepath in tqdm(glob.glob(pattern), desc="Processing samples"):

        # Report progress
        print(f"Processing {filepath}...")

        # extract suffix and parameters
        base = os.path.basename(filepath)
        if not base.endswith('.pkl'): continue
        suffix = base[len("01 Hybrid matches - "):-4]
        params = parse_suffix(suffix)
        df_long = process_sample(filepath, cite, params['baseline_period'])
        # drop any nan citations
        df_long = df_long.dropna(subset=['citations_treated','citations_control'])
        assert df_long['citations_treated'].notna().all()
        assert df_long['citations_control'].notna().all()
        # merge with patents info
        merged = pd.merge(df_long, patents,
                          left_on='treated_id', right_on='patent_id', how='inner')
        # quarter -> int
        merged['quarter'] = merged['quarter'].astype(str).str.replace('q_','').astype(int)
        # attach scenario params
        for k,v in params.items(): merged[k] = v
        # save both formats
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
    finalize_all(args.input_dir, args.output_dir,
                 args.citations_file, args.patents_file)
