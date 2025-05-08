import numpy as np
import pandas as pd
from numpy.linalg import norm, pinv

def get_pre_quarters(acq_date, pretreatment_periods=[1,2,3,4]):
    """Return list of pre-treatment quarter strings t-p for each p in pretreatment_periods."""
    acq_period = pd.Period(acq_date, freq='Q')
    return [str(acq_period - p) for p in pretreatment_periods]

def get_citation_vector(patent_id, citation_counts_dict, pre_quarters):
    """Return citation count vector for patent_id over specified quarters."""
    counts = citation_counts_dict.get(patent_id, {})
    return np.array([counts.get(q, 0) for q in pre_quarters], dtype=float)

def compute_cosine_distances(vec, mat):
    """Compute cosine distance between vec and each row in mat."""
    vec_norm = norm(vec) + 1e-10
    mat_norms = np.linalg.norm(mat, axis=1) + 1e-10
    sims = np.dot(mat, vec) / (mat_norms * vec_norm)
    return 1 - sims

def match_embeddings_then_citations(treated_df, control_df, citation_counts_dict,
                                    caliper=0.2, pretreatment_periods=[1,2,3,4],
                                    group_cols=None, embedding_col='embedding',
                                    date_col='acq_date'):
    """Sequential matching: first embeddings then citations."""
    results = []
    for _, tre in treated_df.iterrows():
        treated_id = tre['patent_id']
        t_embed = tre[embedding_col]
        acq_date = tre[date_col]
        pre_quarters = get_pre_quarters(acq_date, pretreatment_periods)
        cands = control_df
        if group_cols:
            for col in group_cols:
                cands = cands[cands[col] == tre[col]]
        if cands.empty:
            results.append({'treated_id': treated_id, 'control_id': None,
                            'cosine_distance': np.nan, 'mahalanobis_distance': np.nan})
            continue
        cand_embs = np.stack(cands[embedding_col].values)
        cosine_d = compute_cosine_distances(t_embed, cand_embs)
        idxs = np.where(cosine_d <= caliper)[0]
        if len(idxs) == 0:
            results.append({'treated_id': treated_id, 'control_id': None,
                            'cosine_distance': float(np.min(cosine_d)), 'mahalanobis_distance': np.nan})
            continue
        treated_vec = get_citation_vector(treated_id, citation_counts_dict, pre_quarters)
        cand_ids = cands['patent_id'].values[idxs]
        cand_vecs = np.vstack([get_citation_vector(cid, citation_counts_dict, pre_quarters) for cid in cand_ids])
        if cand_vecs.shape[0] > 1:
            cov = np.cov(cand_vecs, rowvar=False)
            inv_cov = pinv(cov)
        else:
            inv_cov = np.eye(cand_vecs.shape[1])
        diffs = cand_vecs - treated_vec
        m_d = np.sqrt(np.sum(diffs @ inv_cov * diffs, axis=1))
        best = np.argmin(m_d)
        results.append({
            'treated_id': treated_id,
            'control_id': cand_ids[best],
            'cosine_distance': float(cosine_d[idxs][best]),
            'mahalanobis_distance': float(m_d[best])
        })
    return pd.DataFrame(results)

def match_citations_then_embeddings(treated_df, control_df, citation_counts_dict,
                                    top_k=10, pretreatment_periods=[1,2,3,4],
                                    group_cols=None, embedding_col='embedding',
                                    date_col='acq_date'):
    """Sequential matching: first citations then embeddings."""
    results = []
    for _, tre in treated_df.iterrows():
        treated_id = tre['patent_id']
        t_embed = tre[embedding_col]
        acq_date = tre[date_col]
        pre_quarters = get_pre_quarters(acq_date, pretreatment_periods)
        treated_vec = get_citation_vector(treated_id, citation_counts_dict, pre_quarters)
        cands = control_df
        if group_cols:
            for col in group_cols:
                cands = cands[cands[col] == tre[col]]
        if cands.empty:
            results.append({'treated_id': treated_id, 'control_id': None,
                            'mahalanobis_distance': np.nan, 'cosine_distance': np.nan})
            continue
        cand_ids_all = cands['patent_id'].values
        cand_vecs_all = np.vstack([get_citation_vector(cid, citation_counts_dict, pre_quarters) for cid in cand_ids_all])
        if cand_vecs_all.shape[0] > 1:
            cov = np.cov(cand_vecs_all, rowvar=False)
            inv_cov = pinv(cov)
        else:
            inv_cov = np.eye(cand_vecs_all.shape[1])
        diffs_all = cand_vecs_all - treated_vec
        m_d_all = np.sqrt(np.sum(diffs_all @ inv_cov * diffs_all, axis=1))
        top_idxs = np.argsort(m_d_all)[:top_k]
        cand_ids = cand_ids_all[top_idxs]
        cand_embs = np.stack(cands.iloc[top_idxs][embedding_col].values)
        cosine_d = compute_cosine_distances(t_embed, cand_embs)
        best = np.argmin(cosine_d)
        results.append({
            'treated_id': treated_id,
            'control_id': cand_ids[best],
            'mahalanobis_distance': float(m_d_all[top_idxs][best]),
            'cosine_distance': float(cosine_d[best])
        })
    return pd.DataFrame(results)

def sequential_match(treated_df, control_df, citation_counts_dict,
                     caliper=0.2, top_k=10, pretreatment_periods=[1,2,3,4],
                     order='emb_then_cit', group_cols=None,
                     embedding_col='embedding', date_col='acq_date'):
    """Wrapper for sequential matching. order: 'emb_then_cit' or 'cit_then_emb'."""
    if order == 'emb_then_cit':
        return match_embeddings_then_citations(treated_df, control_df, citation_counts_dict,
                                               caliper=caliper, pretreatment_periods=pretreatment_periods,
                                               group_cols=group_cols, embedding_col=embedding_col, date_col=date_col)
    elif order == 'cit_then_emb':
        return match_citations_then_embeddings(treated_df, control_df, citation_counts_dict,
                                               top_k=top_k, pretreatment_periods=pretreatment_periods,
                                               group_cols=group_cols, embedding_col=embedding_col, date_col=date_col)
    else:
        raise ValueError("order must be 'emb_then_cit' or 'cit_then_emb'")
