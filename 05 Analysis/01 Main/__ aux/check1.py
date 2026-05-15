#@title Routine
# ============================================================
# D. Main loop (Python / moderndid)
# ============================================================
import gc
import time
from pathlib import Path
import dill as pickle
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt

#from moderndid import att_gt, aggte
#from moderndid import aggte, att_gt
import sys, importlib

for m in [k for k in list(sys.modules) if k.startswith("moderndid")]:
    del sys.modules[m]

import moderndid as did
importlib.reload(did)

run_log = []
t0_all = time.time()

end_of_sample_qid = int(((2024 - 1960) * 4) + 4 - 1)   # 2024Q4
print("end_of_sample_qid =", end_of_sample_qid)


def write_text_summary(path, out, es):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== ATT(g,t) summary ===\n")
        f.write(str(out))
        f.write("\n\n=== Dynamic / event-study summary ===\n")
        f.write(str(es))
        f.write("\n")


def result_to_attgt_df(out):
    # Based on current moderndid MPResult docs
    df = pd.DataFrame({
        "group": np.asarray(out.groups),
        "time": np.asarray(out.times),
        "att": np.asarray(out.att_gt),
        "se": np.asarray(out.se_gt),
    })
    crit = getattr(out, "critical_value", None)
    if crit is not None:
        df["crit_val"] = crit
    return df


def result_to_dynamic_df(es):
    # Based on current moderndid AGGTEResult docs
    df = pd.DataFrame({
        "event_time": np.asarray(es.event_times),
        "att": np.asarray(es.att_by_event),
        "se": np.asarray(es.se_by_event),
    })
    crit = getattr(es, "critical_values", None)
    if crit is not None:
        # sometimes scalar, sometimes array-like depending on implementation
        if np.isscalar(crit):
            df["crit_val"] = crit
        else:
            crit = np.asarray(crit)
            if crit.shape[0] == df.shape[0]:
                df["crit_val"] = crit
    return df


def plot_group_time(attgt_df, title, path):
    plt.figure(figsize=(11, 8))
    plt.scatter(attgt_df["time"], attgt_df["group"], s=18)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Group")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_event_study(dynamic_df, title, path):
    z = 1.96
    lo = dynamic_df["att"] - z * dynamic_df["se"]
    hi = dynamic_df["att"] + z * dynamic_df["se"]

    plt.figure(figsize=(10, 6))
    plt.axhline(0, linewidth=1)
    plt.axvline(-0.5, linewidth=1, linestyle="--")
    plt.plot(dynamic_df["event_time"], dynamic_df["att"], marker="o")
    plt.fill_between(dynamic_df["event_time"], lo, hi, alpha=0.2)
    plt.title(title)
    plt.xlabel("Event time")
    plt.ylabel("ATT")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


for pre_treatment_period in pre_treatment_periods:
    for acq_type in acq_types:
        for caliper in calipers:
            for lambda_value in lambdas:
              for outcome_var in outcome_variables:

                  filename = build_sample_filename(
                      acq_type=acq_type,
                      pre_treatment_period=pre_treatment_period,
                      caliper=caliper,
                      base_tt=base_tt,
                      base_tt_threshold=base_tt_threshold,
                      lambda_val=lambda_value
                  )
                  est_stem = build_est_stem(filename)
                  est_range_str = f"-{pre_treatment_period} - {last_post_treatment_period}"

                  out_prefix = safe_filename(
                      f"{est_stem}, lambda{lambda_value}, p{est_range_str}, {outcome_var}"
                  )

                  ## Skip if estiamted

                  out_path = est_out_dir / f"{out_prefix} - out.pkl"
                  if out_path.is_file():
                      print(f"Estimation {out_prefix} exists, skipping")
                      continue

                  attgt_csv_path   = est_out_dir / f"{out_prefix} - attgt.csv"
                  dynamic_csv_path = est_out_dir / f"{out_prefix} - event_dynamic.csv"
                  summary_txt_path = est_out_dir / f"{out_prefix} - summaries.txt"
                  gt_plot_path     = graph_out_dir / f"{out_prefix} - group_time_plot.png"
                  es_plot_path     = graph_out_dir / f"{out_prefix} - event_plot.png"

                  print(f"\n=== Running: {filename} | lambda = {lambda_value} ===")
                  t0 = time.time()

                  # --------------------------------------------------------
                  # Load config-specific sample and filter lambda
                  # --------------------------------------------------------
                  sample_path = dta_dir / filename
                  df, _meta = pyreadstat.read_dta(sample_path)
                  df.columns = [c.lower() for c in df.columns]
                  print("Rows after load:", len(df))

                  if "lambda" in df.columns:
                      df = df[pd.to_numeric(df["lambda"], errors="coerce") == lambda_value].copy()
                  elif "lambda_val" in df.columns:
                      df = df[pd.to_numeric(df["lambda_val"], errors="coerce") == lambda_value].copy()
                  else:
                      raise ValueError("No lambda column found in sample file.")

                  df["patent_id"] = df["patent_id"].astype(str)
                  print("Rows after lambda filter:", len(df))

                  # --------------------------------------------------------
                  # Merge PCA and patents
                  # --------------------------------------------------------
                  df = df.merge(
                      pca[["patent_id"] + pca_cols],
                      on="patent_id",
                      how="left"
                  )
                  df = df.merge(
                      patents[["patent_id", "cpc_subclass_current", "grant_date", "grant_quarter", "grant_year"]],
                      on="patent_id",
                      how="left",
                      suffixes=("", "_pat")
                  )
                  print("Rows after merges:", len(df))

                  if "cpc_subclass_current_pat" in df.columns:
                      df["cpc_subclass_current"] = df["cpc_subclass_current_pat"].combine_first(df.get("cpc_subclass_current"))
                      df["grant_date"] = df["grant_date_pat"].combine_first(df.get("grant_date"))
                      df["grant_quarter"] = df["grant_quarter_pat"].combine_first(df.get("grant_quarter"))
                      df["grant_year"] = df["grant_year_pat"].combine_first(df.get("grant_year"))

                      drop_cols = [c for c in df.columns if c.endswith("_pat")]
                      df = df.drop(columns=drop_cols)

                  # --------------------------------------------------------
                  # Clean types and covariates
                  # --------------------------------------------------------
                  df["treated"] = pd.to_numeric(df["treated"], errors="coerce")
                  df["citation"] = pd.to_numeric(df["citation"], errors="coerce")
                  df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce")
                  df["cohort"] = pd.to_numeric(df["cohort"], errors="coerce")
                  df["grant_quarter"] = pd.to_numeric(df["grant_quarter"], errors="coerce")
                  df["grant_year"] = pd.to_numeric(df["grant_year"], errors="coerce").astype("Int64")
                  df["cpc_subclass_current"] = df["cpc_subclass_current"].astype(str)

                  df["cpc"] = df["cpc_subclass_current"]
                  df["log_citation"] = np.log1p(df["citation"])
                  df["age"] = df["quarter"] - df["grant_quarter"]
                  df["age_sq"] = df["age"] ** 2
                  df["active"] = (df["age"] <= 80).astype(int)

                  print("Rows after type cleaning:", len(df))

                  # --------------------------------------------------------
                  # Sample filters
                  # --------------------------------------------------------
                  # Drop treated patents with insufficient post horizon
                  df = df.loc[~((df["treated"] == 1) & (df["cohort"] + last_post_treatment_period > end_of_sample_qid))].copy()
                  print("Rows after post-horizon treated filter:", len(df))

                  # Keep within event window or never-treated
                  df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
                  df["acq_date_q"] = (df["acq_date"].dt.year - 1960) * 4 + df["acq_date"].dt.quarter - 1
                  df["quarter_to_treatment"] = df["quarter"] - df["acq_date_q"]

                  df = df.loc[
                      df["acq_date"].isna() |
                      df["quarter_to_treatment"].between(-pre_treatment_period, last_post_treatment_period)
                  ].copy()

                  df = df.drop(columns=["acq_date_q"])
                  print("Rows after event-window filter:", len(df))

                  # cohort sizes
                  num_periods_per_treated = pre_treatment_period + last_post_treatment_period + 1

                  df["num_patents_in_cohort"] = df.groupby("cohort")["patent_id"].transform("size")

                  check_tbl = (
                      df.loc[df["treated"] == 1, ["cohort", "num_patents_in_cohort"]]
                        .drop_duplicates()
                  )

                  if len(check_tbl) > 0:
                      ok = (check_tbl["num_patents_in_cohort"] % num_periods_per_treated == 0).all()
                      if not ok:
                          raise ValueError("Treated cohort size is not divisible by num_periods_per_treated.")

                  df = df.loc[
                      ~((df["treated"] == 1) & (df["num_patents_in_cohort"] < 20 * num_periods_per_treated))
                  ].copy()

                  # full sample horizon
                  df = df.loc[~(df["grant_quarter"] + last_post_treatment_period > end_of_sample_qid)].copy()

                  # drop quarter cells with zero treated
                  df["num_treated"] = df.groupby("quarter")["treated"].transform(lambda s: np.nansum(s))
                  df = df.loc[df["num_treated"] != 0].copy()

                  print("Rows after final sample filters:", len(df))

                  # --------------------------------------------------------
                  # Final checks
                  # --------------------------------------------------------
                  dup_check = (
                      df.groupby(["patent_id", "quarter"], dropna=False)
                        .size()
                        .reset_index(name="n")
                  )
                  dup_check = dup_check.loc[dup_check["n"] > 1]

                  if len(dup_check) > 0:
                      raise ValueError("patent_id-quarter is not unique")

                  needed = [
                      "patent_id", "quarter", "citation", "cohort", "log_citation",
                      "age", "age_sq", "active", "grant_year", "cpc"
                  ]
                  for v in needed:
                      if df[v].isna().any():
                          raise ValueError(f"Missing values in required column: {v}")

                  if (df["citation"] < 0).any() or (df["log_citation"] < 0).any():
                      raise ValueError("Negative citation or log_citation found")

                  # untreated group should be 0 for gname
                  df["cohort"] = df["cohort"].fillna(0)

                  # --------------------------------------------------------
                  # Merge Nearby patents info
                  # --------------------------------------------------------

                  # merge nearby outcomes
                  check = len(df)
                  print(f"Number of rows in df before merging with nearby counts: {len(df)}")
                  df = df.merge(
                      nearby_lookup,
                      left_on = ['patent_id', 'quarter'],
                      right_on = ['patent_id', 'quarter_stata'],
                      how="inner",
                      validate="many_to_one"
                  )
                  check1 = len(df)

                  assert check == check1, "Merge failed"

                  print(f"Number of rows in df after merging with nearby counts: {len(df)}")

                  # --------------------------------------------------------
                  # Covariates / formula
                  # --------------------------------------------------------
                  # Keep these as strings for formula-based handling
                  df["grant_year_code"] = df["grant_year"].astype("category") #.cat.codes.astype("int32")
                  df["cpc_code"] = df["cpc"].astype("category") #.cat.codes.astype("int32")

                  df["grant_year_f"] = df["grant_year"].astype("int32")
                  df["cpc_f"] = df["cpc"].astype("str")

                  xformla = "~ grant_year_f + cpc_f + " + " + ".join(pca_cols)

                  # make dummies
                  gy_dum = pd.get_dummies(df["grant_year"], prefix="gy", drop_first=True, dtype=np.float32)
                  cpc_dum = pd.get_dummies(df["cpc"], prefix="cpc", drop_first=True, dtype=np.float32)

                  df = pd.concat([df, gy_dum, cpc_dum], axis=1)

                  # force PCA numeric
                  for c in pca_cols:
                      df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

                  # choose covariates
                  if lambda_value == 1:
                      covar_cols = list(gy_dum.columns) + list(cpc_dum.columns)
                  else:
                      covar_cols = list(gy_dum.columns) + list(cpc_dum.columns) + pca_cols

                  # final force-to-numeric
                  for c in covar_cols:
                      df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

                  # sanity check
                  bad = df[covar_cols].dtypes[df[covar_cols].dtypes == "object"]
                  print("object columns:", bad)
                  print(df[covar_cols].dtypes.value_counts())

                  xformla = "~ " + " + ".join(covar_cols)

                  print("Rows:", len(df), "| patents:", df["patent_id"].nunique())

                  # stable numeric id
                  df["patent_id_num"] = pd.factorize(df["patent_id"])[0] + 1

                  # --------------------------------------------------------
                  # Estimate ATT(g,t)
                  # --------------------------------------------------------
                  # moderndid currently supports backend='numpy' or 'cupy'
                  # and parallel jobs through n_jobs. API is still under development.
                  out = did.att_gt(
                      data=df,
                      yname=outcome_var,
                      tname="quarter",
                      gname="cohort",
                      xformla=xformla,
                      panel=False,
                      #allow_unbalanced_panel=True,
                      control_group="nevertreated",
                      est_method="reg",
                      boot=False,
                      random_state=seed,
                      backend="numpy",   # change to "numpy" if GPU/CuPy not available
                  )
                  print("estimation done !!!")

                  # Event-study aggregation
                  es = did.aggte(
                      MP=out,
                      type="dynamic",
                      na_rm=True,
                      min_e=-pre_treatment_period,
                      max_e=last_post_treatment_period,
                  )
                  print(es)

                  # Set influence functions to None - heavy objects dont need to be saved
                  out = out._replace(influence_func=None)
                  es = es._replace(influence_func=None)
                  es = es._replace(influence_func_overall=None)

                  # Get p values of Wald pretreatment tests
                  walt_p_val = out.wald_pvalue
                  wald_stat = out.wald_stat


                  # --------------------------------------------------------
                  # Save outputs
                  # --------------------------------------------------------

                  out_path = est_out_dir / f"{out_prefix} - out.pkl"
                  es_path = est_out_dir / f"{out_prefix} - es.pkl"
                  es_csv_path = est_out_dir / f"{out_prefix} - event study.csv"

                  with open(out_path, "wb") as f:
                    pickle.dump(out, f)

                  with open(es_path, "wb") as f:
                    pickle.dump(es, f)

                  from moderndid.core.converters import aggteresult_to_polars

                  es_pd = aggteresult_to_polars(es).to_pandas()
                  es_pd["p_value"] = walt_p_val
                  es_pd["wald_stat"] = wald_stat
                  es_pd.to_csv(es_csv_path, index = False)

                  attgt_df = result_to_attgt_df(out)
                  dynamic_df = result_to_dynamic_df(es)

                  attgt_df.to_csv(attgt_csv_path, index=False)
                  dynamic_df.to_csv(dynamic_csv_path, index=False)

                  write_text_summary(summary_txt_path, out, es)

                  # --------------------------------------------------------
                  # Plots
                  # --------------------------------------------------------


                  elapsed = time.time() - t0

                  run_log.append({
                      "filename": filename,
                      "outcome_var": outcome_var,
                      "lambda": lambda_value,
                      "pre_treatment_period": pre_treatment_period,
                      "acq_type": acq_type,
                      "caliper": caliper,
                      "n_rows": len(df),
                      "n_patents": df["patent_id"].nunique(),
                      "n_treated_rows": int((df["treated"] == 1).sum()),
                      "n_treated_patents": int(df.loc[df["treated"] == 1, "patent_id"].nunique()),
                      "seconds": elapsed,
                      "status": "ok",
                  })

                  print(f"Done in {elapsed:.1f} seconds -> {out_prefix}")

                  del df, out, es, attgt_df, dynamic_df
                  gc.collect()

# ============================================================
# E. Save run log
# ============================================================
run_log_df = pd.DataFrame(run_log)
run_log_path = log_dir / "08a_estimate_python_run_log_MA_bl_nearby.csv"
run_log_df.to_csv(run_log_path, index=False)

total_minutes = (time.time() - t0_all) / 60
print(f"\nTotal elapsed: {total_minutes:.1f} minutes")
print(f"Run log saved to: {run_log_path}")