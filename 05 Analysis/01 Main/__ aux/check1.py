#@title Overwrite the functions

from pathlib import Path
import importlib
import sys

path = Path("/usr/local/lib/python3.12/dist-packages/moderndid/did/compute_att_gt.py")

new_source = r'''
"""Multi-period difference-in-differences group-time ATT computation."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from moderndid.core.parallel import parallel_map
from moderndid.core.preprocess import ControlGroup, DIDData, EstimationMethod
from moderndid.cupy.backend import get_backend
from moderndid.drdid.estimators.drdid_panel import drdid_panel
from moderndid.drdid.estimators.drdid_rc import drdid_rc
from moderndid.drdid.estimators.reg_did_panel import reg_did_panel
from moderndid.drdid.estimators.reg_did_rc import reg_did_rc
from moderndid.drdid.estimators.std_ipw_did_panel import std_ipw_did_panel
from moderndid.drdid.estimators.std_ipw_did_rc import std_ipw_did_rc


class ATTgtResult(NamedTuple):
    """Container for a single group-time ATT estimate.

    Attributes
    ----------
    att : float
        The estimated ATT for this group-time cell.
    group : float
        The treatment group identifier (first treatment period).
    year : float
        The time period for this estimate.
    post : int
        Indicator for whether this is a post-treatment period (1) or
        pre-treatment period (0).
    """

    #: Estimated ATT for this group-time cell.
    att: float
    #: Treatment group identifier (first treatment period).
    group: float
    #: Time period for this estimate.
    year: float
    #: Indicator for post-treatment (1) or pre-treatment (0).
    post: int


class ComputeATTgtResult(NamedTuple):
    """Container for compute_att_gt results.

    Attributes
    ----------
    attgt_list : list[ATTgtResult]
        List of group-time ATT results, one per (group, time) cell.
    influence_functions : scipy.sparse.csr_matrix
        Sparse matrix of influence functions with shape
        (n_units, n_group_time_cells).
    """

    #: List of group-time ATT results, one per (group, time) cell.
    attgt_list: list[ATTgtResult]
    #: Sparse matrix of influence functions.
    influence_functions: sp.csr_matrix


def compute_att_gt(data: DIDData, n_jobs=1):
    """Compute group-time average treatment effects.

    Parameters
    ----------
    data : DIDData
        Preprocessed DiD data object containing all necessary data and configuration.
    n_jobs : int, default=1
        Number of parallel jobs. 1 = sequential, -1 = all cores, >1 = that many workers.

    Returns
    -------
    ComputeATTgtResult
        NamedTuple containing list of ATT results and influence functions
    """
    n_units = data.config.id_count
    time_periods = data.config.time_periods

    n_time_periods = len(time_periods) - 1 if data.config.base_period != "universal" else len(time_periods)
    group_time_pairs = [(g, t) for g in range(data.config.treated_groups_count) for t in range(n_time_periods)]

    args_list = [(g_idx, t_idx, data) for g_idx, t_idx in group_time_pairs]
    cell_results = parallel_map(_process_gt_cell_did, args_list, n_jobs=n_jobs)

    att_results = []
    influence_func_list = []

    for att_result, inf_func in cell_results:
        if att_result is not None:
            att_results.append(att_result)
            influence_func_list.append(inf_func)

    if influence_func_list:
        xp = get_backend()
        if xp is not np:
            try:
                import cupyx.scipy.sparse as cusp

                influence_matrix = xp.column_stack([xp.asarray(f) for f in influence_func_list])
                sparse_influence_funcs = cusp.csr_matrix(influence_matrix)
            except ImportError:
                influence_matrix = np.column_stack(influence_func_list)
                sparse_influence_funcs = sp.csr_matrix(influence_matrix)
        else:
            influence_matrix = np.column_stack(influence_func_list)
            sparse_influence_funcs = sp.csr_matrix(influence_matrix)
    else:
        sparse_influence_funcs = sp.csr_matrix((n_units, 0))

    return ComputeATTgtResult(attgt_list=att_results, influence_functions=sparse_influence_funcs)


def run_att_gt_estimation(
    group_idx,
    time_idx,
    data,
):
    """Run ATT estimation for a given group-time pair.

    Parameters
    ----------
    group_idx : int
        Index of the treated group.
    time_idx : int
        Index of the time period.
    data : DIDData
        Preprocessed DiD data object.

    Returns
    -------
    dict or None
        Dictionary with ATT and influence function, or None if estimation not feasible.
    """
    time_factor = 1 if data.config.base_period != "universal" else 0

    if data.config.base_period == "universal":
        pre_periods = np.where(
            data.config.time_periods < (data.config.treated_groups[group_idx] - data.config.anticipation)
        )[0]
        pre_treatment_idx = pre_periods[-1] if len(pre_periods) > 0 else None
    else:
        pre_treatment_idx = time_idx

    is_post_treatment = data.config.treated_groups[group_idx] <= data.config.time_periods[time_idx + time_factor]

    if is_post_treatment and data.config.base_period != "universal":
        pre_periods = np.where(
            data.config.time_periods < (data.config.treated_groups[group_idx] - data.config.anticipation)
        )[0]

        if len(pre_periods) == 0:
            warnings.warn(
                f"No pre-treatment periods for group first treated at {data.config.treated_groups[group_idx]}. "
                "Units from this group are dropped.",
                UserWarning,
            )
            return None

        pre_treatment_idx = pre_periods[-1]

    if (
        data.config.base_period == "universal"
        and pre_treatment_idx is not None
        and data.config.time_periods[pre_treatment_idx] == data.config.time_periods[time_idx + time_factor]
    ):
        return None

    cohort_index = get_did_cohort_index(group_idx, time_idx, time_factor, pre_treatment_idx, data)

    has_treated = np.any(cohort_index == 1)
    has_control = np.any(cohort_index == 0)

    if not (has_treated and has_control):
        return None

    if data.config.panel:
        cohort_data = {
            "D": cohort_index,
            "y1": data.outcomes_tensor[time_idx + time_factor],
            "y0": data.outcomes_tensor[pre_treatment_idx],
            "weights": data.weights,
        }
        covariates = data.covariates_tensor[min(pre_treatment_idx, time_idx)]
    else:
        post_mask = (data.data[data.config.tname] == data.config.time_periods[time_idx + time_factor]).to_numpy()
        cohort_data = {
            "D": cohort_index,
            "y": data.data[data.config.yname].to_numpy(),
            "post": post_mask.astype(int),
            "weights": data.data["weights"].to_numpy(),
        }
        if data.config.allow_unbalanced_panel:
            cohort_data["rowid"] = data.data[".rowid"].to_numpy()
        covariates = data.covariates_matrix

    if not callable(data.config.est_method):
        valid_obs = ~np.isnan(cohort_index)
        d_valid = cohort_index[valid_obs]
        g_val = data.config.treated_groups[group_idx]
        t_val = data.config.time_periods[time_idx + time_factor]

        if not data.config.panel:
            post = cohort_data["post"][valid_obs]
            G = d_valid == 1
            C = d_valid == 0
            skip = False
            if np.sum(G & (post == 1)) == 0:
                warnings.warn(f"No treated units in group {g_val} in time period {t_val}", UserWarning)
                skip = True
            if np.sum(G & (post == 0)) == 0:
                warnings.warn(f"No treated units in group {g_val} in pre-treatment period", UserWarning)
                skip = True
            if np.sum(C & (post == 1)) == 0:
                warnings.warn(f"No control units for group {g_val} in time period {t_val}", UserWarning)
                skip = True
            if np.sum(C & (post == 0)) == 0:
                warnings.warn(f"No control units for group {g_val} in pre-treatment period", UserWarning)
                skip = True
            if skip:
                return None

        overlap_violated = False
        reg_ill_conditioned = False

        if data.config.est_method in (EstimationMethod.DOUBLY_ROBUST, EstimationMethod.IPW):
            cov_valid = covariates[valid_obs] if covariates.ndim > 1 else np.ones((valid_obs.sum(), 1))
            G_valid = (d_valid == 1).astype(float)
            try:
                logit_model = sm.GLM(G_valid, cov_valid, family=sm.families.Binomial())
                logit_result = logit_model.fit(maxiter=100, disp=False)
                pscores = logit_result.fittedvalues
                if np.max(pscores) >= 0.999:
                    overlap_violated = True
                    warnings.warn(
                        f"Overlap condition violated for {g_val} in time period {t_val}",
                        UserWarning,
                    )
            except (np.linalg.LinAlgError, PerfectSeparationError):
                pass

                ### Change
        from scipy.linalg import qr as scipy_qr

        try:
            import cupy as cp
            HAS_CUPY = True
        except Exception:
            cp = None
            HAS_CUPY = False


        def keep_lindep_cols_from_controls(X, tol=1e-10, keep_intercept=True):
            """
            Column reduction on CPU using SciPy pivoted QR.
            Returns:
                X_reduced_cpu, keep_var, keep_qr
            where keep_var and keep_qr are NumPy index arrays/masks.
            """
            # explicitly move to CPU if this is a CuPy array
            if HAS_CUPY and isinstance(X, cp.ndarray):
                X_cpu = X.get()
            else:
                X_cpu = np.asarray(X)

            if X_cpu.ndim == 1:
                X_cpu = X_cpu.reshape(-1, 1)

            is_const_one = np.array([
                np.all(np.isfinite(col)) and np.max(np.abs(col - 1.0)) < tol
                for col in X_cpu.T
            ])

            is_nonconst = np.array([
                np.var(col[np.isfinite(col)]) > tol if np.isfinite(col).sum() > 1 else False
                for col in X_cpu.T
            ])

            keep_var = is_nonconst | (keep_intercept & is_const_one)

            if not np.any(keep_var):
                raise ValueError("No covariate columns with variation remain in this control subsample.")

            X1 = X_cpu[:, keep_var]

            # pivoted QR on CPU
            q, r, piv = scipy_qr(X1, mode="economic", pivoting=True)
            diag_r = np.abs(np.diag(r))
            rank = int(np.sum(diag_r > tol))
            keep_qr = np.sort(piv[:rank])

            if keep_qr.size == 0:
                raise ValueError("No linearly independent covariate columns remain in this control subsample.")

            X2 = X1[:, keep_qr]
            return X2, keep_var, keep_qr
        """
        if data.config.est_method in (EstimationMethod.DOUBLY_ROBUST, EstimationMethod.REGRESSION):
            cov_valid = covariates[valid_obs] if covariates.ndim > 1 else np.ones((valid_obs.sum(), 1))
            control_covs = cov_valid[d_valid == 0]
            print(control_covs.shape)
            if control_covs.shape[0] > 0:
                gram = control_covs.T @ control_covs
                try:
                    cond = np.linalg.cond(gram)
                    if cond > 1.0 / np.finfo(float).eps:
                        reg_ill_conditioned = True
                        warnings.warn(
                            f"Not enough control units for group {g_val} in time period {t_val} "
                            "to run specified regression",
                            UserWarning,
                        )
                except np.linalg.LinAlgError:
                    reg_ill_conditioned = True
                    warnings.warn(
                        f"Singular covariate matrix for group {g_val} in time period {t_val}",
                        UserWarning,
                    )
        """
        if data.config.est_method in (EstimationMethod.DOUBLY_ROBUST, EstimationMethod.REGRESSION):
          cov_valid = covariates[valid_obs] if covariates.ndim > 1 else np.ones((valid_obs.sum(), 1))
          if cov_valid.ndim == 1:
              cov_valid = cov_valid.reshape(-1, 1)

          control_covs = cov_valid[d_valid == 0]
          print("control_covs original shape:", control_covs.shape)

          covariates_for_estimation = covariates

          if control_covs.shape[0] > 0:
              try:
                  _, keep_var, keep_qr = keep_lindep_cols_from_controls(
                      control_covs, tol=1e-10, keep_intercept=True
                  )

                  # IMPORTANT: reduce FULL covariates matrix, not just cov_valid
                  covariates_red = covariates[:, keep_var]
                  covariates_red = covariates_red[:, keep_qr]

                  print("covariates reduced full shape:", covariates_red.shape)

                  covariates_for_estimation = covariates_red

              except Exception as e:
                  reg_ill_conditioned = True
                  warnings.warn(
                      f"Could not reduce singular covariate matrix for group {g_val} in time period {t_val}: {e}",
                      UserWarning,
                  )
          try:

            try:
                import cupy as cp
                HAS_CUPY = True
            except Exception:
                cp = None
                HAS_CUPY = False


            def to_numpy_safe(x):
                if HAS_CUPY and isinstance(x, cp.ndarray):
                    return x.get()
                if isinstance(x, np.ndarray):
                    return x
                return x


            cohort_data_cpu = {k: to_numpy_safe(v) for k, v in cohort_data.items()}
            covariates_cpu = to_numpy_safe(covariates_for_estimation)

            return run_drdid(cohort_data_cpu, covariates_cpu, data)
              #return run_drdid(cohort_data, covariates_for_estimation, data)
          except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
              warnings.warn(
                  f"Error in computing 2x2 DiD for (g,t) = ({data.config.treated_groups[group_idx]},"
                  f"{data.config.time_periods[time_idx]}): {e}",
                  UserWarning,
              )
              return None


def get_did_cohort_index(
    group_idx,
    time_idx,
    time_factor,
    pre_treatment_idx,
    data,
):
    """Get cohort indices for current group-time pair.

    Parameters
    ----------
    group_idx : int
        Index of the treated group.
    time_idx : int
        Index of the time period.
    time_factor : int
        Time factor (1 for varying base period, 0 for universal).
    pre_treatment_idx : int
        Index of the pre-treatment period.
    data : DIDData
        Preprocessed DiD data object.

    Returns
    -------
    np.ndarray
        Array of 1s (treated), 0s (control), and NaNs indicating cohort membership.
    """
    if data.config.panel:
        # Determine control group boundaries
        if data.config.control_group == ControlGroup.NOT_YET_TREATED:
            # Find first cohort treated after the relevant period
            relevant_period = (
                data.config.time_periods[max(time_idx, pre_treatment_idx) + time_factor] + data.config.anticipation
            )
            future_cohorts = data.cohort_counts.filter(data.cohort_counts["cohort"] > relevant_period)
            min_control = future_cohorts["cohort"][0] if len(future_cohorts) > 0 else np.inf
        else:  # nevertreated
            min_control = np.inf

        max_control = np.inf

        n_units = len(data.time_invariant_data) if data.config.allow_unbalanced_panel else data.config.id_count
        cohort_index = np.full(n_units, np.nan)

        cohort_values = data.cohort_counts["cohort"].to_numpy()
        if max_control not in cohort_values:
            max_control = cohort_values[-1]

        # Control group indices - this includes all matching cohorts
        control_mask = (data.cohort_counts["cohort"] >= min_control) & (data.cohort_counts["cohort"] <= max_control)
        control_mask_np = control_mask.to_numpy()
        cohort_sizes = data.cohort_counts["cohort_size"].to_numpy()
        for idx in np.where(control_mask_np)[0]:
            start_control = int(cohort_sizes[:idx].sum()) if idx > 0 else 0
            end_control = int(cohort_sizes[: idx + 1].sum())
            cohort_index[start_control:end_control] = 0

        # Treated group indices
        treated_mask = data.cohort_counts["cohort"] == data.config.treated_groups[group_idx]
        treated_mask_np = treated_mask.to_numpy()
        if treated_mask_np.any():
            treat_idx = int(np.argmax(treated_mask_np))
            cohort_sizes = data.cohort_counts["cohort_size"].to_numpy()
            start_treat = int(cohort_sizes[:treat_idx].sum()) if treat_idx > 0 else 0
            end_treat = int(cohort_sizes[: treat_idx + 1].sum())
            cohort_index[start_treat:end_treat] = 1

    else:
        n_units = len(data.data)
        cohort_index = np.full(n_units, np.nan)

        treated_flag = (data.data[data.config.gname] == data.config.treated_groups[group_idx]).to_numpy()

        if data.config.control_group == ControlGroup.NEVER_TREATED:
            control_flag = (data.data[data.config.gname] == np.inf).to_numpy()
        else:  # NOT_YET_TREATED
            relevant_period = (
                data.config.time_periods[max(time_idx, pre_treatment_idx) + time_factor] + data.config.anticipation
            )
            control_flag = (
                (data.data[data.config.gname] == np.inf)
                | (
                    (data.data[data.config.gname] > relevant_period)
                    & (data.data[data.config.gname] != data.config.treated_groups[group_idx])
                )
            ).to_numpy()

        keep_periods = (
            data.data[data.config.tname]
            .is_in([data.config.time_periods[time_idx + time_factor], data.config.time_periods[pre_treatment_idx]])
            .to_numpy()
        )

        cohort_index[keep_periods & control_flag] = 0
        cohort_index[keep_periods & treated_flag] = 1

    return cohort_index


def run_drdid(
    cohort_data,
    covariates,
    data,
):
    """Run DR-DiD estimation for current group-time pair.

    Parameters
    ----------
    cohort_data : dict
        Dictionary containing outcome and treatment data for the cohort.
    covariates : ndarray
        Covariate matrix for the estimation.
    data : DIDData
        Preprocessed DiD data object.

    Returns
    -------
    dict
        Dictionary with ATT estimate and influence function.
    """
    n = len(cohort_data["D"])
    est_method = data.config.est_method
    valid_obs = ~np.isnan(cohort_data["D"])

    if valid_obs.sum() == 0:
        return {"att": np.nan, "inf_func": np.full(n, np.nan)}

    if data.config.panel:
        y1 = cohort_data["y1"][valid_obs]
        y0 = cohort_data["y0"][valid_obs]
        d = cohort_data["D"][valid_obs]
        weights = cohort_data["weights"][valid_obs]

        if covariates.ndim > 1:
            cov_valid = covariates[valid_obs]
        else:
            cov_valid = covariates[valid_obs] if len(covariates) > 1 else np.ones(valid_obs.sum())

        if callable(est_method):
            result = est_method(y1=y1, y0=y0, d=d, covariates=cov_valid, i_weights=weights, influence_func=True)
        elif est_method == EstimationMethod.IPW:
            result = std_ipw_did_panel(
                y1=y1, y0=y0, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        elif est_method == EstimationMethod.REGRESSION:
            result = reg_did_panel(
                y1=y1, y0=y0, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        else:  # DOUBLY_ROBUST (default)
            result = drdid_panel(
                y1=y1, y0=y0, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )

        influence_func = np.zeros(n)
        influence_func[valid_obs] = (n / valid_obs.sum()) * result.att_inf_func

    else:
        y = cohort_data["y"][valid_obs]
        post = cohort_data["post"][valid_obs]
        d = cohort_data["D"][valid_obs]
        weights = cohort_data["weights"][valid_obs]

        if covariates.ndim > 1:
            cov_valid = covariates[valid_obs]
        else:
            cov_valid = covariates if len(covariates) == n else covariates[valid_obs]

        if callable(est_method):
            result = est_method(y=y, post=post, d=d, covariates=cov_valid, i_weights=weights, influence_func=True)
        elif est_method == EstimationMethod.IPW:
            result = std_ipw_did_rc(
                y=y, post=post, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        elif est_method == EstimationMethod.REGRESSION:
            result = reg_did_rc(
                y=y, post=post, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        else:  # DOUBLY_ROBUST (default)
            result = drdid_rc(
                y=y, post=post, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )

        # Handle influence function for unbalanced panel
        if data.config.allow_unbalanced_panel and "rowid" in cohort_data:
            inf_func_long = np.zeros(n)
            inf_func_long[valid_obs] = (data.config.id_count / valid_obs.sum()) * result.att_inf_func

            unique_ids = np.unique(cohort_data["rowid"])
            influence_func = np.zeros(len(unique_ids))
            for i, uid in enumerate(unique_ids):
                mask = cohort_data["rowid"] == uid
                influence_func[i] = inf_func_long[mask].sum()
        else:
            influence_func = np.zeros(n)
            influence_func[valid_obs] = (n / valid_obs.sum()) * result.att_inf_func

    if np.isnan(result.att):
        return {"att": np.nan, "inf_func": np.full(n, np.nan)}

    return {"att": result.att, "inf_func": influence_func}


def _process_gt_cell_did(group_idx, time_idx, data):
    """Process a single (group, time) cell for DiD estimation.

    Returns
    -------
    tuple of (ATTgtResult or None, ndarray or None)
    """
    n_units = data.config.id_count
    time_factor = 1 if data.config.base_period != "universal" else 0

    estimation_result = run_att_gt_estimation(group_idx, time_idx, data)

    is_post_treatment = int(data.config.treated_groups[group_idx] <= data.config.time_periods[time_idx + time_factor])

    if estimation_result is None or estimation_result["att"] is None:
        if data.config.base_period == "universal":
            pre_periods = np.where(
                data.config.time_periods < (data.config.treated_groups[group_idx] - data.config.anticipation)
            )[0]
            is_reference = (
                len(pre_periods) > 0
                and data.config.time_periods[pre_periods[-1]] == data.config.time_periods[time_idx + time_factor]
            )
            att_val = 0.0 if is_reference else np.nan
            inf_func = np.zeros(n_units) if is_reference else np.full(n_units, np.nan)
            return (
                ATTgtResult(
                    att=att_val,
                    group=data.config.treated_groups[group_idx],
                    year=data.config.time_periods[time_idx + time_factor],
                    post=is_post_treatment,
                ),
                inf_func,
            )
        return (None, None)

    att_estimate = estimation_result["att"]
    influence_func = estimation_result["inf_func"]

    if np.isnan(att_estimate):
        influence_func = np.full(n_units, np.nan)

    return (
        ATTgtResult(
            att=att_estimate,
            group=data.config.treated_groups[group_idx],
            year=data.config.time_periods[time_idx + time_factor],
            post=is_post_treatment,
        ),
        influence_func,
    )

'''

path.write_text(new_source, encoding="utf-8")
importlib.invalidate_caches()


print(f"Overwrote {path}")



## Overwrite the estimator function

path = Path("/usr/local/lib/python3.12/dist-packages/moderndid/drdid/estimators/reg_did_rc.py")

new_source = r'''
"""Outcome regression DiD estimator for repeated cross-sections data."""

from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from scipy import stats

from moderndid.cupy.backend import get_backend, to_numpy
from moderndid.cupy.regression import cupy_wls

from ..bootstrap.boot_mult import mboot_did
from ..bootstrap.boot_reg_rc import wboot_reg_rc


class RegDIDRCResult(NamedTuple):
    """Result from the regression DiD RC estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    args: dict


def reg_did_rc(
    y,
    post,
    d,
    covariates=None,
    i_weights=None,
    boot=False,
    boot_type="weighted",
    nboot=999,
    influence_func=False,
):
    r"""Compute the outcome regression DiD estimator for the ATT with repeated cross-section data.

    Implements outcome regression difference-in-differences (DiD) estimator for the ATT when stationary
    repeated cross-sectional data are available. The estimator is a sample analogue of equation (2.2)
    in [2]_. The estimator is given by

    .. math::
        \widehat{\tau}^{reg} = \bar{Y}_{1,1} - \left[\bar{Y}_{1,0} + n_{treat}^{-1}
        \sum_{i|D_i=1} (\widehat{\mu}_{0,1}(X_i) - \widehat{\mu}_{0,0}(X_i))\right].

    The estimator follows the same spirit of the nonparametric estimators proposed by [1]_, though here the
    outcome regression models are assumed to be linear in covariates (parametric). The nuisance parameters (outcome
    regression coefficients) are estimated via ordinary least squares.

    Parameters
    ----------
    y : ndarray
        A 1D array of outcomes from both pre- and post-treatment periods.
    post : ndarray
        A 1D array of post-treatment dummies (1 if post-treatment, 0 if pre-treatment).
    d : ndarray
        A 1D array of group indicators (1 if treated in post-treatment, 0 otherwise).
    covariates : ndarray, optional
        A 2D array of covariates to be used in the regression estimation. If None,
        this leads to an unconditional DiD estimator.
    i_weights : ndarray, optional
        A 1D array of weights. If None, then every observation has equal weight.
        Weights are normalized to have mean 1.
    boot : bool, optional
        Whether to use bootstrap for inference. Default is False.
    boot_type : str, optional
        Type of bootstrap ("weighted" or "multiplier"). Default is "weighted".
    nboot : int, optional
        Number of bootstrap repetitions. Default is 999.
    influence_func : bool, optional
        Whether to return the influence function. Default is False.

    Returns
    -------
    RegDIDRCResult
        A named tuple containing:

        - att : float
            The outcome regression DiD point estimate.
        - se : float
            The outcome regression DiD standard error.
        - uci : float
            Upper bound of a 95% confidence interval.
        - lci : float
            Lower bound of a 95% confidence interval.
        - boots : ndarray or None
            Bootstrap draws of the ATT if boot=True.
        - att_inf_func : ndarray or None
            Influence function values if influence_func=True.
        - args : dict
            Arguments used in the estimation.

    See Also
    --------
    ipw_did_rc : Inverse propensity weighted DiD for repeated cross-sections.
    drdid_rc : Doubly robust DiD for repeated cross-sections.

    References
    ----------

    .. [1] Heckman, J., Ichimura, H., and Todd, P. (1997), "Matching as an Econometric Evaluation
           Estimator: Evidence from Evaluating a Job Training Programme", Review of Economic Studies,
           vol. 64(4), p. 605–654. https://doi.org/10.2307/2971733

    .. [2] Sant'Anna, P. H. C. and Zhao, J. (2020), "Doubly Robust Difference-in-Differences Estimators."
           Journal of Econometrics, Vol. 219 (1), pp. 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
    """
    y, post, d, int_cov, i_weights, n_units = _validate_and_preprocess_inputs(y, post, d, covariates, i_weights)

    out_y_pre, out_y_post = _fit_outcome_regressions(y, post, d, int_cov, i_weights)
    weights = _compute_weights(d, post, i_weights)

    reg_att_treat_pre = weights["w_treat_pre"] * y
    reg_att_treat_post = weights["w_treat_post"] * y
    reg_att_cont = weights["w_cont"] * (out_y_post - out_y_pre)

    eta_treat_pre = np.mean(reg_att_treat_pre) / np.mean(weights["w_treat_pre"])
    eta_treat_post = np.mean(reg_att_treat_post) / np.mean(weights["w_treat_post"])
    eta_cont = np.mean(reg_att_cont) / np.mean(weights["w_cont"])

    reg_att = (eta_treat_post - eta_treat_pre) - eta_cont

    influence_quantities = _get_influence_quantities(y, post, d, int_cov, out_y_pre, out_y_post, i_weights, n_units)

    reg_att_inf_func = _compute_influence_function(
        reg_att_treat_pre,
        reg_att_treat_post,
        reg_att_cont,
        eta_treat_pre,
        eta_treat_post,
        eta_cont,
        weights,
        int_cov,
        influence_quantities,
    )

    # Inference
    if not boot:
        se_reg_att = np.std(reg_att_inf_func, ddof=1) * np.sqrt(n_units - 1) / n_units
        uci = reg_att + 1.96 * se_reg_att
        lci = reg_att - 1.96 * se_reg_att
        reg_boot = None
    else:
        if boot_type == "multiplier":
            reg_boot = mboot_did(reg_att_inf_func, nboot)
            se_reg_att = stats.iqr(reg_boot) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.quantile(np.abs(reg_boot / se_reg_att), 0.95)
            uci = reg_att + cv * se_reg_att
            lci = reg_att - cv * se_reg_att
        else:  # "weighted"
            reg_boot = wboot_reg_rc(y, post, d, int_cov, i_weights, n_bootstrap=nboot)
            se_reg_att = stats.iqr(reg_boot - reg_att) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            cv = np.quantile(np.abs((reg_boot - reg_att) / se_reg_att), 0.95)
            uci = reg_att + cv * se_reg_att
            lci = reg_att - cv * se_reg_att

    if not influence_func:
        reg_att_inf_func = None

    args = {
        "panel": False,
        "boot": boot,
        "boot_type": boot_type,
        "nboot": nboot,
        "type": "or",
    }

    return RegDIDRCResult(
        att=reg_att,
        se=se_reg_att,
        uci=uci,
        lci=lci,
        boots=reg_boot,
        att_inf_func=reg_att_inf_func,
        args=args,
    )


def _validate_and_preprocess_inputs(y, post, d, covariates, i_weights):
    d = np.asarray(d).flatten()
    post = np.asarray(post).flatten()
    y = np.asarray(y).flatten()

    n_units = len(d)

    if covariates is None:
        int_cov = np.ones((n_units, 1))
    else:
        int_cov = np.asarray(covariates)
        if int_cov.ndim == 1:
            int_cov = int_cov.reshape(-1, 1)

    if i_weights is None:
        i_weights = np.ones(n_units)
    else:
        i_weights = np.asarray(i_weights).flatten()
        if np.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")
    i_weights = i_weights / np.mean(i_weights)

    return y, post, d, int_cov, i_weights, n_units

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _screen_bad_columns_before_fit(
    X,
    y,
    min_active=20,
    zero_outcome_only=True,
    near_constant_tol=1e-12,
):
    """
    Identify columns that are likely problematic before fitting Poisson QMLE.

    Flags:
      - all-zero columns
      - sparse columns
      - columns active only where y sums to zero
      - near-constant columns other than intercept-like columns
    """
    bad_cols = set()
    rows = []

    n = X.shape[0]

    for j in range(X.shape[1]):
        xj = X[:, j]
        active = xj != 0
        n_active = active.sum()

        if n_active > 0:
            y_active = y[active]
            y_sum_active = y_active.sum()
            y_mean_active = y_active.mean()
            share_zero_active = np.mean(y_active == 0)
        else:
            y_sum_active = np.nan
            y_mean_active = np.nan
            share_zero_active = np.nan

        is_all_zero = n_active == 0
        is_sparse = (0 < n_active < min_active)
        is_only_zero_outcomes = (
            zero_outcome_only and n_active > 0 and y_sum_active == 0
        )

        # Keep intercept-like columns. Usually intercept has all ones.
        is_intercept_like = np.allclose(xj, xj[0])

        is_near_constant = (
            not is_intercept_like and
            np.nanstd(xj) < near_constant_tol
        )

        is_bad = (
            is_all_zero or
            is_sparse or
            is_only_zero_outcomes or
            is_near_constant
        )

        if is_bad:
            bad_cols.add(j)

        rows.append({
            "column": j,
            "n_active": n_active,
            "share_active": n_active / n,
            "y_sum_when_active": y_sum_active,
            "y_mean_when_active": y_mean_active,
            "share_zero_when_active": share_zero_active,
            "all_zero": is_all_zero,
            "sparse": is_sparse,
            "only_zero_outcomes": is_only_zero_outcomes,
            "near_constant": is_near_constant,
            "bad_before_fit": is_bad,
        })

    diag = pd.DataFrame(rows)
    return bad_cols, diag


def _fit_poisson_qmle(y_train, X_train, weights_train, maxiter=100):
    """Fit Poisson QMLE using statsmodels GLM."""
    glm = sm.GLM(
        y_train,
        X_train,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        var_weights=weights_train,
    )

    res = glm.fit(maxiter=maxiter)

    return res


def _screen_bad_columns_after_fit(
    res,
    X,
    y,
    coef_cutoff=10,
    se_cutoff=100,
):
    """
    Identify columns that look numerically unstable after a first-pass fit.
    """
    bad_cols = set()
    rows = []

    params = np.asarray(res.params)
    bse = np.asarray(res.bse)

    for j in range(X.shape[1]):
        xj = X[:, j]
        active = xj != 0
        n_active = active.sum()

        if n_active > 0:
            y_active = y[active]
            y_sum_active = y_active.sum()
            y_mean_active = y_active.mean()
            share_zero_active = np.mean(y_active == 0)
        else:
            y_sum_active = np.nan
            y_mean_active = np.nan
            share_zero_active = np.nan

        huge_coef = np.abs(params[j]) > coef_cutoff
        huge_se = bse[j] > se_cutoff
        nonfinite = (not np.isfinite(params[j])) or (not np.isfinite(bse[j]))

        is_bad = huge_coef or huge_se or nonfinite

        if is_bad:
            bad_cols.add(j)

        rows.append({
            "column": j,
            "coef": params[j],
            "se": bse[j],
            "n_active": n_active,
            "y_sum_when_active": y_sum_active,
            "y_mean_when_active": y_mean_active,
            "share_zero_when_active": share_zero_active,
            "huge_coef": huge_coef,
            "huge_se": huge_se,
            "nonfinite": nonfinite,
            "bad_after_fit": is_bad,
        })

    diag = pd.DataFrame(rows)
    return bad_cols, diag


def _summarize_predictions(name, mu, y=None, mask=None):
    """Print compact prediction diagnostics."""
    if mask is not None:
        mu_use = mu[mask]
        y_use = y[mask] if y is not None else None
    else:
        mu_use = mu
        y_use = y

    print(f"\n{name} predicted mean summary")
    print(pd.Series(mu_use).describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    print("nan:", np.isnan(mu_use).any())
    print("inf:", np.isinf(mu_use).any())
    print("min:", np.min(mu_use))
    print("max:", np.max(mu_use))

    if y_use is not None:
        print(f"\n{name} observed y summary")
        print(pd.Series(y_use).describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))


def _fit_outcome_regressions(y, post, d, int_cov, i_weights):
    """
    Fit Poisson QMLE outcome regression models for control units
    in pre and post periods.

    This version:
      - uses CPU/statsmodels only
      - expects raw nonnegative citation counts, not log outcomes
      - screens sparse/separated columns before fitting
      - does first-pass fit
      - screens huge coefficients / huge SEs after fitting
      - refits final pre/post models on cleaned covariates
      - returns predicted counterfactual means for all observations
    """

    y = np.asarray(y)
    post = np.asarray(post)
    d = np.asarray(d)
    int_cov = np.asarray(int_cov)
    i_weights = np.asarray(i_weights)

    if np.any(y < 0):
        raise ValueError(
            "Poisson QMLE requires raw nonnegative count outcomes. "
            "Use raw citation counts, not log_citation."
        )

    if np.any(~np.isfinite(y)):
        raise ValueError("Outcome y contains non-finite values.")

    if np.any(~np.isfinite(int_cov)):
        raise ValueError("int_cov contains non-finite values.")

    if np.any(~np.isfinite(i_weights)):
        raise ValueError("i_weights contains non-finite values.")

    # --------------------------------------------------
    # Define pre/post control samples
    # --------------------------------------------------
    pre_filter = (d == 0) & (post == 0)
    post_filter = (d == 0) & (post == 1)

    n_control_pre = np.sum(pre_filter)
    n_control_post = np.sum(post_filter)

    if n_control_pre == 0:
        raise ValueError("No control units in pre-treatment period.")

    if n_control_post == 0:
        raise ValueError("No control units in post-treatment period.")

    X_pre = int_cov[pre_filter]
    y_pre = y[pre_filter]
    w_pre = i_weights[pre_filter]

    X_post = int_cov[post_filter]
    y_post = y[post_filter]
    w_post = i_weights[post_filter]

    print("\nPoisson QMLE outcome regression")
    print("Full X shape:", int_cov.shape)
    print("Pre control X shape:", X_pre.shape)
    print("Post control X shape:", X_post.shape)
    print("Pre y mean:", y_pre.mean(), "share zero:", np.mean(y_pre == 0), "max:", y_pre.max())
    print("Post y mean:", y_post.mean(), "share zero:", np.mean(y_post == 0), "max:", y_post.max())

    # --------------------------------------------------
    # 1. Pre-fit screening
    # --------------------------------------------------
    bad_pre_before, pre_diag_before = _screen_bad_columns_before_fit(X_pre, y_pre)
    bad_post_before, post_diag_before = _screen_bad_columns_before_fit(X_post, y_post)

    bad_before = bad_pre_before.union(bad_post_before)

    print("\nBad columns before fit")
    print("pre:", len(bad_pre_before))
    print("post:", len(bad_post_before))
    print("union:", len(bad_before))

    keep_cols_0 = np.array(
        [j for j in range(int_cov.shape[1]) if j not in bad_before],
        dtype=int
    )

    if len(keep_cols_0) == 0:
        raise ValueError("All columns were flagged before fitting Poisson QMLE.")

    X_pre_0 = X_pre[:, keep_cols_0]
    X_post_0 = X_post[:, keep_cols_0]

    if X_pre_0.shape[0] < X_pre_0.shape[1]:
        raise ValueError(
            f"Insufficient pre-period controls after screening: "
            f"n={X_pre_0.shape[0]}, k={X_pre_0.shape[1]}"
        )

    if X_post_0.shape[0] < X_post_0.shape[1]:
        raise ValueError(
            f"Insufficient post-period controls after screening: "
            f"n={X_post_0.shape[0]}, k={X_post_0.shape[1]}"
        )

    # --------------------------------------------------
    # 2. First-pass Poisson fits
    # --------------------------------------------------
    try:
        res_pre_0 = _fit_poisson_qmle(y_pre, X_pre_0, w_pre)
    except Exception as e:
        raise ValueError(f"First-pass pre-period Poisson QMLE failed: {e}") from e

    try:
        res_post_0 = _fit_poisson_qmle(y_post, X_post_0, w_post)
    except Exception as e:
        raise ValueError(f"First-pass post-period Poisson QMLE failed: {e}") from e

    # --------------------------------------------------
    # 3. Post-fit screening for huge coef / SE
    # --------------------------------------------------
    bad_pre_after_local, pre_diag_after = _screen_bad_columns_after_fit(
        res_pre_0, X_pre_0, y_pre
    )
    bad_post_after_local, post_diag_after = _screen_bad_columns_after_fit(
        res_post_0, X_post_0, y_post
    )

    # Convert local indices in screened X back to original int_cov columns
    bad_pre_after = {keep_cols_0[j] for j in bad_pre_after_local}
    bad_post_after = {keep_cols_0[j] for j in bad_post_after_local}

    bad_after = bad_pre_after.union(bad_post_after)
    bad_all = bad_before.union(bad_after)

    print("\nBad columns after first-pass fit")
    print("pre:", len(bad_pre_after))
    print("post:", len(bad_post_after))
    print("union after:", len(bad_after))
    print("total bad columns:", len(bad_all))

    keep_cols = np.array(
        [j for j in range(int_cov.shape[1]) if j not in bad_all],
        dtype=int
    )

    if len(keep_cols) == 0:
        raise ValueError("All columns were flagged as problematic in Poisson QMLE.")

    X_pre_keep = X_pre[:, keep_cols]
    X_post_keep = X_post[:, keep_cols]
    X_all_keep = int_cov[:, keep_cols]

    print("Final X shape after dropping bad columns:", X_all_keep.shape)

    # --------------------------------------------------
    # 4. Final refits
    # --------------------------------------------------
    try:
        res_pre = _fit_poisson_qmle(y_pre, X_pre_keep, w_pre)
    except Exception as e:
        raise ValueError(f"Final pre-period Poisson QMLE failed: {e}") from e

    try:
        res_post = _fit_poisson_qmle(y_post, X_post_keep, w_post)
        print(res_post.summary())
    except Exception as e:
        raise ValueError(f"Final post-period Poisson QMLE failed: {e}") from e

    if np.any(~np.isfinite(res_pre.params)):
        raise ValueError("Final pre-period Poisson QMLE produced non-finite coefficients.")

    if np.any(~np.isfinite(res_post.params)):
        raise ValueError("Final post-period Poisson QMLE produced non-finite coefficients.")

    # --------------------------------------------------
    # 5. Predict counterfactual means for all observations
    # --------------------------------------------------
    out_y_pre = res_pre.predict(X_all_keep)
    out_y_post = res_post.predict(X_all_keep)

    if np.any(~np.isfinite(out_y_pre)):
        raise ValueError("Pre-period Poisson QMLE produced non-finite predictions.")

    if np.any(~np.isfinite(out_y_post)):
        raise ValueError("Post-period Poisson QMLE produced non-finite predictions.")

    # Optional but useful diagnostics
    treated = d == 1

    _summarize_predictions("Pre all", out_y_pre, y=y)
    _summarize_predictions("Post all", out_y_post, y=y)
    _summarize_predictions("Pre treated", out_y_pre, y=y, mask=treated)
    _summarize_predictions("Post treated", out_y_post, y=y, mask=treated)

    # If you want to inspect later, attach diagnostics to function object.
    # This is a convenient hack if the larger package only expects two returns.
    _fit_outcome_regressions.keep_cols_ = keep_cols
    _fit_outcome_regressions.bad_cols_ = sorted(bad_all)
    _fit_outcome_regressions.pre_diag_before_ = pre_diag_before
    _fit_outcome_regressions.post_diag_before_ = post_diag_before
    _fit_outcome_regressions.pre_diag_after_ = pre_diag_after
    _fit_outcome_regressions.post_diag_after_ = post_diag_after
    _fit_outcome_regressions.res_pre_ = res_pre
    _fit_outcome_regressions.res_post_ = res_post

    return out_y_pre, out_y_post


def _compute_weights(d, post, i_weights):
    """Compute weights for outcome regression DiD estimator."""
    w_treat_pre = i_weights * d * (1 - post)
    w_treat_post = i_weights * d * post
    w_cont = i_weights * d

    return {
        "w_treat_pre": w_treat_pre,
        "w_treat_post": w_treat_post,
        "w_cont": w_cont,
    }

"""
def _get_influence_quantities(y, post, d, int_cov, out_y_pre, out_y_post, i_weights, n_units):
    """
    Compute quantities needed for the influence function when the outcome
    regression is estimated by Poisson QMLE.

    For Poisson QMLE:
        mu_i = E[Y_i | X_i] = exp(X_i' beta)
        score_i = w_i X_i (Y_i - mu_i)
        H = E[w_i mu_i X_i X_i']

    The asymptotic linear representation of beta_hat is:
        H^{-1} score_i
    """

    # -----------------------------
    # Pre-treatment outcome model
    # -----------------------------
    weights_pre = i_weights * (1 - d) * (1 - post)

    mu_pre = out_y_pre
    resid_pre = y - mu_pre

    # Hessian / bread for Poisson QMLE:
    # E[w * mu * X X']
    weighted_mu_x_pre = (weights_pre * mu_pre)[:, np.newaxis] * int_cov
    hessian_pre = weighted_mu_x_pre.T @ int_cov / n_units

    if np.linalg.cond(hessian_pre) > 1e15:
        raise ValueError(
            "The Poisson QMLE Hessian for pre-treatment is singular. "
            "Consider removing sparse or collinear covariates."
        )

    hessian_inv_pre = np.linalg.inv(hessian_pre)

    # Score contribution:
    # w * X * (Y - mu)
    score_pre = weights_pre[:, np.newaxis] * resid_pre[:, np.newaxis] * int_cov

    # Influence representation for beta_pre_hat
    asy_lin_rep_pois_pre = score_pre @ hessian_inv_pre

    # -----------------------------
    # Post-treatment outcome model
    # -----------------------------
    weights_post = i_weights * (1 - d) * post

    mu_post = out_y_post
    resid_post = y - mu_post

    weighted_mu_x_post = (weights_post * mu_post)[:, np.newaxis] * int_cov
    hessian_post = weighted_mu_x_post.T @ int_cov / n_units

    if np.linalg.cond(hessian_post) > 1e15:
        raise ValueError(
            "The Poisson QMLE Hessian for post-treatment is singular. "
            "Consider removing sparse or collinear covariates."
        )

    hessian_inv_post = np.linalg.inv(hessian_post)

    score_post = weights_post[:, np.newaxis] * resid_post[:, np.newaxis] * int_cov

    asy_lin_rep_pois_post = score_post @ hessian_inv_post

    return {
        "asy_lin_rep_pois_pre": asy_lin_rep_pois_pre,
        "asy_lin_rep_pois_post": asy_lin_rep_pois_post,
        "mu_pre": mu_pre,
        "mu_post": mu_post,
    }
"""

import numpy as np
from scipy.linalg import qr as scipy_qr


def _keep_lindep_cols(X, tol=1e-10, keep_intercept=True):
    """
    Keep columns that:
    1) vary in the estimation sample (or are intercept-like if keep_intercept=True)
    2) are linearly independent by pivoted QR

    Returns
    -------
    X_reduced : ndarray
    keep_var : boolean mask over original columns
    keep_qr : integer indices over columns after keep_var
    kept_cols : integer indices over original columns
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    is_const_one = np.array([
        np.all(np.isfinite(col)) and np.max(np.abs(col - 1.0)) < tol
        for col in X.T
    ])

    is_nonconst = np.array([
        np.var(col[np.isfinite(col)]) > tol if np.isfinite(col).sum() > 1 else False
        for col in X.T
    ])

    keep_var = is_nonconst | (keep_intercept & is_const_one)

    if not np.any(keep_var):
        raise ValueError("No covariate columns with variation remain in this subsample.")

    X1 = X[:, keep_var]

    q, r, piv = scipy_qr(X1, mode="economic", pivoting=True)
    diag_r = np.abs(np.diag(r))
    rank = int(np.sum(diag_r > tol))
    keep_qr = np.sort(piv[:rank])

    if keep_qr.size == 0:
        raise ValueError("No linearly independent covariate columns remain in this subsample.")

    kept_cols = np.where(keep_var)[0][keep_qr]
    X2 = X[:, kept_cols]

    return X2, keep_var, keep_qr, kept_cols


def _get_influence_quantities(y, post, d, int_cov, out_y_pre, out_y_post, i_weights, n_units, tol=1e-10):
    """Compute quantities needed for influence function using linearly independent columns."""

    y = np.asarray(y).ravel()
    post = np.asarray(post).ravel()
    d = np.asarray(d).ravel()
    int_cov = np.asarray(int_cov)
    out_y_pre = np.asarray(out_y_pre).ravel()
    out_y_post = np.asarray(out_y_post).ravel()
    i_weights = np.asarray(i_weights).ravel()

    if int_cov.ndim == 1:
        int_cov = int_cov.reshape(-1, 1)

    # -------- PRE --------
    weights_ols_pre = i_weights * (1 - d) * (1 - post)
    pre_mask = weights_ols_pre > 0

    X_pre_controls = int_cov[pre_mask, :]
    X_pre_red_controls, keep_var_pre, keep_qr_pre, kept_cols_pre = _keep_lindep_cols(
        X_pre_controls, tol=tol, keep_intercept=True
    )

    int_cov_pre = int_cov[:, kept_cols_pre]

    weighted_x_pre = weights_ols_pre[:, np.newaxis] * int_cov_pre
    weighted_resid_x_pre = (
        weights_ols_pre[:, np.newaxis]
        * (y - out_y_pre)[:, np.newaxis]
        * int_cov_pre
    )
    gram_pre = weighted_x_pre.T @ int_cov_pre / n_units

    if np.linalg.cond(gram_pre) > 1e15:
        raise ValueError(
            "The regression design matrix for pre-treatment is still singular/ill-conditioned after dropping dependent columns."
        )

    gram_inv_pre = np.linalg.inv(gram_pre)
    asy_lin_rep_ols_pre = weighted_resid_x_pre @ gram_inv_pre

    # -------- POST --------
    weights_ols_post = i_weights * (1 - d) * post
    post_mask = weights_ols_post > 0

    X_post_controls = int_cov[post_mask, :]
    X_post_red_controls, keep_var_post, keep_qr_post, kept_cols_post = _keep_lindep_cols(
        X_post_controls, tol=tol, keep_intercept=True
    )

    int_cov_post = int_cov[:, kept_cols_post]

    weighted_x_post = weights_ols_post[:, np.newaxis] * int_cov_post
    weighted_resid_x_post = (
        weights_ols_post[:, np.newaxis]
        * (y - out_y_post)[:, np.newaxis]
        * int_cov_post
    )
    gram_post = weighted_x_post.T @ int_cov_post / n_units

    if np.linalg.cond(gram_post) > 1e15:
        raise ValueError(
            "The regression design matrix for post-treatment is still singular/ill-conditioned after dropping dependent columns."
        )

    gram_inv_post = np.linalg.inv(gram_post)
    asy_lin_rep_ols_post = weighted_resid_x_post @ gram_inv_post

    return {
        "asy_lin_rep_ols_pre": asy_lin_rep_ols_pre,
        "asy_lin_rep_ols_post": asy_lin_rep_ols_post,
        "int_cov_pre": int_cov_pre,
        "int_cov_post": int_cov_post,
        "kept_cols_pre": kept_cols_pre,
        "kept_cols_post": kept_cols_post,
    }
def _compute_influence_function(
    reg_att_treat_pre,
    reg_att_treat_post,
    reg_att_cont,
    eta_treat_pre,
    eta_treat_post,
    eta_cont,
    weights,
    int_cov,
    influence_quantities,
):
    """
    Compute the influence function for the OR estimator when the outcome
    regression is estimated by Poisson QMLE.
    """

    w_treat_pre = weights["w_treat_pre"]
    w_treat_post = weights["w_treat_post"]
    w_cont = weights["w_cont"]

    asy_lin_rep_pois_pre = influence_quantities["asy_lin_rep_pois_pre"]
    asy_lin_rep_pois_post = influence_quantities["asy_lin_rep_pois_post"]

    mu_pre = influence_quantities["mu_pre"]
    mu_post = influence_quantities["mu_post"]

    # --------------------------------------------------
    # Influence function of the treated component
    # --------------------------------------------------
    inf_treat_pre = (
        reg_att_treat_pre - w_treat_pre * eta_treat_pre
    ) / np.mean(w_treat_pre)

    inf_treat_post = (
        reg_att_treat_post - w_treat_post * eta_treat_post
    ) / np.mean(w_treat_post)

    inf_treat = inf_treat_post - inf_treat_pre

    # --------------------------------------------------
    # Influence function of the control/counterfactual component
    # --------------------------------------------------
    # Leading term: same as before
    inf_cont_1 = reg_att_cont - w_cont * eta_cont

    # Estimation effect from beta_hat under Poisson QMLE
    #
    # Linear OR:
    #     d m(X; beta) / d beta = X
    #
    # Poisson OR:
    #     m(X; beta) = exp(X'beta)
    #     d m(X; beta) / d beta = mu(X) X
    #
    # Therefore the derivative differs for pre and post because mu_pre != mu_post.
    control_pois_derivative_post = np.mean(
        w_cont[:, np.newaxis] * mu_post[:, np.newaxis] * int_cov,
        axis=0,
    )

    control_pois_derivative_pre = np.mean(
        w_cont[:, np.newaxis] * mu_pre[:, np.newaxis] * int_cov,
        axis=0,
    )

    inf_cont_2_post = asy_lin_rep_pois_post @ control_pois_derivative_post
    inf_cont_2_pre = asy_lin_rep_pois_pre @ control_pois_derivative_pre

    inf_control = (
        inf_cont_1 + inf_cont_2_post - inf_cont_2_pre
    ) / np.mean(w_cont)

    reg_att_inf_func = inf_treat - inf_control

    return reg_att_inf_func

    """
def _compute_influence_function(
    reg_att_treat_pre,
    reg_att_treat_post,
    reg_att_cont,
    eta_treat_pre,
    eta_treat_post,
    eta_cont,
    weights,
    int_cov,
    influence_quantities,
):
    w_treat_pre = weights["w_treat_pre"]
    w_treat_post = weights["w_treat_post"]
    w_cont = weights["w_cont"]

    asy_lin_rep_ols_pre = influence_quantities["asy_lin_rep_ols_pre"]
    asy_lin_rep_ols_post = influence_quantities["asy_lin_rep_ols_post"]

    # Influence function of the "treat" component
    # Leading term of the influence function
    inf_treat_pre = (reg_att_treat_pre - w_treat_pre * eta_treat_pre) / np.mean(w_treat_pre)
    inf_treat_post = (reg_att_treat_post - w_treat_post * eta_treat_post) / np.mean(w_treat_post)
    inf_treat = inf_treat_post - inf_treat_pre

    # Influence function of control component
    # Leading term of the influence function: no estimation effect
    inf_cont_1 = reg_att_cont - w_cont * eta_cont
    # Estimation effect from beta hat (OLS using only controls)
    # Derivative matrix (k x 1 vector)
    control_ols_derivative = np.mean(w_cont[:, np.newaxis] * int_cov, axis=0)
    # Now get the influence function related to the estimation effect related to beta's in post-treatment
    inf_cont_2_post = asy_lin_rep_ols_post @ control_ols_derivative
    # Now get the influence function related to the estimation effect related to beta's in pre-treatment
    inf_cont_2_pre = asy_lin_rep_ols_pre @ control_ols_derivative
    # Influence function for the control component
    inf_control = (inf_cont_1 + inf_cont_2_post - inf_cont_2_pre) / np.mean(w_cont)

    # Get the influence function of the OR estimator (put all pieces together)
    reg_att_inf_func = inf_treat - inf_control

    return reg_att_inf_func
    """


'''

path.write_text(new_source, encoding="utf-8")
importlib.invalidate_caches()

print(f"Overwrote {path}")


