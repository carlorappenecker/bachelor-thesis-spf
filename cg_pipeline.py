"""
============================================================================
Coibion-Gorodnichenko (2015) predictability test  --  CORE PIPELINE
============================================================================

This module contains the *reusable* pipeline. It does NOT care whether the
data are simulated or the real ECB SPF -- it only needs:

  (1) micro-level forecasts in "long" format, one row per
      (survey_round, target, forecaster_id) with a numeric point forecast, and
  (2) a mapping target -> realized inflation (one number per target).

The CG test regresses the CONSENSUS (mean-across-forecasters) forecast ERROR
on the CONSENSUS forecast REVISION:

        error_t  =  c  +  beta * revision_t  +  u_t

    error_t     = realized(target)      - F_t(target)       (how wrong they were)
    revision_t  = F_t(target)           - F_{t-1}(target)   (how they changed their mind)

Interpretation of beta:
    * Full-Information Rational Expectations (FIRE)  ->  beta = 0
    * Information rigidity / under-reaction          ->  beta > 0
    * Sticky-information degree lambda               ->  lambda = beta / (1 + beta)
      (equivalently beta = lambda / (1 - lambda))

Because consecutive fixed-target forecasts overlap in the future shocks they
must predict, the error series is serially correlated (an MA process). Plain
OLS standard errors are therefore WRONG; we use Newey-West (HAC) standard
errors. This is the single most common mistake in applied CG tests.
============================================================================
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# STEP A -- build the consensus (mean across forecasters) for each (round, target)
# ---------------------------------------------------------------------------
def build_consensus(micro, round_col="survey_round", target_col="target",
                    id_col="forecaster_id", value_col="point_forecast",
                    method="mean"):
    """Collapse micro-level forecasts to one consensus number per (round, target).

    The CG test runs on the consensus, NOT on individual forecasters: the
    predictability of errors from revisions is an *emergent property of
    aggregation* (Coibion & Gorodnichenko 2015). 'method' lets you switch the
    aggregator for robustness (mean is the baseline; median/trimmed as checks).
    """
    g = micro.groupby([round_col, target_col])[value_col]
    if method == "mean":
        consensus = g.mean()
    elif method == "median":
        consensus = g.median()
    elif method == "trimmed":               # 10% trimmed mean, robustness only
        from scipy.stats import trim_mean
        consensus = g.apply(lambda s: trim_mean(s, 0.1))
    else:
        raise ValueError(f"unknown method {method!r}")
    out = consensus.rename("consensus").reset_index()
    out["n_forecasters"] = g.size().values     # keep the panel width for reporting
    return out


# ---------------------------------------------------------------------------
# STEP B -- build forecast errors and forecast revisions
# ---------------------------------------------------------------------------
def build_error_revision(consensus, realized,
                        round_col="survey_round", target_col="target",
                        round_order=None):
    """Given the consensus panel and a realized-inflation lookup, construct the
    two regression columns.

    revision_t = consensus_t(target) - consensus_{t-1}(target)
                 -> consecutive survey rounds for the SAME target
    error_t    = realized(target)    - consensus_t(target)

    'round_order' is an ordered list mapping each survey_round label to its
    chronological position, so that "previous round" is well defined even when
    rounds are strings like '2019Q4'.
    """
    df = consensus.copy()

    # 1) put rounds in true chronological order -------------------------------
    if round_order is None:
        # assume the labels already sort correctly (e.g. pandas Period or ints)
        df["_rank"] = df[round_col].rank(method="dense").astype(int)
    else:
        rank_map = {r: i for i, r in enumerate(round_order)}
        df["_rank"] = df[round_col].map(rank_map)
        if df["_rank"].isna().any():
            missing = df.loc[df["_rank"].isna(), round_col].unique()
            raise ValueError(f"rounds not found in round_order: {missing}")

    # 2) within each TARGET, the previous round's consensus = the revision base
    df = df.sort_values([target_col, "_rank"]).reset_index(drop=True)
    df["prev_consensus"] = df.groupby(target_col)["consensus"].shift(1)
    df["prev_rank"] = df.groupby(target_col)["_rank"].shift(1)

    # only keep genuinely CONSECUTIVE rounds (guard against gaps in the panel)
    consecutive = (df["_rank"] - df["prev_rank"]) == 1
    df["revision"] = np.where(consecutive, df["consensus"] - df["prev_consensus"], np.nan)

    # 3) merge realized inflation and form the error --------------------------
    realized = realized.rename("realized")
    df = df.merge(realized, left_on=target_col, right_index=True, how="left")
    df["error"] = df["realized"] - df["consensus"]

    return df


# ---------------------------------------------------------------------------
# STEP C -- the CG regression (with Newey-West / HAC standard errors)
# ---------------------------------------------------------------------------
def run_cg_regression(df, hac_lags, error_col="error", revision_col="revision",
                    order_col="_rank"):
    """OLS of error on revision with Newey-West (HAC) standard errors.

    'hac_lags' MUST reflect the overlap in the data: for h-quarter-ahead
    quarterly forecasts the error series is MA(h-1), so a natural choice is
    hac_lags = h - 1 (e.g. 3 for a one-year/4-quarter horizon).
    """
    d = df.dropna(subset=[error_col, revision_col]).copy()
    d = d.sort_values(order_col)                    # HAC assumes time ordering
    X = sm.add_constant(d[[revision_col]].rename(columns={revision_col: "revision"}))
    y = d[error_col]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model


# ---------------------------------------------------------------------------
# STEP D -- the pre/post-2020 test, done the RIGHT way (pooled + interaction)
# ---------------------------------------------------------------------------
def run_cg_split(df, hac_lags, break_rank, error_col="error",
                revision_col="revision", order_col="_rank"):
    """Pooled regression with a post-break dummy D and an interaction:

        error = c + beta*revision + gamma*D + delta*(D*revision) + u

    * beta          = the CG slope BEFORE the break
    * beta + delta  = the CG slope AFTER the break
    * delta         = the CHANGE in the slope  <-- this is hypothesis H2,
                      tested directly with one coefficient and its HAC t-stat.

    'break_rank' is the chronological rank at/after which D = 1 (i.e. the first
    post-2020 survey round). Splitting on the survey-round DATE (not the target
    year) is correct, because beta describes updating behaviour AT that date.
    """
    d = df.dropna(subset=[error_col, revision_col]).copy()
    d = d.sort_values(order_col)
    d["D"] = (d[order_col] >= break_rank).astype(float)
    d["revision"] = d[revision_col]
    d["D_x_revision"] = d["D"] * d["revision"]
    X = sm.add_constant(d[["revision", "D", "D_x_revision"]])
    y = d[error_col]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model


def beta_to_lambda(beta):
    """Map the CG slope to the implied sticky-information share lambda."""
    return beta / (1.0 + beta)
