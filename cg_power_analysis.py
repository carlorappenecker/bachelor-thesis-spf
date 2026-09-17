"""
============================================================================
CG THESIS -- STATISTICAL POWER OF THE PRE/POST-2020 BREAK TEST (Section 4.3)
Companion to cg_spf_replication.py
============================================================================
Section 4.3 makes two power claims about the pre/post-2020 comparison:

  (A) The pooled predictability slope beta_hat is itself estimated with
      considerable sampling uncertainty at the sample sizes available here.
  (B) The ability to detect a CHANGE in beta (the interaction coefficient
      delta) is governed by the standard error of delta; given that
      standard error, only an implausibly large change could be detected
      with conventional (80%) power at the 5% level.

This script quantifies both claims with a Monte Carlo experiment calibrated
directly to the real estimation sample (not to illustrative or generic
parameters): the true sample sizes (N = 102 for the one-year-ahead
specification, N = 255 pooled), the true pre/post-2020 split of those
samples, the true variance of the revision regressor in each regime, the
true OLS residual variance, and beta = 0.812 (the estimated slope, which
implies lambda = beta / (1 + beta) = 0.448 under the sticky-information
mapping used throughout the thesis).

PART A resamples beta_hat at the two real sample sizes under the true DGP
(e = beta * r + u) to show how dispersed the slope estimate is on its own.

PART B simulates the true null of NO break (same beta in both regimes),
using the true pre/post sample sizes and the true variance of the revision
in each regime, and computes the empirical standard error of the OLS
interaction coefficient delta. The minimum detectable effect (MDE) is
2.8 x SE(delta), the standard rule of thumb for 80% power at a two-sided
5% test.

USAGE:  python cg_power_analysis.py
OUTPUT: prints the exact numbers reported in Section 4.3.

Requires: python>=3.10, numpy, pandas, statsmodels
============================================================================
"""
import re, glob, os, io
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SPF_ZIP     = "Rates_BA_1.zip"
EUROSTAT_GZ = "Rates_BA_2_csv.gz"
BREAK_ROUND = "2020Q1"
REPS        = 10000
SEED        = 20260917
MDE_FACTOR  = 2.8   # 80% power, two-sided 5% test
SECTION_MARKERS = ("INFLATION EXPECTATIONS", "CORE INFLATION", "GROWTH EXPECTATIONS",
                "EXPECTED UNEMPLOYMENT", "ASSUMPTIONS")


# ---------------------------------------------------------------------------
# DATA LOADING (identical parsing to the baseline script)
# ---------------------------------------------------------------------------
def read_hicp_block(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("INFLATION EXPECTATIONS"))
    except StopIteration:
        return None
    stop = next((i for i in range(start + 2, len(lines))
                if any(lines[i].startswith(m) for m in SECTION_MARKERS)), len(lines))
    df = pd.read_csv(io.StringIO("\n".join(lines[start + 1:stop])))
    if not {"TARGET_PERIOD", "FCT_SOURCE", "POINT"}.issubset(df.columns):
        return None
    df = df[["TARGET_PERIOD", "FCT_SOURCE", "POINT"]].copy()
    df["POINT"] = pd.to_numeric(df["POINT"], errors="coerce")
    return df


def load_spf(zip_path, workdir="_spf_power"):
    import zipfile
    os.makedirs(workdir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(workdir)
    frames = []
    for path in sorted(glob.glob(os.path.join(workdir, "*.csv"))):
        rnd = os.path.splitext(os.path.basename(path))[0]
        if not re.fullmatch(r"\d{4}Q[1-4]", rnd):
            continue
        d = read_hicp_block(path)
        if d is None:
            continue
        d = d[d["TARGET_PERIOD"].astype(str).str.fullmatch(r"\d{4}")].copy()
        d["target"] = d["TARGET_PERIOD"].astype(int)
        d["round_year"] = int(rnd[:4])
        d["h"] = d["target"] - d["round_year"]
        d["round"] = rnd
        frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    return m.rename(columns={"FCT_SOURCE": "forecaster_id", "POINT": "point"})


def load_realized(gz_path):
    eu = pd.read_csv(gz_path)
    r = eu[(eu["unit"] == "Annual average rate of change") &
        (eu["coicop"] == "All-items HICP") &
        (eu["geo"].astype(str).str.startswith("Euro area (EA11"))].copy()
    r["target"] = r["TIME_PERIOD"].astype(int)
    return r.set_index("target")["OBS_VALUE"].astype(float).sort_index()


def build_consensus(m, realized, rank):
    cons = m.groupby(["round", "target", "h", "rank"])["point"].mean().rename("cons").reset_index()
    cons = cons.sort_values(["target", "rank"])
    cons["prev"] = cons.groupby("target")["cons"].shift(1)
    cons["prank"] = cons.groupby("target")["rank"].shift(1)
    cons["rev"] = np.where(cons["rank"] - cons["prank"] == 1, cons["cons"] - cons["prev"], np.nan)
    cons = cons.merge(realized.rename("real"), left_on="target", right_index=True, how="left")
    cons["err"] = cons["real"] - cons["cons"]
    return cons.dropna(subset=["err", "rev"]).sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CALIBRATION: extract real N, real Var(revision) by regime, and the real
# OLS residual variance, for both specifications
# ---------------------------------------------------------------------------
def calibrate(cons, break_rank):
    cons = cons.copy()
    cons["is_post"] = cons["rank"] >= break_rank
    out = {}
    for label, sub in [("one-year-ahead (h=1)", cons[cons.h == 1]),
                        ("pooled (h=0,1,2)", cons)]:
        X = sm.add_constant(sub[["rev"]])
        fit = sm.OLS(sub["err"], X).fit()
        out[label] = dict(
            N=len(sub),
            n_pre=int((~sub["is_post"]).sum()),
            n_post=int(sub["is_post"].sum()),
            mean_rev=sub["rev"].mean(),
            var_rev=sub["rev"].var(ddof=1),
            var_rev_pre=sub.loc[~sub.is_post, "rev"].var(ddof=1),
            var_rev_post=sub.loc[sub.is_post, "rev"].var(ddof=1),
            resid_var=fit.resid.var(ddof=2),
            beta_hat=fit.params["rev"],
        )
    return out


# ---------------------------------------------------------------------------
# PART A: sampling dispersion of beta_hat at the two real sample sizes
# ---------------------------------------------------------------------------
def part_a_dispersion(calib, beta_true, reps, rng):
    print("=" * 74)
    print("PART A: sampling dispersion of beta_hat at the two real sample sizes")
    print("=" * 74)
    results = {}
    for label, c in calib.items():
        N, mu_r, var_r, resid_sd = c["N"], c["mean_rev"], c["var_rev"], c["resid_var"] ** 0.5
        betas = np.empty(reps)
        for i in range(reps):
            r = rng.normal(mu_r, var_r ** 0.5, N)
            u = rng.normal(0, resid_sd, N)
            e = beta_true * r + u
            X = sm.add_constant(r)
            betas[i] = np.linalg.lstsq(X, e, rcond=None)[0][1]
        lo, hi = np.percentile(betas, [5, 95])
        results[label] = (lo, hi)
        print(f"[{label}]  N={N}: mean(beta_hat)={betas.mean():.3f}  "
            f"90% range [{lo:.2f}, {hi:.2f}]")
    return results


# ---------------------------------------------------------------------------
# PART B: SE of delta under the true null of no break, and the implied MDE
# ---------------------------------------------------------------------------
def part_b_delta_se(calib, beta_true, reps, mde_factor, rng):
    print()
    print("=" * 74)
    print("PART B: SE of delta under the null of no true break "
        "(real pre/post split)")
    print("=" * 74)
    results = {}
    for label, c in calib.items():
        n_pre, n_post = c["n_pre"], c["n_post"]
        var_pre, var_post = c["var_rev_pre"], c["var_rev_post"]
        resid_sd = c["resid_var"] ** 0.5
        deltas = np.empty(reps)
        for i in range(reps):
            r_pre = rng.normal(0, var_pre ** 0.5, n_pre)
            r_post = rng.normal(0, var_post ** 0.5, n_post)
            r = np.concatenate([r_pre, r_post])
            D = np.concatenate([np.zeros(n_pre), np.ones(n_post)])
            u = rng.normal(0, resid_sd, n_pre + n_post)
            e = beta_true * r + u  # same beta throughout: true delta = 0
            X = np.column_stack([np.ones(n_pre + n_post), r, D, D * r])
            coef = np.linalg.lstsq(X, e, rcond=None)[0]
            deltas[i] = coef[3]
        se_delta = deltas.std(ddof=1)
        mde = mde_factor * se_delta
        results[label] = (se_delta, mde)
        print(f"[{label}]  SE(delta) = {se_delta:.3f}   "
            f"MDE ({mde_factor}x) = {mde:.3f}")
    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    m = load_spf(SPF_ZIP)
    m = m[m["h"].isin([0, 1, 2])].dropna(subset=["point"]).copy()
    order = sorted(m["round"].unique(), key=lambda r: (int(r[:4]), int(r[5])))
    rank = {r: i for i, r in enumerate(order)}
    m["rank"] = m["round"].map(rank)
    realized = load_realized(EUROSTAT_GZ)
    cons = build_consensus(m, realized, rank)

    calib = calibrate(cons, rank[BREAK_ROUND])
    beta_true = calib["pooled (h=0,1,2)"]["beta_hat"]
    lam = beta_true / (1 + beta_true)
    print(f"Calibration: beta = {beta_true:.4f}  (implied lambda = {lam:.4f})")
    for label, c in calib.items():
        print(f"  [{label}] N={c['N']} (pre={c['n_pre']}, post={c['n_post']}), "
            f"Var(rev|pre)={c['var_rev_pre']:.5f}, "
            f"Var(rev|post)={c['var_rev_post']:.5f}, "
            f"resid_var={c['resid_var']:.5f}")
    print()

    rng = np.random.default_rng(SEED)
    part_a_dispersion(calib, beta_true, REPS, rng)
    part_b_delta_se(calib, beta_true, REPS, MDE_FACTOR, rng)


if __name__ == "__main__":
    main()
