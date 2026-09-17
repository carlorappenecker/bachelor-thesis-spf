"""
============================================================================
CG THESIS -- FORMAL TEST OF THE CONSENSUS/INDIVIDUAL WEDGE (Section 6.7)
Companion to cg_spf_replication.py and cg_extensions.py
============================================================================
Section 6.7 reports beta_consensus = 0.812 and beta_individual = 0.330
(0.286 with forecaster fixed effects) and interprets the gap as the
aggregation wedge of Coibion and Gorodnichenko (2015) and Bordalo et al.
(2020), without a formal test. This module tests
    H0: delta = beta_consensus - beta_individual = 0
against the alternative that under-reaction is materially stronger at the
consensus level than at the individual level.

The two slopes are not independent: the consensus observation for a given
(round, target, horizon) cell is the cross-forecaster mean of exactly the
individual observations for that same cell, so a comparison must account
for this shared dependence rather than treat the two samples as if drawn
independently.

Two complementary tests are used.

(1) Stacked interaction regression. The consensus rows and the individual
    rows are stacked into one dataset with an indicator C = 1 for consensus
    rows, 0 for individual rows, and

        e = alpha + alpha_C * C + beta * r + delta * (C * r) + u

    is estimated by OLS, where delta is beta_consensus - beta_individual by
    construction. Standard errors are clustered on survey round, which
    allows arbitrary within-round correlation between a round's consensus
    observation and the individual observations it is built from; a
    two-way clustered version (round and target year) is also reported,
    matching the inference approach used elsewhere in the thesis (Section
    4.4).

(2) Cluster bootstrap over survey rounds. Survey rounds are resampled with
    replacement (2,000 replications, fixed seed). Within each replication,
    both the consensus rows and the individual rows belonging to the
    resampled rounds are pulled together (not re-differenced against a
    resampled time order, since that would corrupt the fixed-event
    revision definition) and both slopes are re-estimated by OLS; the same
    resampled round list is used for both samples in a given replication,
    preserving the mechanical link between them. The empirical distribution
    of (beta_consensus - beta_individual) across replications gives a
    bootstrap standard error, a 95% percentile interval and a two-sided
    p-value (normal approximation using the bootstrap standard error,
    matching the convention used for the Huber and median-regression
    bootstrap in Section 6.8).

BEFORE ESTIMATING, the two samples are checked for exact cell-level
harmonisation: the set of (survey_round, target, horizon) cells covered by
the consensus sample must equal the set covered by the individual panel.

USAGE:  python cg_wedge_test.py
INPUT:  the same two raw files as the rest of the pipeline (edit the paths
        below if needed).
OUTPUT: prints the harmonisation check and the full results block.

Requires: python>=3.10, pandas, numpy, statsmodels, scipy
============================================================================
"""
import re, glob, os, io
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SPF_ZIP     = "Rates_BA_1.zip"
EUROSTAT_GZ = "Rates_BA_2_csv.gz"
HAC_LAGS    = 4
BOOT_REPS   = 2000
BOOT_SEED   = 20260908
SECTION_MARKERS = ("INFLATION EXPECTATIONS", "CORE INFLATION", "GROWTH EXPECTATIONS",
                "EXPECTED UNEMPLOYMENT", "ASSUMPTIONS")


# ---------------------------------------------------------------------------
# DATA LOADING (identical parsing to the baseline and extension scripts)
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


def load_spf(zip_path, workdir="_spf_wedge"):
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


# ---------------------------------------------------------------------------
# BUILD THE TWO SAMPLES (horizons 0, 1, 2; identical construction rules to
# cg_spf_replication.py for consensus and cg_extensions.py Section 6.7 for
# the individual panel)
# ---------------------------------------------------------------------------
def build_consensus(m, realized, order, rank):
    cons = m.groupby(["round", "target", "h", "rank"])["point"].mean().rename("cons").reset_index()
    cons = cons.sort_values(["target", "rank"])
    cons["prev"] = cons.groupby("target")["cons"].shift(1)
    cons["prank"] = cons.groupby("target")["rank"].shift(1)
    cons["rev"] = np.where(cons["rank"] - cons["prank"] == 1, cons["cons"] - cons["prev"], np.nan)
    cons = cons.merge(realized.rename("real"), left_on="target", right_index=True, how="left")
    cons["err"] = cons["real"] - cons["cons"]
    return cons.dropna(subset=["err", "rev"]).sort_values("rank").reset_index(drop=True)


def build_individual(m, realized, rank):
    mi = m.sort_values(["forecaster_id", "target", "rank"]).copy()
    g = mi.groupby(["forecaster_id", "target"])
    mi["prev"] = g["point"].shift(1)
    mi["prank"] = g["rank"].shift(1)
    mi["rev"] = np.where(mi["rank"] - mi["prank"] == 1, mi["point"] - mi["prev"], np.nan)
    mi = mi.merge(realized.rename("real"), left_on="target", right_index=True, how="left")
    mi["err"] = mi["real"] - mi["point"]
    return mi.dropna(subset=["err", "rev"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# HARMONISATION CHECK
# ---------------------------------------------------------------------------
def check_harmonisation(cons, indiv):
    cons_cells = set(zip(cons["round"], cons["target"], cons["h"]))
    indiv_cells = set(zip(indiv["round"], indiv["target"], indiv["h"]))
    only_cons = cons_cells - indiv_cells
    only_indiv = indiv_cells - cons_cells
    print(f"Consensus cells: {len(cons_cells)} | Individual-panel cells: {len(indiv_cells)}")
    print(f"Cells in consensus but not individual: {len(only_cons)}"
        + (f" -> {sorted(only_cons)}" if only_cons else ""))
    print(f"Cells in individual but not consensus: {len(only_indiv)}"
        + (f" -> {sorted(only_indiv)[:10]}" if only_indiv else ""))
    per_cell = indiv.groupby(["round", "target", "h"]).size()
    print(f"Individual observations per cell: min={per_cell.min()}, "
        f"median={per_cell.median():.0f}, max={per_cell.max()}")
    return len(only_cons) == 0 and len(only_indiv) == 0


# ---------------------------------------------------------------------------
# (1) STACKED INTERACTION REGRESSION
# ---------------------------------------------------------------------------
def stacked_interaction_test(cons, indiv):
    c = cons[["round", "target", "h", "rev", "err"]].copy(); c["C"] = 1.0
    i = indiv[["round", "target", "h", "rev", "err"]].copy(); i["C"] = 0.0
    stacked = pd.concat([c, i], ignore_index=True)
    stacked["Cxrev"] = stacked["C"] * stacked["rev"]
    X = sm.add_constant(stacked[["C", "rev", "Cxrev"]])
    y = stacked["err"]

    g_round = stacked["round"].astype("category").cat.codes.values
    m_round = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g_round, "use_correction": True})

    g_target = stacked["target"].astype("category").cat.codes.values
    g2 = np.column_stack([g_round, g_target])
    m_2way = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g2, "use_correction": True})

    return stacked, m_round, m_2way


# ---------------------------------------------------------------------------
# (2) CLUSTER BOOTSTRAP OVER SURVEY ROUNDS (paired resampling)
# ---------------------------------------------------------------------------
def cluster_bootstrap(cons, indiv, reps=BOOT_REPS, seed=BOOT_SEED):
    round_list = sorted(set(cons["round"]) & set(indiv["round"]))
    cons_groups = {r: cons[cons["round"] == r] for r in round_list}
    indiv_groups = {r: indiv[indiv["round"] == r] for r in round_list}

    def beta_ols(d):
        X = sm.add_constant(d[["rev"]])
        return sm.OLS(d["err"], X).fit().params["rev"]

    rng = np.random.default_rng(seed)
    deltas = np.empty(reps); bc = np.empty(reps); bi = np.empty(reps)
    for b in range(reps):
        draw = rng.choice(round_list, size=len(round_list), replace=True)
        c_pseudo = pd.concat([cons_groups[r] for r in draw], ignore_index=True)
        i_pseudo = pd.concat([indiv_groups[r] for r in draw], ignore_index=True)
        bc[b] = beta_ols(c_pseudo)
        bi[b] = beta_ols(i_pseudo)
        deltas[b] = bc[b] - bi[b]
    return deltas, bc, bi


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

    cons = build_consensus(m, realized, order, rank)
    indiv = build_individual(m, realized, rank)
    print(f"Consensus sample: {len(cons)} obs | Individual sample: {len(indiv)} obs\n")

    print("=== Harmonisation check ===")
    ok = check_harmonisation(cons, indiv)
    print("Harmonised (identical cell sets):", ok)
    if not ok:
        print("WARNING: samples are not cell-harmonised; see cells listed above.")
    print()

    # baseline slopes on these exact samples, for reference
    Xc = sm.add_constant(cons[["rev"]])
    mc = sm.OLS(cons["err"], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    Xi = sm.add_constant(indiv[["rev"]])
    g2i = np.column_stack([indiv["forecaster_id"].astype("category").cat.codes,
                            indiv["round"].astype("category").cat.codes])
    mi_ = sm.OLS(indiv["err"], Xi).fit(cov_type="cluster", cov_kwds={"groups": g2i, "use_correction": True})
    print(f"Consensus (harmonised sample):  beta={mc.params['rev']:.4f} "
        f"(NW se {mc.bse['rev']:.4f}) N={int(mc.nobs)}")
    print(f"Individual (harmonised sample): beta={mi_.params['rev']:.4f} "
        f"(2-way cluster se {mi_.bse['rev']:.4f}) N={int(mi_.nobs)}\n")

    print("=== (1) Stacked interaction regression ===")
    stacked, m_round, m_2way = stacked_interaction_test(cons, indiv)
    print(f"stacked N = {len(stacked)} ({len(cons)} + {len(indiv)})\n")
    for name, mres in [("cluster(round)", m_round), ("cluster(round,target) two-way", m_2way)]:
        d, se = mres.params["Cxrev"], mres.bse["Cxrev"]
        print(f"[{name}]")
        print(f"  delta = {d:.4f}  se = {se:.4f}  t = {d/se:.3f}  p = {mres.pvalues['Cxrev']:.4f}")
        print(f"  beta_individual (rev) = {mres.params['rev']:.4f}, "
            f"implied beta_consensus = {mres.params['rev']+d:.4f}, N = {int(mres.nobs)}\n")

    print("=== (2) Cluster bootstrap over survey rounds ===")
    deltas, bc, bi = cluster_bootstrap(cons, indiv)
    point_delta = mc.params["rev"] - mi_.params["rev"]
    se_boot = deltas.std(ddof=1)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    z = point_delta / se_boot
    p_boot = 2 * (1 - norm.cdf(abs(z)))
    print(f"reps = {len(deltas)}, seed = {BOOT_SEED}")
    print(f"point delta (original sample) = {point_delta:.4f}")
    print(f"bootstrap SE                  = {se_boot:.4f}")
    print(f"95% percentile interval       = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"z = {z:.3f}, two-sided p (normal approx) = {p_boot:.4f}")
    print(f"share of bootstrap draws with delta <= 0: {(deltas<=0).mean():.4f}")


if __name__ == "__main__":
    main()
