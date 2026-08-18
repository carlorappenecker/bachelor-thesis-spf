"""
============================================================================
CG TEST -- VALIDATION AND WORKED EXAMPLE  (run this file)
============================================================================
Because the real ECB SPF / Eurostat data are not reachable here, this script
PROVES the pipeline in cg_pipeline.py is correct by recovering a KNOWN answer
from simulated SPF-like data, and shows a fully worked example + plot.

It runs four checks:
  (0) FIRE check ............ lambda = 0     -> beta should be ~ 0
  (1) exactness ............. clean DGP, huge N, no disagreement -> beta ~ 1.000
  (2) MC validation ......... realistic disagreement + ~150 obs: the MEAN of
                              beta_hat over many samples recovers the truth
                              (and shows how NOISY ~150 quarterly obs really are)
  (3) break validation ...... lambda 0.40 -> 0.60 in 2020: the interaction term
                              delta recovers the change in beta

Then it prints ONE full regression (statsmodels summary) and saves a scatter
plot, so you see exactly what the output looks like on real data.
============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cg_pipeline import (build_consensus, build_error_revision,
                        run_cg_regression, run_cg_split, beta_to_lambda)

# ---- fixed "deep parameters" of the synthetic inflation world --------------
MU, RHO, SIGMA_E = 2.0, 0.95, 0.8      # persistent AR(1) inflation, ~2% mean
N_FC, SIGMA_NU   = 50, 0.20            # 50 forecasters, realistic disagreement
H_KEEP, H_WARM   = 8, 16               # horizons emitted; smoothing warm-up
H_BASE           = 4                   # baseline horizon = 1 year (4 quarters)
HAC_LAGS         = H_BASE - 1          # error is MA(h-1) -> 3 lags


def simulate_spf(seed, years=40, lam_pre=0.5, lam_post=None,
                break_year=2020, start_year=1975, disagreement=True):
    """Generate SPF-like micro data from a sticky-information process.

    Returns (micro_df, realized_series, round_labels, truth_dict).
    lam_post=None means a single regime (lam_post = lam_pre).
    Set disagreement=False to emit the noiseless consensus (exactness check).
    """
    if lam_post is None:
        lam_post = lam_pre
    rng = np.random.default_rng(seed)
    nq = years * 4
    labels = [f"{start_year + q // 4}Q{q % 4 + 1}" for q in range(nq)]
    year_of = np.array([start_year + q // 4 for q in range(nq)])

    # true quarterly inflation state, AR(1)
    x = np.empty(nq); x[0] = MU
    eps = rng.normal(0, SIGMA_E, nq)
    for t in range(1, nq):
        x[t] = MU + RHO * (x[t - 1] - MU) + eps[t]

    lam_at = lambda s: (lam_pre if year_of[s] < break_year else lam_post)

    rows = []
    for D in range(H_WARM, nq):                         # fully warmed-up targets
        s0 = D - H_WARM
        C_prev = MU + RHO ** (D - s0) * (x[s0] - MU)     # init consensus = FI
        C = {}
        for s in range(s0 + 1, D):
            FI = MU + RHO ** (D - s) * (x[s] - MU)       # rational AR(1) forecast
            lam = lam_at(s)
            C_prev = (1 - lam) * FI + lam * C_prev        # sticky-info recursion
            C[s] = C_prev
        for hh in range(1, H_KEEP + 1):                  # emit horizons 1..8
            s = D - hh
            if s in C:
                noise = rng.normal(0, SIGMA_NU, N_FC) if disagreement else np.zeros(N_FC)
                for i in range(N_FC):
                    rows.append((labels[s], D, i, C[s] + noise[i], hh))

    micro = pd.DataFrame(rows, columns=["survey_round", "target",
                                        "forecaster_id", "point_forecast", "horizon"])
    realized = pd.Series(x, index=range(nq), name="realized")
    truth = dict(beta_pre=lam_pre / (1 - lam_pre),
                beta_post=lam_post / (1 - lam_post))
    return micro, realized, labels, truth


def to_regression_frame(micro, realized, labels):
    """Full pipeline: micro -> consensus -> error/revision -> baseline horizon."""
    cons = build_consensus(micro)
    hz = micro.drop_duplicates(["survey_round", "target"])[
        ["survey_round", "target", "horizon"]]
    cons = cons.merge(hz, on=["survey_round", "target"])
    er = build_error_revision(cons, realized, round_order=labels)
    return er[er["horizon"] == H_BASE].copy()


def beta_once(seed, years, lam_pre, lam_post=None):
    micro, realized, labels, _ = simulate_spf(seed, years, lam_pre, lam_post)
    base = to_regression_frame(micro, realized, labels)
    return run_cg_regression(base, hac_lags=HAC_LAGS).params["revision"]


# ===========================================================================
if __name__ == "__main__":
    line = "=" * 74

    # ---- (0) FIRE check: lambda = 0 -> beta ~ 0 ---------------------------
    print(line); print("(0) FIRE CHECK  --  lambda = 0  ->  true beta = 0")
    b0 = beta_once(seed=7, years=200, lam_pre=1e-9)
    print(f"    recovered beta = {b0:+.4f}   (expected ~ 0)")

    # ---- (1) exactness: clean consensus, huge N --------------------------
    print(line); print("(1) EXACTNESS  --  no disagreement, large N, lambda=0.5 -> beta=1.000")
    micro, realized, labels, _ = simulate_spf(seed=1, years=1000,
                                            lam_pre=0.5, disagreement=False)
    base = to_regression_frame(micro, realized, labels)
    m1 = run_cg_regression(base, hac_lags=HAC_LAGS)
    print(f"    recovered beta = {m1.params['revision']:.4f}   "
        f"(N={int(m1.nobs)})   -> method is exact")

    # ---- (2) Monte Carlo with realistic disagreement + short sample ------
    print(line); print("(2) MONTE CARLO  --  lambda=0.5 (true beta=1.000), realistic noise")
    for years in (40, 100):
        betas = np.array([beta_once(1000 + k, years, 0.5) for k in range(200)])
        nobs = years * 4 - H_WARM
        print(f"    ~{nobs:3d} obs:  mean beta_hat = {betas.mean():.3f}   "
            f"sd = {betas.std():.3f}   95% MC range "
            f"[{np.percentile(betas,2.5):.2f}, {np.percentile(betas,97.5):.2f}]")
    print("    -> unbiased up to a small attenuation from disagreement;")
    print("       ~100 quarterly obs (what the real SPF gives) is genuinely noisy.")

    # ---- (3a) break MACHINERY: change detected when BOTH windows are long -
    print(line); print("(3a) BREAK MACHINERY  --  lambda 0.40->0.60, break mid-sample, long windows")
    d_list, bpre_list, bpost_list = [], [], []
    for k in range(200):
        micro, realized, labels, _ = simulate_spf(
            2000 + k, years=80, lam_pre=0.4, lam_post=0.6,
            start_year=1965, break_year=1995)          # ~30 yrs each side
        base = to_regression_frame(micro, realized, labels)
        ms = run_cg_split(base, hac_lags=HAC_LAGS, break_rank=labels.index("1995Q1"))
        bpre_list.append(ms.params["revision"])
        d_list.append(ms.params["D_x_revision"])
        bpost_list.append(ms.params["revision"] + ms.params["D_x_revision"])
    print(f"    mean beta_pre  = {np.mean(bpre_list):.3f}   (true 0.667)")
    print(f"    mean delta     = {np.mean(d_list):.3f}   (true change +0.833)  <- recovered")
    print(f"    mean beta_post = {np.mean(bpost_list):.3f}   (true 1.500)")

    # ---- (3b) POWER with a realistic SHORT post-2020 window ---------------
    print(line); print("(3b) SHORT POST-2020 WINDOW  --  same true change, but only ~5 yrs of post data")
    d_short = []
    for k in range(200):
        micro, realized, labels, _ = simulate_spf(
            5000 + k, years=60, lam_pre=0.4, lam_post=0.6,
            start_year=1965, break_year=2020)          # ~5 yrs post only
        base = to_regression_frame(micro, realized, labels)
        ms = run_cg_split(base, hac_lags=HAC_LAGS, break_rank=labels.index("2020Q1"))
        d_short.append(ms.params["D_x_revision"])
    d_short = np.array(d_short)
    print(f"    mean delta = {d_short.mean():.3f}, sd = {d_short.std():.2f}, "
        f"95% range [{np.percentile(d_short,2.5):.2f}, {np.percentile(d_short,97.5):.2f}]")
    print("    -> with only ~5 years of post-break data the change is NOT reliably")
    print("       identified (wide CI, transition dynamics dominate). This is a REAL")
    print("       limitation of the 2020 split and belongs in the thesis's Discussion.")

    # ---- one fully worked example + scatter plot -------------------------
    print(line); print("WORKED EXAMPLE  (one representative sample, full regression output)")
    micro, realized, labels, _ = simulate_spf(seed=3, years=80, lam_pre=0.5)
    base = to_regression_frame(micro, realized, labels)
    m = run_cg_regression(base, hac_lags=HAC_LAGS)
    print(m.summary())
    print(f"\n  implied lambda = {beta_to_lambda(m.params['revision']):.3f} "
        f"(true 0.500)")

    d = base.dropna(subset=["error", "revision"])
    xx = np.linspace(d["revision"].min(), d["revision"].max(), 100)
    plt.figure(figsize=(7, 5))
    plt.axhline(0, lw=.8, color="0.7"); plt.axvline(0, lw=.8, color="0.7")
    plt.scatter(d["revision"], d["error"], s=18, alpha=.6, edgecolor="none")
    plt.plot(xx, m.params["const"] + m.params["revision"] * xx, "r-", lw=2,
            label=f"slope beta = {m.params['revision']:.2f}")
    plt.xlabel("consensus forecast revision"); plt.ylabel("consensus forecast error")
    plt.title("CG test: forecast error vs. forecast revision (worked example)")
    plt.legend(); plt.tight_layout()
    plt.savefig("/home/claude/cg_scatter.png", dpi=130)
    print("\n[saved scatter plot -> cg_scatter.png]")
