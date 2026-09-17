"""
============================================================================
CG THESIS -- EXTENSION ANALYSES (Sections 6.4 to 6.7)
Companion to cg_spf_replication.py  --  Carlo Rappenecker, Bachelor Thesis
============================================================================
This single, self-contained file reproduces the four complementary analyses
that go beyond the baseline Coibion-Gorodnichenko regression:

  6.4  Anchoring of long-term inflation expectations  -> cg_anchoring.png
  6.5  Time variation in the coefficient (rolling beta) -> cg_rolling_beta.png
  6.6  Forecaster disagreement                        -> cg_disagreement.png
  6.7  Individual-level CG test (pooled and with forecaster fixed effects)

INPUTS (same two raw files as the baseline script; edit the paths if needed):
  SPF_ZIP     : ECB SPF microdata, one CSV per round (e.g. "Rates_BA_1.zip")
  EUROSTAT_GZ : Eurostat PRC_HICP_AIND, realized HICP inflation (.csv.gz)

USAGE:   python cg_extensions.py
OUTPUT:  prints the numbers reported in Sections 6.4 to 6.7 and writes the
         three figures. The estimation is deterministic.

Requires: python>=3.10, pandas, numpy, statsmodels, matplotlib
============================================================================
"""
import re, glob, os, io, zipfile
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SPF_ZIP     = "Rates_BA_1.zip"
EUROSTAT_GZ = "Rates_BA_2_csv.gz"
HAC_LAGS    = 4
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


def load_spf(zip_path, workdir="_spf_ext"):
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
        d["h"] = d["target"] - d["round_year"]           # horizon in years
        d["round"] = rnd
        frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    m = m.rename(columns={"FCT_SOURCE": "forecaster_id", "POINT": "point"})
    return m


def load_realized(gz_path):
    eu = pd.read_csv(gz_path)
    r = eu[(eu["unit"] == "Annual average rate of change") &
        (eu["coicop"] == "All-items HICP") &
        (eu["geo"].astype(str).str.startswith("Euro area (EA11"))].copy()
    r["target"] = r["TIME_PERIOD"].astype(int)
    return r.set_index("target")["OBS_VALUE"].astype(float).sort_index()


def yq(round_label):
    """decimal year of a survey round, e.g. 2021Q2 -> 2021.25"""
    return int(round_label[:4]) + (int(round_label[5]) - 1) / 4


# ---------------------------------------------------------------------------
# Build the consensus error/revision panel (needed for 6.5); horizons 0,1,2
# ---------------------------------------------------------------------------
def consensus_panel(m, realized, order, rank):
    m3 = m[m["h"].isin([0, 1, 2])]
    cons = m3.groupby(["round", "target", "h"])["point"].mean().rename("cons").reset_index()
    cons["rank"] = cons["round"].map(rank)
    cons = cons.sort_values(["target", "rank"])
    cons["prev"] = cons.groupby("target")["cons"].shift(1)
    cons["prank"] = cons.groupby("target")["rank"].shift(1)
    cons["rev"] = np.where(cons["rank"] - cons["prank"] == 1, cons["cons"] - cons["prev"], np.nan)
    cons = cons.merge(realized.rename("real"), left_on="target", right_index=True, how="left")
    cons["err"] = cons["real"] - cons["cons"]
    return cons.dropna(subset=["err", "rev"]).sort_values("rank").reset_index(drop=True)


# ===========================================================================
# 6.4  ANCHORING OF LONG-TERM INFLATION EXPECTATIONS
# ===========================================================================
def section_64(m, realized):
    lt = m[m["h"] >= 4].copy()                            # ~5-year-ahead question
    g = lt.groupby("round")["point"]
    s = pd.DataFrame({"consensus": g.mean(), "sd": g.std(), "n": g.count()}).reset_index()
    s["yq"] = s["round"].map(yq)
    s = s.sort_values("yq")
    pre = s[s["yq"] < 2021]["consensus"].mean()
    surge = s[(s["yq"] >= 2021) & (s["yq"] < 2024)]["consensus"].mean()
    print("\n[6.4] Long-term (5y-ahead) anchoring")
    print(f"   consensus: pre-2021 {pre:.2f}, 2021-2023 {surge:.2f}, "
        f"range {s.consensus.min():.2f}-{s.consensus.max():.2f}")
    print(f"   disagreement (SD): pre-2021 {s[s.yq<2021].sd.mean():.2f}, "
        f"peak {s.sd.max():.2f} in {s.loc[s.sd.idxmax(),'round']}")

    yrs = [y for y in range(2000, 2026) if y in realized.index]
    plt.figure(figsize=(8, 4.6))
    plt.axhline(2.0, color="0.55", lw=1.0, ls=":", label="ECB 2% target")
    plt.plot(yrs, [realized[y] for y in yrs], "o-", color="#c44", lw=2, ms=4, label="Realized HICP inflation")
    plt.fill_between(s["yq"], s["consensus"] - s["sd"], s["consensus"] + s["sd"],
                    color="#3b6ea5", alpha=0.15, label="Long-term consensus ± 1 SD")
    plt.plot(s["yq"], s["consensus"], "-", color="#3b6ea5", lw=2.2, label="SPF long-term (5y-ahead) consensus")
    plt.xlabel("year"); plt.ylabel("annual HICP inflation (%)")
    plt.title("Realized inflation vs. anchored long-term SPF expectations")
    plt.xlim(1999, 2026); plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig("cg_anchoring.png", dpi=140); plt.close()


# ===========================================================================
# 6.5  TIME VARIATION IN THE COEFFICIENT (ROLLING BETA)
# ===========================================================================
def section_65(er, order, window=40):
    res = []
    for end in range(window - 1, len(order)):
        win = set(order[end - window + 1:end + 1])
        d = er[er["round"].isin(win)]
        if len(d) < 25:
            continue
        mm = sm.OLS(d["err"], sm.add_constant(d[["rev"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        res.append((yq(order[end]), mm.params["rev"], mm.bse["rev"]))
    rb = pd.DataFrame(res, columns=["yq", "beta", "se"])
    print("\n[6.5] Rolling beta (10-year windows)")
    print(f"   range {rb.beta.min():.2f} to {rb.beta.max():.2f}, "
        f"final {rb.beta.iloc[-1]:.2f}, {len(rb)} windows")
    plt.figure(figsize=(8, 4.6))
    plt.axhline(0, color="0.7", lw=.8)
    plt.axhline(0.812, color="#3b6ea5", lw=1, ls=":", label="full-sample β = 0.81")
    plt.axvline(2020, color="#c44", lw=1.2, ls="--", label="2020")
    plt.fill_between(rb.yq, rb.beta - 1.645 * rb.se, rb.beta + 1.645 * rb.se,
                    color="#3b6ea5", alpha=.15, label="90% confidence band")
    plt.plot(rb.yq, rb.beta, "-", color="#204060", lw=2, label="rolling β (10-year window)")
    plt.xlabel("window end (year)"); plt.ylabel("CG coefficient β")
    plt.title("Rolling estimate of the CG coefficient β (10-year windows)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("cg_rolling_beta.png", dpi=140); plt.close()


# ===========================================================================
# 6.6  FORECASTER DISAGREEMENT
# ===========================================================================
def section_66(m, realized):
    h1 = m[m["h"] == 1]
    dis = h1.groupby("round")["point"].std().rename("sd").reset_index()
    dis["yq"] = dis["round"].map(yq)
    dis = dis.sort_values("yq")
    dis["ry"] = dis["round"].str[:4].astype(int)
    dis["abschg"] = dis["ry"].map(realized.diff().abs())
    corr = dis[["sd", "abschg"]].dropna().corr().iloc[0, 1]
    print("\n[6.6] Forecaster disagreement (1-year-ahead)")
    print(f"   pre-2021 {dis[dis.ry<2021].sd.mean():.2f}, "
        f"2021-2023 {dis[(dis.ry>=2021)&(dis.ry<=2023)].sd.mean():.2f}, "
        f"peak {dis.sd.max():.2f} in {dis.loc[dis.sd.idxmax(),'round']}")
    print(f"   corr(disagreement, |change in realized inflation|) = {corr:.2f}")
    plt.figure(figsize=(8, 4.6))
    plt.plot(dis.yq, dis.sd, "-", color="#3b6ea5", lw=2, label="Disagreement (SD of 1y-ahead forecasts)")
    plt.axvspan(2021, 2023, color="#c44", alpha=.08, label="2021-2023 surge")
    plt.xlabel("year"); plt.ylabel("cross-forecaster standard deviation (pp)")
    plt.title("Forecaster disagreement about one-year-ahead inflation")
    plt.legend(fontsize=8, loc="upper left"); plt.tight_layout()
    plt.savefig("cg_disagreement.png", dpi=140); plt.close()


# ===========================================================================
# 6.7  INDIVIDUAL-LEVEL CG TEST
# ===========================================================================
def section_67(m, realized, rank):
    mi = m[m["h"].isin([0, 1, 2])].dropna(subset=["point"]).copy()
    mi["rank"] = mi["round"].map(rank)
    mi = mi.sort_values(["forecaster_id", "target", "rank"])
    g = mi.groupby(["forecaster_id", "target"])
    mi["prev"] = g["point"].shift(1)
    mi["prank"] = g["rank"].shift(1)
    mi["rev"] = np.where(mi["rank"] - mi["prank"] == 1, mi["point"] - mi["prev"], np.nan)
    mi = mi.merge(realized.rename("real"), left_on="target", right_index=True, how="left")
    mi["err"] = mi["real"] - mi["point"]
    di = mi.dropna(subset=["err", "rev"]).copy()

    def twoway(y, x, d, g1, g2):
        groups = np.column_stack([d[g1].astype("category").cat.codes,
                                d[g2].astype("category").cat.codes])
        return sm.OLS(d[y], sm.add_constant(d[[x]])).fit(
            cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})

    print("\n[6.7] Individual-level CG test")
    print(f"   observations {len(di)}, forecasters {di.forecaster_id.nunique()}")
    mp = twoway("err", "rev", di, "forecaster_id", "round")
    print(f"   pooled individual:        beta={mp.params['rev']:.3f} "
        f"(se {mp.bse['rev']:.3f}, p {mp.pvalues['rev']:.3f})")
    di["err_d"] = di["err"] - di.groupby("forecaster_id")["err"].transform("mean")
    di["rev_d"] = di["rev"] - di.groupby("forecaster_id")["rev"].transform("mean")
    mfe = twoway("err_d", "rev_d", di, "forecaster_id", "round")
    print(f"   with forecaster fixed FX: beta={mfe.params['rev_d']:.3f} "
        f"(se {mfe.bse['rev_d']:.3f}, p {mfe.pvalues['rev_d']:.3f})")
    print("   (consensus beta for comparison = 0.812)")


# ---------------------------------------------------------------------------
def main():
    m = load_spf(SPF_ZIP)
    realized = load_realized(EUROSTAT_GZ)
    order = sorted(m["round"].unique(), key=lambda r: (int(r[:4]), int(r[5])))
    rank = {r: i for i, r in enumerate(order)}
    er = consensus_panel(m, realized, order, rank)

    section_64(m, realized)
    section_65(er, order)
    section_66(m, realized)
    section_67(m, realized, rank)
    print("\nSaved figures: cg_anchoring.png, cg_rolling_beta.png, cg_disagreement.png")


if __name__ == "__main__":
    main()
