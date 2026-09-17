# Inflation Expectations and Monetary Policy Credibility after 2020

Replication package for the bachelor thesis *"Inflation Expectations and
Monetary Policy Credibility after 2020 — A Coibion-Gorodnichenko Test of the
ECB Survey of Professional Forecasters before and after 2020"* (Goethe
University Frankfurt, Professorship of Monetary Economics, Prof. Wieland,
Ph.D., 2026).

The thesis applies the Coibion and Gorodnichenko (2015) predictability test
to the ECB Survey of Professional Forecasters (SPF) to test whether the
euro-area consensus under-reacts to news (H1), whether that under-reaction
changed after the 2021–2023 inflation surge (H2), and whether it differs
between the consensus and individual forecasters.

## Methodology

- Fixed-event forecast errors and revisions (current-year, one-year-ahead,
  two-year-ahead horizons)
- Newey-West HAC and two-way cluster-robust inference
- Pre/post-2020 structural-break and rolling-window analysis
- Huber M-estimation, median regression and cluster bootstrap
- Individual-level panel regression with forecaster fixed effects
- Stacked interaction regression to test the consensus/individual wedge

## Data

Two raw inputs are required (not included in this repository, see
`data/README.md` for retrieval instructions):

| File | Source | Content |
|---|---|---|
| `Rates_BA_1.zip` | ECB SPF microdata | Individual HICP point forecasts, all 111 survey rounds, 1999Q1–2026Q3, for the current-year, one-year-ahead and two-year-ahead calendar-year targets |
| `Rates_BA_2_csv.gz` | Eurostat, dataset `prc_hicp_aind` | Realised annual HICP inflation, euro area of changing composition |

Each survey round contains up to three forecast horizons per forecaster; two
consecutive rounds for the same target year are required to construct a
revision, which is why the usable estimation sample (255 consensus
observations) is smaller than the raw number of survey responses.

## Requirements

- Python 3.12
- pandas, NumPy, SciPy, statsmodels, matplotlib

Install dependencies:

```
pip install pandas numpy scipy statsmodels matplotlib
```

## Installation

1. Clone the repository
2. Place `Rates_BA_1.zip` and `Rates_BA_2_csv.gz` in the repository root
   (see `data/README.md`)
3. Install dependencies (see above)

## Project Structure

```
.
├── cg_spf_replication.py   # Entry point. Builds the consensus sample (255 obs),
│                            #   estimates the baseline CG regression, the
│                            #   pre/post-2020 split, and the robustness checks
│                            #   (median consensus, COVID exclusion, decomposition,
│                            #   alternative break dates). Produces Tables 1-4 and
│                            #   Figures 1-3.
├── cg_extensions.py         # Section 6.4-6.7 extensions: long-term anchoring
│                            #   analysis, rolling-window beta, forecaster
│                            #   disagreement, and the individual-level panel
│                            #   regression (pooled and with forecaster fixed
│                            #   effects). Produces Table 5 and Figures 4-6.
├── cg_wedge_test.py         # Formal test of the consensus/individual wedge
│                            #   (Section 6.7): stacked interaction regression
│                            #   (delta = beta_consensus - beta_individual) with
│                            #   cluster-robust and two-way cluster-robust
│                            #   standard errors, plus a paired cluster bootstrap
│                            #   over survey rounds. Adds two rows to Table 5/6.
├── data/
│   └── README.md            # Where to obtain the two raw data files
└── README.md
```

## Running the Code

Each script is self-contained and can be run independently from the
repository root once the two raw data files are in place:

```
python cg_spf_replication.py
python cg_extensions.py
python cg_wedge_test.py
```

There is no single combined entry point at present; the three scripts are
run in the order above to reproduce, respectively, the baseline results, the
extension analyses, and the wedge test. All estimation is deterministic
except the two bootstrap procedures (in `cg_extensions.py` and
`cg_wedge_test.py`), which use a fixed random seed and therefore reproduce
identical results on every run.

## About

Bachelor thesis, Goethe University Frankfurt, Faculty of Economics and
Business (Department 02), Professorship of Monetary Economics.
Submitted by Carlo Rappenecker, 23.09.2026.
