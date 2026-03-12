# Alpine-GP BayBE Optimizer (Streamlit)

## 1) Project Overview

This project provides a simple web app for **Bayesian optimization of chemical reactions**.

In plain terms, it helps you:
- define a reaction design space (solvent, catalyst, temperature, etc.),
- generate suggested experiments,
- enter measured results,
- and iteratively improve reaction conditions with BayBE.

The interface is built with **Streamlit**, so you can use it from a browser without writing Python code during routine use.

---

## 2) Installation

### Step A: Install Python
1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/).
2. Confirm installation in a terminal:

```bash
python --version
```

### Step B (recommended): Create a virtual environment
From inside the `alpine_GP_app` folder:

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell)**
```bash
.\.venv\Scripts\Activate.ps1
```

- **macOS/Linux**
```bash
source .venv/bin/activate
```

### Step C: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3) Running the App

From the `alpine_GP_app` directory:

```bash
streamlit run app.py
```

What you should see:
- A browser tab opens with the Alpine-GP app.
- A left sidebar with navigation pages:
  1. Configure
  2. Initialize
  3. Recommend
  4. Ingest Results
  5. History
  6. Campaign Dashboard
- Storage controls for your working directory and campaign selection.

---

## 4) Creating a New Optimization Campaign

1. **Choose a working directory (WORKDIR)** in the sidebar.
   - This is where all files are saved.
2. Go to **1) Configure**:
   - define parameter names/types/values,
   - set objective target column (typically `yield`),
   - set optimization direction (usually maximize).
3. Go to **2) Initialize**:
   - choose `sobol` or `existing_data` initialization,
   - generate initial experiments.
4. Go to **3) Recommend** to generate the next batch.

Files created in WORKDIR:
- `plans/runN.csv` for suggested experiments,
- `results/runN_results.csv` for ingested outcomes,
- `results/all_runs.csv` combined run history,
- `campaign_jsons/<campaign>_latest.json` active campaign state,
- `campaign_jsons/<campaign>_*.json` snapshots.

---

## 5) Resuming an Existing Campaign

1. In the sidebar, use the **Campaign browser**.
2. Select a campaign from the dropdown (`campaign name + timestamp`).
3. Review metadata shown below the dropdown.
4. Click **Load selected campaign**.
5. Continue using Recommend/Ingest pages.

The app keeps the loaded campaign in session state and marks it as active.

---

## 6) Entering Experimental Results

Prepare a CSV with:
- all parameter columns used in the campaign, and
- the objective column (for example `yield`).

Example:

```csv
solvent,catalyst,temp,yield
MeCN,A,25,0.63
HFIP,A,25,0.71
```

### Important yield format
**Yields must be fractions between 0 and 1.**

- 63% → `0.63`
- 91% → `0.91`

Invalid examples:
- `63`
- `120`
- `-5`

The app now strictly validates this and rejects out-of-range entries.

---

## 7) Typical Workflow

1. Initialize campaign  
2. Run suggested experiments in the lab  
3. Measure yields  
4. Enter yields into CSV (fraction format 0–1)  
5. Upload/ingest results  
6. Generate next experiment suggestions  
7. Repeat until performance is satisfactory

---

## 8) Troubleshooting

### Issue: “yield outside [0,1]”
Cause: Results entered as percentages (e.g., `63`) instead of fractions (`0.63`).
Fix: Convert all yield entries to values between 0 and 1.

### Issue: Configuration mismatch warning
Cause: Current UI settings do not match the loaded campaign design space/objective.
Fix:
- Load the correct campaign from the campaign browser, or
- update parameter/objective settings to match.

### Issue: Missing required CSV columns
Cause: CSV does not include one or more parameter columns or target column.
Fix: Ensure column names exactly match configured parameter and target names.

### Issue: No campaigns in browser
Cause: No JSON campaign files found in `WORKDIR/campaign_jsons/`.
Fix: Verify WORKDIR and confirm a campaign has been initialized previously.

---

## 9) Campaign Dashboard (tab 6)

The **Campaign Dashboard** tab provides package-agnostic campaign analysis from a trials table (`results/all_runs.csv`).
It does not rely on optimizer-specific APIs and works from dataframe columns only.

### Expected columns

The dashboard supports trials data containing any subset of:

- `trial_index`
- `yield` (or another numeric objective column)
- `status`
- `round_index`
- `batch_index`
- `run_index`
- `sem`
- `campaign`
- `timestamp`

All non-bookkeeping columns are treated as optimization parameters and analyzed automatically.

### Dashboard controls

At the top of the tab, users can:

- choose the objective column,
- switch between maximize/minimize mode,
- filter to completed trials,
- select which parameters to include in parameter-level analysis.

### What the plots mean

- **Campaign progress**: objective vs trial index plus best-so-far trend.
- **Objective distribution**: histogram (or scatter-style view for very small datasets).
- **Status distribution**: count of trials per status if a `status` column is present.
- **Top trials table**: best rows sorted by objective in chosen direction.
- **Categorical analysis**: mean objective by category with counts and summary table.
- **Numerical analysis**: parameter value vs objective scatter, with binned mean overlay.
- **Round/Batch analysis**: objective distribution by `round_index`, `batch_index`, or `run_index` if present.

### Exports

- Figures are auto-saved as PNG files under `WORKDIR/plots/<campaign_name>/analysis/`.
- Cleaned dashboard data can be downloaded as CSV.
- Dashboard summary statistics can be downloaded as JSON.
