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

### Step A (recommended): Install Anaconda
Using Anaconda is the recommended way to run this app because it gives you an isolated, reproducible Python environment without manually managing system Python.

1. Download Anaconda from the official installer page: [Anaconda Distribution](https://www.anaconda.com/download).
2. Run the installer for your operating system.
3. Open a terminal (**Anaconda Prompt** on Windows, Terminal on macOS/Linux).
4. Confirm Conda is available:

```bash
conda --version
```

### Step B: Create and activate a dedicated Conda environment
From any terminal location, create a new environment with a supported Python version:

```bash
conda create -n alpine-gp python=3.10 -y
```

Activate it:

- **Windows (Anaconda Prompt / PowerShell with Conda initialized)**
```bash
conda activate alpine-gp
```

- **macOS/Linux**
```bash
conda activate alpine-gp
```

Optional sanity check:

```bash
python --version
```

### Step C: Install project dependencies
Navigate to the app folder and install requirements into the active Conda environment:

```bash
cd alpine_GP_app
pip install -r requirements.txt
```

### Step D: Re-activate the environment in future sessions
Whenever you open a new terminal, activate the same environment before running the app:

```bash
conda activate alpine-gp
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
   - set optimization direction (usually maximize),
   - choose the **backend engine** (BayBE or Ax — see Section 10 below).
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
- all parameter columns used in the campaign,
- the objective column (for example `yield`), and
- optionally a `status` column (see below).

Minimal example:

```csv
solvent,catalyst,temp,yield
MeCN,A,25,0.63
HFIP,A,25,0.71
```

Example with a status column (recommended for mixed-outcome batches):

```csv
solvent,catalyst,temp,yield,status
MeCN,A,25,0.63,completed
HFIP,A,25,,failed
EtOH,B,40,0.45,completed
```

### Trial status (required for every ingestion)

Every row you ingest must have a **trial status**. This determines whether the row is used to update the model.

| Status | Meaning | Used in model? |
|--------|---------|----------------|
| `completed` | Valid experimental result | **Yes** |
| `failed` | Experiment attempted but produced no usable data | No |
| `abandoned` | Experiment not completed | No |
| `partial` | Incomplete data | No |
| `invalid` | Data corrupted or unusable | No |

**How status is assigned on the Ingest Results page:**

- If your CSV already has a `status` column, the app uses those values (you can still edit individual rows in the table shown).
- If your CSV has no `status` column, a selectbox appears so you can assign a status to all rows at once. You can then change individual rows in the editable table before ingesting.

For disk-path ingestion, a status selectbox is shown before the ingest button. If the file already has a `status` column, that setting is ignored.

**Rows with any status other than `completed` are recorded in `all_runs.csv` for traceability but are never passed to the optimization model.**

### Important yield format
**Yields must be fractions between 0 and 1** for completed rows.

- 63% → `0.63`
- 91% → `0.91`

Invalid examples:
- `63`
- `120`
- `-5`

The app strictly validates this and rejects out-of-range entries for completed rows. Non-completed rows may have a blank or missing yield value.

---

## 7) Model Output on the Recommend Page

After the first batch of results has been ingested, the **Recommend** page shows a model output table alongside the suggested experiments.

### Columns

| Column | Meaning |
|--------|---------|
| `pred_mean` | Surrogate posterior mean for the target — the model's best estimate of the yield you would observe |
| `pred_std` | Posterior standard deviation — how uncertain the model is about that prediction |
| `acq_score` | Raw acquisition function value — the score the optimizer used to rank candidates |
| `rank` | 1-indexed rank by acquisition score; **rank 1 is the most recommended candidate** |

The table is pre-sorted by rank so the best candidates appear first. You can click any column header to re-sort.

### Downloads

Two download buttons are shown:
- **Plan CSV (for lab)** — clean parameter columns only; this is what you take to the bench.
- **Model output CSV** — same rows with all four model columns attached; useful for record-keeping and analysis.

### When predictions are not available

During the initial Sobol phase (before any results are ingested) the surrogate model has not yet been fitted. The page shows a plain candidate table with a note: *"Model predictions not available — ingest at least one completed measurement first."* No placeholder or fabricated values are ever shown.

---

## 8) Typical Workflow

1. Initialize campaign (Sobol or warm-start)
2. Run suggested experiments in the lab
3. Record outcomes — note which experiments completed, which failed or were abandoned
4. Enter results into CSV (yield as fraction 0–1; optionally add a `status` column)
5. Upload/ingest results — assign status per row in the editable table
6. Generate next experiment suggestions — model output table now shows predictions
7. Repeat until performance is satisfactory

---

## 8) Troubleshooting

### Issue: “Missing required 'status' column”
Cause: A CSV was passed to the engine without a `status` column, bypassing the UI.
Fix: Always use the Ingest Results page in the app, which injects the status column automatically. If scripting directly, add a `status` column to your DataFrame before calling `engine.ingest()`.

### Issue: “Invalid trial status values”
Cause: The `status` column in your CSV contains a value that is not one of the five supported statuses.
Fix: Use only: `completed`, `failed`, `abandoned`, `partial`, `invalid`. Values are case-insensitive.

### Issue: Model is not improving despite ingested results
Cause: All ingested rows may have a non-completed status and so were excluded from model fitting.
Fix: Check `results/all_runs.csv` — look at the `status` column. Only rows with `status=completed` update the model. The ingest confirmation message shows how many rows were added vs. excluded.

### Issue: No model output table on the Recommend page
Cause: The surrogate has not been fitted yet (no completed measurements ingested).
Fix: Ingest at least one run with `status=completed` before requesting the next recommendation. The model output table appears automatically once the surrogate has data to train on.

### Issue: `pred_mean` / `pred_std` are missing from the model output CSV
Cause: The campaign was loaded from an older save that pre-dates Phase 4, or the surrogate was not fitted at save time.
Fix: Ingest at least one completed result and generate a new recommendation. The model output CSV is always recomputed from the current model state.

### Issue: “yield outside [0,1]”
Cause: Results entered as percentages (e.g., `63`) instead of fractions (`0.63`).
Fix: Convert all yield entries to values between 0 and 1. This check applies only to completed rows.

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

## 10) Choosing a Backend Engine

The Configure page lets you select the optimization engine. Both engines implement the same interface (recommend, ingest, predict, save, load).

### BayBE (default)

- Full acquisition function selection (qEI, UCB, qNEI, etc.)
- UCB beta parameter for exploration tuning
- Molecular encoding for substance parameters (MORDRED, ECFP, etc.)
- Suitable for chemical spaces with structural diversity

### Ax (ax-platform)

- Ax manages the generation strategy and acquisition function internally; no manual selection required
- Sobol initialization, then automatic BoTorch Bayesian optimization
- Supports continuous, discrete, and categorical parameters
- **SubstanceSpec limitation**: SMILES are treated as opaque categorical labels — no molecular descriptor encoding is applied. If your search space contains chemical structures that benefit from molecular similarity, use the BayBE engine.
- `acq_score` in the model output table is a proxy value (predicted mean adjusted for optimization direction) because Ax does not expose per-candidate acquisition function scores for arbitrary points

### Engine and acquisition notes

- The engine is persisted in the campaign JSON. A campaign started with BayBE cannot be loaded as Ax, and vice versa. The `_engine` key in the saved JSON determines which loader is used.
- When using Ax, the acquisition function selector and UCB beta slider on the Configure page are hidden — they have no effect on Ax.

---

## AI Development Rules

All AI-assisted development MUST follow:
- AGENT_PRINCIPLES.md
- AGENT_ROADMAP.md

Failure to comply results in invalid implementation.