# BayBE Optimizer GUI (Streamlit)

This is a **worked example** Streamlit GUI that wraps a BayBE + BoTorch backend for lab-friendly Bayesian optimization.

It implements the same core loop as your notebook:

1. **Recommend** a batch → write `plans/runN.csv`
2. Run experiments
3. Copy plan → `results/runN_results.csv`, add a `yield` column
4. **Ingest** results (validates columns) → `campaign.add_measurements(...)`
5. Repeat

## Folder layout (inside WORKDIR)

- `campaign_config.json` (human-editable)
- `plans/run0.csv`, `plans/run1.csv`, ...
- `results/run0_results.csv`, ...
- `results/all_runs.csv` (append-only log across all runs)
- `campaign_jsons/<campaign>_latest.json` (mutable state)
- `campaign_jsons/<campaign>_after_*.json` (immutable snapshots)

## Running

```bash
cd alpine_GP_app
streamlit run app.py
```

In the sidebar you can set **WORKDIR** (e.g., a shared network folder) and the **campaign name**.

## Parameter types and encodings

### Numerical continuous
- UI: lower/upper bounds
- BayBE: `NumericalContinuousParameter(bounds=(lo, hi))`

### Numerical discrete
- UI: explicit list of values
- BayBE: `NumericalDiscreteParameter(values=[...])`

### Categorical
- UI: explicit list of categories + encoding
- BayBE: `CategoricalParameter(values=[...], encoding=OHE|INT)`

### Substance (cheminformatics)
- UI: map **labels → SMILES** + an embedding/descriptor option
- BayBE: `SubstanceParameter(data={label: smiles, ...}, encoding=SubstanceEncoding.<...>)`

BayBE exposes many descriptor/fingerprint options (e.g., **MORDRED**, **RDKIT2DDESCRIPTORS**, **ECFP**, **MACCS**). Under the hood these route through `scikit-fingerprints` / RDKit-compatible tooling (see BayBE docs).

## Acquisition functions

The GUI lets you choose BayBE acquisition wrappers:

- `qExpectedImprovement` (≈ qEI)
- `qUpperConfidenceBound` (≈ qUCB)
- `qThompsonSampling` (≈ qTS)
- plus a few single-point variants.

Internally we call `baybe.acquisition.utils.str_to_acqf(...)`. In the currently pinned BayBE version, extra JSON kwargs are ignored because `str_to_acqf` only accepts the acquisition name.

## Notes / limitations

- The **Sobol init** here is pragmatic: it uses `scipy.stats.qmc.Sobol` and indexes categorical/substance values by the Sobol coordinate. For many chemistry problems this is “good enough” to seed the model with diverse conditions; if you want a more principled discrete DOE (e.g., FPS / maximin), we can swap in BayBE’s space-filling recommenders.
- Deleting a parameter deletes it from the config; you should **re-initialize** the campaign if the parameter set changes.

