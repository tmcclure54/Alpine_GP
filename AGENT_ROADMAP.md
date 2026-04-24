# Alpine-GP Development Roadmap
## Feature Goals and Implementation Plan

---

> **Status key**
> - ✅ COMPLETE — implemented, tested, merged
> - ⚠️ PARTIAL — partially implemented; gaps noted
> - ❌ NOT STARTED — no code exists yet

---

## Objective

Elevate Alpine-GP from a BayBE interface into a:
- Chemically aware optimization platform
- Experimentally realistic system
- Scientifically robust tool suitable for publication

---

## Phase 1: Core Infrastructure — ✅ COMPLETE

### 1. Backend Abstraction Layer

Create unified interface:

```python
class CampaignEngine:
    def recommend(self, batch_size: int): ...
    def ingest(self, df): ...
    def save(self, path): ...
    def load(self, path): ...
```

**Done:**
- `CampaignEngine` ABC defined in `core/campaign_engine.py` with all required methods plus `cached_recommendation_info()` and `measurement_count()`.
- Factory functions `create_campaign_engine` and `load_campaign_engine` decouple `app.py` from concrete engine classes.
- `extract_saved_campaign_metadata` provides engine-agnostic campaign inspection.
- Tests: `tests/test_campaign_engine.py` — create, save/load roundtrip, ingest normalization, metadata extraction.

---

# Phase 2: Engine Implementations (BayBE + Ax) — ✅ COMPLETE

## Engine Implementations

---

### BayBEEngine — ✅ COMPLETE

#### Responsibilities

- Construct BayBE campaign
- Generate recommendations
- Ingest completed experiments
- Ignore failed/abandoned trials in training

**Done:**
- `BayBEEngine` in `core/baybe_engine.py` fully implements `CampaignEngine`.
- Supports all four parameter types: continuous, discrete, categorical, substance.
- Acquisition function registry (`ACQUISITION_SPECS`) in `core/baybe_factory.py` covers 8 functions with batch-size constraints, kwarg validation, and beta wiring.
- `ingest()` filters to `status == "completed"` rows only before calling `add_measurements`; failed/abandoned/partial/invalid rows are recorded but excluded from model fitting.
- Full parameter validation (`validate_parameter_specs`, `validate_campaign_config`, `validate_config_payload`) raises loudly on invalid input.
- Tests: `tests/test_truthful_controls.py` — UCB beta wiring, substance decorrelate, encoding validation, unsupported kwarg rejection.

---

#### Requirements

- Must use actual BayBE API ✅
- Must support:
  - continuous parameters ✅
  - discrete parameters ✅
  - categorical parameters ✅
  - substance parameters ✅

---

#### Behavior Rules

- Failed / abandoned trials:
  - Must NOT be included in model fitting ✅ (implemented in Phase 3 — status-filtered ingest)
- Partial data:
  - Must be rejected OR explicitly handled ✅ (partial status recorded, excluded from model)

---

### AxEngine — ✅ COMPLETE

#### Responsibilities

- Create Ax experiment
- Manage trial lifecycle
- Handle trial statuses explicitly
- Generate recommendations via Ax API

**Done:**
- `core/ax_engine.py` implementing `CampaignEngine` with all 7 required methods.
- `engine: Literal["baybe", "ax"]` field added to `CampaignConfig` in `core/schema.py`.
- Factory functions `create_campaign_engine` and `load_campaign_engine` route by `cfg.engine` and by `_engine` marker in saved JSON.
- `extract_saved_campaign_metadata` detects Ax saves and calls `extract_ax_campaign_metadata`.
- `baybe_factory.py` `CAMPAIGN_CONFIG_KEYS` includes `"engine"`; acquisition validation skipped for Ax engine.
- Engine selector added to Configure page in `app.py`; BayBE-specific controls (acquisition, UCB beta, spectrum map) hidden when Ax is selected.
- SubstanceSpec limitation (no molecular encoding in Ax) surfaced as info message in UI.
- Tests: `tests/test_ax_engine.py` — 29 tests covering: from_config, recommend, ingest for all 5 status values, predict (Sobol phase + BO phase), save/load round-trip, measurement_count, cached_recommendation_info, warm-start attach, factory routing.

---

#### Trial Status Mapping (implemented)

| Internal Status | Ax Mapping |
|----------------|------------|
| completed      | `complete_trial()` with raw data ✅ |
| failed         | `log_trial_failure()` ✅ |
| abandoned      | `abandon_trial()` ✅ |
| partial        | `log_trial_failure()` ✅ |
| invalid        | `log_trial_failure()` ✅ |

---

#### Requirements

- Must use real Ax API (no simulation) ✅ — uses `AxClient` with `get_next_trials`, `complete_trial`, `log_trial_failure`, `abandon_trial`, `attach_trial`
- Must support:
  - trial creation ✅
  - trial status assignment ✅
  - data attachment ✅
  - candidate generation ✅

---

#### Behavior Rules

- Failed trials:
  - Must be recorded ✅
  - Must not be treated as valid observations ✅

- Abandoned trials:
  - Must not be re-suggested ✅

---

#### Known Limitations

- **No molecular encoding for SubstanceSpec**: Ax treats SMILES as opaque categorical labels. No descriptor/fingerprint encoding is applied. Users are warned in the Configure page UI.
- **`acq_score` is a proxy**: Ax has no direct API to return per-candidate acquisition values for arbitrary points. `acq_score` is set to `pred_mean` (adjusted for optimization direction). This is documented and surfaced as a proxy, not a fabricated value.

---

#### Forbidden

- Treating all trials as completed ✅ enforced
- Ignoring trial state ✅ enforced
- Using Ax without lifecycle tracking ✅ enforced


## Phase 3: Trial Status System — ✅ COMPLETE

---

### Objective

Accurately represent real experimental outcomes.

---

### Required Status Values

- completed ✅
- failed ✅
- abandoned ✅
- partial ✅
- invalid ✅

Defined as `VALID_TRIAL_STATUSES` (frozenset) and `TrialStatus` (Literal type) in `core/schema.py`.

---

### Definitions

- completed: valid experimental result
- failed: experiment attempted but did not produce usable data
- abandoned: experiment not completed
- partial: incomplete data
- invalid: data corrupted or unusable

---

### Requirements

- Status MUST be provided for every ingestion event ✅ — `BayBEEngine.ingest` raises `ValueError` if `status` column is absent
- Status MUST be stored persistently ✅ — returned df includes `status` column; written to `results/runN_results.csv` and `results/all_runs.csv`
- Status MUST influence backend behavior ✅ — only `completed` rows reach `add_measurements`

---

### Backend Behavior

#### BayBE
- completed → used in model ✅
- failed → ignored ✅
- abandoned → ignored ✅
- partial → ignored ✅
- invalid → ignored ✅

#### Ax
- Not yet implemented (see Phase 2 AxEngine)

---

### UI Requirements

- Status selection must be mandatory ✅ — `_inject_status_ui()` in `app.py` always produces a `status` column before ingest
- Status per-row editing supported ✅ — `st.data_editor` with `SelectboxColumn` allows row-level overrides
- Status must be visible in dashboard ✅ — `campaign_dashboard.py` already renders a status distribution bar chart when `status` column is present in `all_runs.csv`
- Status filtering must be supported ✅ — dashboard "Use completed trials only" checkbox filters by status

---

### Tests

- `tests/test_trial_status.py` — 10 tests covering: missing column, invalid value, all-completed, all-failed, mixed, abandoned/partial/invalid, return-all-rows, case normalisation, constant completeness, cumulative count.

---

### Forbidden

- Defaulting status silently ✅ enforced — when no status column in CSV, UI explicitly presents a selectbox; for disk-path ingestion, the assigned status is shown with `st.info`
- Treating all data as completed ✅ enforced
- Ignoring status in modeling ✅ enforced


## Phase 4: Model Visibility — ✅ COMPLETE

---

### Objective

Expose internal model predictions to the user.

---

### Required Outputs (per candidate)

- predicted mean ✅
- uncertainty (standard deviation) ✅
- acquisition score ✅
- ranking ✅

**Done:**
- `CampaignEngine.predict(candidates)` abstract method added to `core/campaign_engine.py`. Returns input df with `pred_mean`, `pred_std`, `acq_score`, `rank` columns attached.
- `ModelNotFittedError` (subclass of `ValueError`) defined in `core/campaign_engine.py`. Raised when no completed measurements have been ingested yet (Sobol phase).
- `BayBEEngine.predict()` in `core/baybe_engine.py` uses BayBE 0.14.3's real APIs:
  - `Campaign.posterior_stats(candidates, stats=("mean", "std"))` → `{target}_mean`, `{target}_std` columns
  - `Campaign.acquisition_values(candidates)` → per-candidate acquisition scores
  - Rank computed as `acq_score.rank(ascending=False, method="first")`; rank 1 = best candidate
  - Catches `baybe.exceptions.ModelNotTrainedError` and `NoMeasurementsError`, re-raises as `ModelNotFittedError`
- `_render_model_output()` helper in `app.py`:
  - Called from `render_recommend_page` for both cached active batches and newly generated batches
  - Shows sortable table ordered by rank
  - Provides two download buttons: clean plan CSV (for lab) and enriched model output CSV
  - If model not yet fitted: shows informative message, falls back to plain candidate table
- Tests: `tests/test_model_visibility.py` — 9 tests covering: required columns, parameter column preservation, rank-1 = max acq_score, all ranks unique, pre-measurement raises, non-constant pred_mean, non-negative pred_std, row count, error subclass.

---

### Requirements

- All values must come directly from backend ✅ — uses `Campaign.posterior_stats` and `Campaign.acquisition_values`
- No approximations ✅
- No fabricated values ✅

---

### UI Requirements

- Display as sortable table ✅ — `st.dataframe` sorted by rank ascending
- Allow filtering and ranking ✅ — sortable by any column in Streamlit
- Must update after each ingestion ✅ — page reload after ingest re-runs `engine.predict()` on the cached batch

---

### Forbidden

- Simulated predictions ✅ enforced — `ModelNotFittedError` raised rather than returning NaN/placeholder
- Placeholder values ✅ enforced
- Static or cached outputs presented as live ✅ enforced — predictions recomputed from current model state on each page load

## Phase 5: Multi-Objective Optimization — ❌ NOT STARTED

---

### Objective

Support optimization across multiple targets.

---

### Requirements

- Multiple targets must be definable in schema
- Each target must specify:
  - name
  - direction (maximize/minimize)

**Still needed:**
- Extend `CampaignConfig` to support a list of target specs.
- Ax engine: full multi-objective support.
- BayBE engine: disable multi-objective in UI with explicit label (not a silent fallback).
- Pareto front display in dashboard.

---

### Backend Support

#### Ax
- Full multi-objective support required ❌

#### BayBE
- If unsupported:
  - Feature must be disabled in UI ❌
  - Must NOT silently fallback to single-objective

---

### UI Requirements

- Display Pareto front
- Allow selection of trade-off points

---

### Forbidden

- Pretending multi-objective support exists
- Collapsing objectives without user knowledge

## Phase 6: Search Space Editing — ❌ NOT STARTED

---

### Objective

Allow modification of search space during active campaigns.

---

### Allowed Operations

- Add categorical values
- Remove categorical values
- Adjust numeric bounds

**Still needed:**
- Snapshot-before-mutate pattern for any search space modification.
- Timestamp + user-provided reason recorded per modification.
- Append-only history; historical configs remain accessible.

---

### Requirements

Every modification must:
- Create a snapshot
- Record timestamp
- Record user-provided reason

---

### Data Integrity Rules

- No in-place mutation without snapshot
- Historical configurations must remain accessible

---

### Forbidden

- Silent changes to search space
- Overwriting previous configurations
- Losing reproducibility

## Phase 7: Deduplication and Recommendation Hygiene — ⚠️ PARTIAL

---

### Objective

Prevent redundant or wasteful experiments.

**Done:**
- `core/dedup.py` — `measured_keys()` reads `all_runs.csv` and returns the set of previously measured parameter tuples. `drop_measured()` removes exact duplicates from a recommendation batch.
- `render_recommend_page` in `app.py` iterates up to 6 attempts to fill the requested batch size, dropping measured points each time.

**Still needed:**
- Near-duplicate detection for discrete grids (e.g., tolerance-based matching for numerical parameters).
- User-facing warning explaining why specific candidates were filtered.
- Intentional replicate override (allow user to request a duplicate deliberately).

---

### Requirements

- Block exact duplicates ✅
- Detect near-duplicates (for discrete grids) ❌
- Allow intentional replicates ❌

---

### Behavior

- Warn user before duplicate recommendation ❌ (silent removal currently)
- Provide explanation for filtering ❌

---

### Forbidden

- Silent removal of recommendations ⚠️ (exact duplicates silently dropped — warning not yet surfaced per-candidate)
- Recommending identical experiments without notice ✅ (blocked by dedup loop)

## Testing Requirements

---

### Unit Tests

- Parameter validation ✅ (`test_truthful_controls.py`)
- Engine recommend() ✅ (`test_campaign_engine.py`)
- Engine ingest() ✅ (`test_campaign_engine.py`, `test_trial_status.py`)
- Status handling ✅ (`test_trial_status.py` — 10 tests)
- Model predictions ✅ (`test_model_visibility.py` — 9 tests)
- AxEngine full coverage ✅ (`test_ax_engine.py` — 29 tests)

Total: 58 tests, all passing.

---

### Integration Tests

Full workflow:
- initialize ✅ (covered via engine create + save)
- recommend ✅
- ingest ✅
- repeat ✅ (cumulative ingest test in `test_trial_status.py`)
- predict after ingest ✅ (`test_model_visibility.py`)
- Ax full workflow ✅ (`test_ax_engine.py` — from_config → recommend → ingest → predict → save/load)

---

### UI Consistency Tests

For each UI control:
- Verify backend receives parameter ✅ (UCB beta, substance decorrelate)
- Verify behavior changes accordingly ✅
- Verify model output table appears after measurements ingested ✅

---

### Forbidden

- Untested features ✅ enforced so far
- Manual-only validation ✅ enforced so far