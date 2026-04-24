# Alpine-GP — Claude Working Document

## Project Purpose

Alpine-GP is a Bayesian optimization platform for chemistry. It surfaces a Streamlit GUI over a pluggable backend optimization engine (BayBE or Ax). The core mandate is scientific integrity: every UI control must map to real backend logic, every output must be reproducible, and no placeholder or faked behavior is permitted.

**If it appears in the GUI, it must be implemented and testable. No exceptions.**

---

## Architecture Overview

```
app.py                         — Streamlit UI
core/
  schema.py                    — CampaignConfig (engine field: "baybe"|"ax") + ParameterSpec dataclasses
  campaign_engine.py           — CampaignEngine ABC + factory functions (route by engine type)
  baybe_engine.py              — BayBEEngine (implements CampaignEngine)
  ax_engine.py                 — AxEngine (implements CampaignEngine via AxClient)
  baybe_factory.py             — BayBE object construction + validation (engine-aware)
  persistence.py               — File I/O helpers
  sobol_init.py                — Sobol initial-design generator
  dedup.py                     — Duplicate-point detection
  campaign_dashboard.py        — Dashboard rendering helpers
tests/
  test_campaign_engine.py      — Engine interface + save/load/ingest tests
  test_truthful_controls.py    — Validation + wiring tests
  test_trial_status.py         — Trial status system tests (10 tests)
  test_model_visibility.py     — Model predictions tests (9 tests)
  test_ax_engine.py            — AxEngine tests (29 tests)
```

---

## Non-Negotiable Principles (from AGENT_PRINCIPLES.md)

1. **Truthful systems only** — no placeholder features, ignored parameters, or silent fallbacks.
2. **Scientific integrity** — all outputs physically meaningful, statistically valid, derived from real computation.
3. **No embellishment** — UI must not suggest capabilities the backend lacks.
4. **UI ↔ Backend consistency** — every control maps to a backend parameter deterministically.
5. **Data integrity** — no silent type coercion, no dropped rows without error; validation is mandatory.
6. **Reproducibility** — every campaign must be reconstructible from config + data + model state.
7. **Testing** — every feature needs unit + integration + UI-consistency tests.

**Definition of Done**: GUI control exists AND backend implementation exists AND data persists correctly AND tests exist AND behavior is verifiable.

---

## Roadmap Status

### Phase 1: Backend Abstraction Layer — COMPLETE
- `CampaignEngine` ABC in `core/campaign_engine.py` with all 7 methods.
- Factory functions route by `cfg.engine` and by `_engine` marker in saved JSON.
- Tests in `test_campaign_engine.py`.

### Phase 2: BayBE Engine — COMPLETE
- Supports continuous, discrete, categorical, and substance parameters.
- Acquisition function registry with 8 functions, batch-size constraints, UCB beta wiring.
- `ingest()` filters to completed rows only; failed/abandoned/partial/invalid excluded from model.
- Tests in `test_truthful_controls.py`.

### Phase 2: Ax Engine — COMPLETE
- `core/ax_engine.py` — full `AxEngine` implementing all 7 `CampaignEngine` methods.
- `engine: Literal["baybe", "ax"]` field in `CampaignConfig`; persisted in saved JSON.
- Factory routing: `create_campaign_engine` and `load_campaign_engine` route by engine type + `_engine` marker.
- `validate_campaign_config` skips acquisition validation for Ax.
- Configure page: engine selector; BayBE acquisition controls hidden when Ax selected.
- SubstanceSpec limitation (no molecular encoding) surfaced in UI info message.
- `acq_score` proxy documented: Ax has no direct per-candidate acquisition API; uses `pred_mean` adjusted for direction.
- Tests: `test_ax_engine.py` — 29 tests.

### Phase 3: Trial Status System — COMPLETE
- `VALID_TRIAL_STATUSES` frozenset in `schema.py`.
- `ingest()` raises `ValueError` if `status` column absent; filters by status before model update.
- `st.data_editor` with `SelectboxColumn` for per-row status editing in UI.
- Tests: `test_trial_status.py` — 10 tests.

### Phase 4: Model Visibility — COMPLETE
- `CampaignEngine.predict()` + `ModelNotFittedError` in `core/campaign_engine.py`.
- BayBE: `Campaign.posterior_stats()` + `Campaign.acquisition_values()`.
- Ax: `get_model_predictions_for_parameterizations()`.
- Sortable model output table; degrades gracefully with info message when model not fitted.
- Tests: `test_model_visibility.py` — 9 tests.

**Total: 58 tests, all passing.**

### Phase 5: Multi-Objective Optimization — NOT STARTED

### Phase 6: Search Space Editing — NOT STARTED

### Phase 7: Deduplication Hygiene — PARTIALLY DONE
- Exact-duplicate blocking via `dedup.py` + retry loop in `render_recommend_page`.
- Near-duplicate detection and user-facing warnings not yet implemented.

---

## Execution Plan (priority order)

### Next: Phase 5 — Multi-Objective (Ax only)
- Extend `CampaignConfig` to hold a list of target specs.
- Ax engine: multi-objective support.
- BayBE: disable with explicit UI label (not silent fallback).
- Pareto front display in dashboard.

### Later: Phase 6 — Search Space Editing
- Snapshot-before-mutate; append-only history.

### Later: Phase 7 (remainder) — Dedup Hygiene
- Near-duplicate detection + user warning UI.

---

## Key Invariants to Preserve

- `CampaignEngine` is the only interface `app.py` touches for engine operations; never import `BayBEEngine` or `AxEngine` directly in the UI.
- The `_engine` key in the campaign JSON determines which loader is used; preserve it on every save.
- `validate_campaign_config` must be called before any `build_campaign` call; errors must surface to the user, not be swallowed.
- Tests use module-level stubs so BayBE/Ax are not required to run the test suite.
- `all_runs.csv` is the source of truth for deduplication; the campaign JSON is the source of truth for model state.
