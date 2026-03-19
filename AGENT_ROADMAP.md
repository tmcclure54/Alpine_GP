# Alpine-GP Development Roadmap
## Feature Goals and Implementation Plan

---

## Objective

Elevate Alpine-GP from a BayBE interface into a:
- Chemically aware optimization platform
- Experimentally realistic system
- Scientifically robust tool suitable for publication

---

## Phase 1: Core Infrastructure (MANDATORY FIRST)

### 1. Backend Abstraction Layer

Create unified interface:

```python
class CampaignEngine:
    def recommend(self, batch_size: int): ...
    def ingest(self, df): ...
    def save(self, path): ...
    def load(self, path): ...
  ```  

---

# 📄 Phase 2: Engine Implementations (BayBE + Ax)

## Engine Implementations

---

### BayBEEngine

#### Responsibilities

- Construct BayBE campaign
- Generate recommendations
- Ingest completed experiments
- Ignore failed/abandoned trials in training

---

#### Requirements

- Must use actual BayBE API
- Must support:
  - continuous parameters
  - discrete parameters
  - categorical parameters
  - substance parameters

---

#### Behavior Rules

- Failed / abandoned trials:
  - Must NOT be included in model fitting
- Partial data:
  - Must be rejected OR explicitly handled

---

#### Forbidden

- Mocking BayBE behavior
- Skipping parameter validation
- Silent failure handling

---

### AxEngine

#### Responsibilities

- Create Ax experiment
- Manage trial lifecycle
- Handle trial statuses explicitly
- Generate recommendations via Ax API

---

#### Required Trial Status Mapping

| Internal Status | Ax Mapping      |
|----------------|----------------|
| completed      | COMPLETED      |
| failed         | FAILED         |
| abandoned      | ABANDONED      |
| partial        | RUNNING / custom |
| invalid        | FAILED         |

---

#### Requirements

- Must use real Ax API (no simulation)
- Must support:
  - trial creation
  - trial status assignment
  - data attachment
  - candidate generation

---

#### Behavior Rules

- Failed trials:
  - Must be recorded
  - Must not be treated as valid observations

- Abandoned trials:
  - Must not be re-suggested

---

#### Forbidden

- Treating all trials as completed
- Ignoring trial state
- Using Ax without lifecycle tracking


## Phase 3: Trial Status System

---

### Objective

Accurately represent real experimental outcomes.

---

### Required Status Values

- completed
- failed
- abandoned
- partial
- invalid

---

### Definitions

- completed: valid experimental result
- failed: experiment attempted but did not produce usable data
- abandoned: experiment not completed
- partial: incomplete data
- invalid: data corrupted or unusable

---

### Requirements

- Status MUST be provided for every ingestion event
- Status MUST be stored persistently
- Status MUST influence backend behavior

---

### Backend Behavior

#### BayBE
- completed → used in model
- failed → ignored
- abandoned → ignored

#### Ax
- completed → valid trial
- failed → failed trial
- abandoned → abandoned trial

---

### UI Requirements

- Status selection must be mandatory
- Status must be visible in dashboard
- Status filtering must be supported

---

### Forbidden

- Defaulting status silently
- Treating all data as completed
- Ignoring status in modeling


## Phase 4: Model Visibility

---

### Objective

Expose internal model predictions to the user.

---

### Required Outputs (per candidate)

- predicted mean
- uncertainty (variance or standard deviation)
- acquisition score
- ranking

---

### Requirements

- All values must come directly from backend
- No approximations
- No fabricated values

---

### UI Requirements

- Display as sortable table
- Allow filtering and ranking
- Must update after each ingestion

---

### Forbidden

- Simulated predictions
- Placeholder values
- Static or cached outputs presented as live

## Phase 5: Multi-Objective Optimization

---

### Objective

Support optimization across multiple targets.

---

### Requirements

- Multiple targets must be definable in schema
- Each target must specify:
  - name
  - direction (maximize/minimize)

---

### Backend Support

#### Ax
- Full multi-objective support required

#### BayBE
- If unsupported:
  - Feature must be disabled in UI
  - Must NOT silently fallback to single-objective

---

### UI Requirements

- Display Pareto front
- Allow selection of trade-off points

---

### Forbidden

- Pretending multi-objective support exists
- Collapsing objectives without user knowledge

## Phase 6: Search Space Editing

---

### Objective

Allow modification of search space during active campaigns.

---

### Allowed Operations

- Add categorical values
- Remove categorical values
- Adjust numeric bounds

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

## Phase 7: Deduplication and Recommendation Hygiene

---

### Objective

Prevent redundant or wasteful experiments.

---

### Requirements

- Block exact duplicates
- Detect near-duplicates (for discrete grids)
- Allow intentional replicates

---

### Behavior

- Warn user before duplicate recommendation
- Provide explanation for filtering

---

### Forbidden

- Silent removal of recommendations
- Recommending identical experiments without notice

## Testing Requirements

---

### Unit Tests

- Parameter validation
- Engine recommend()
- Engine ingest()
- Status handling

---

### Integration Tests

Full workflow:
- initialize
- recommend
- ingest
- repeat

---

### UI Consistency Tests

For each UI control:
- Verify backend receives parameter
- Verify behavior changes accordingly

---

### Forbidden

- Untested features
- Manual-only validation