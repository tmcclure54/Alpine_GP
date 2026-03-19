# Alpine-GP Agent Principles
## Non-Negotiable Standards for All Implementations

---

## 1. Core Principle: Truthful Systems Only

> Every feature exposed in the GUI MUST correspond exactly to real, functioning backend logic.

There is **zero tolerance** for:
- Placeholder features
- Ignored parameters
- Silent fallbacks
- UI elements that do not affect computation

### Explicit Rule
If it is visible in the GUI:
- It must be implemented
- It must be wired to backend logic
- It must be testable

If it is not implemented:
- It must NOT appear in the GUI  
OR  
- It must be clearly labeled: `"Not implemented"`

---

## 2. Scientific Integrity Requirement

> This system must behave like a scientific instrument, not a demo.

All outputs must be:
- Physically meaningful
- Statistically valid
- Derived from real computations

### Forbidden
- Fabricated values
- Placeholder calculations
- Approximate outputs presented as exact
- “Mock” model outputs

### Required
- Explicit computation paths
- Traceability from input → output
- Reproducibility from saved state

---

## 3. No Embellishment Rule

> The system must not exaggerate its capabilities.

Strictly prohibited:
- UI suggesting functionality beyond backend capability
- Implicit assumptions not communicated to user
- Overstating model confidence or accuracy

All limitations must be:
- Explicit
- Visible
- Honest

---

## 4. UI ↔ Backend Consistency

Every UI control must:
- Map directly to a backend parameter or function
- Change system behavior deterministically
- Be verifiable through testing

### Example of violation (DO NOT DO THIS)
- A slider for acquisition beta that is ignored internally

### Correct behavior
- If acquisition beta is exposed:
  - It must be passed into the optimizer
  - It must change recommendations
  - It must be testable

---

## 5. Data Integrity

### No Silent Transformations
- No implicit type coercion
- No hidden normalization
- No dropped rows without error

### Validation is Mandatory
All data must be validated before ingestion:
- Parameter names
- Parameter types
- Bounds
- Target values
- Trial status

Errors must:
- Fail loudly
- Provide actionable messages

---

## 6. Reproducibility

Every campaign must be reconstructible from:
- Configuration
- All experimental data
- All model states
- All snapshots

No hidden state is allowed.

---

## 7. Testing Requirement

Every feature must include:

### Unit Tests
- Individual function correctness

### Integration Tests
- Full workflow:
  - initialize → recommend → ingest → repeat

### UI Consistency Tests
- Verify that UI controls affect backend behavior

---

## 8. Anti-Patterns (Strictly Forbidden)

- Fake parameters
- Ignored kwargs
- Silent fallbacks
- Hidden backend logic
- UI-driven logic that bypasses backend
- Mixing UI, persistence, and modeling logic
- Hardcoded behavior that contradicts user input

---

## 9. Definition of Done

A feature is complete ONLY if:

- GUI control exists
- Backend implementation exists
- Data is persisted correctly
- Tests exist
- Behavior is verifiable

If any of the above is missing:
→ The feature is NOT complete

---

## 10. Final Principle

> If a chemist cannot trust the system, the system has failed.

- Every knob must work
- Every output must be real
- Every decision must be explainable

No exceptions.