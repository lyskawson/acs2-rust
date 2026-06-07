# Golden test vectors (pyalcs oracle)

Language-neutral JSON fixtures capturing the behaviour of canonical ACS2 as
implemented by **pyalcs**. They are the shared oracle for both the Python
baseline and the Rust port: every fixture's `expected` block was produced by
calling the unmodified pyalcs functions, and is re-checked against pyalcs by
`verify_fixtures.py`. P3–P5 consume these as Rust unit tests.

**62 fixtures across 10 files.**

| File | Count | Covers |
|------|------:|--------|
| `condition_matching.json` | 9 | `Condition.does_match`, `Condition.subsumes` (wildcard on either side; full/partial/no match) |
| `effect_anticipation.json` | 8 | `Effect.anticipates_correctly` (pass-through vs specified change), `Effect.is_specializable` |
| `mark_differences.json` | 6 | `PMark.get_differences` (empty; nr2 all-positions; nr1 single/multi-candidate) |
| `alp_expected_case.json` | 4 | `alp.expected_case` (None branch + specialized child) |
| `alp_unexpected_case.json` | 5 | `alp.unexpected_case` (effect specialization, marking, quality decrease, None) |
| `alp_covering.json` | 2 | `alp.cover` (fresh classifier, experience=0, reward=0) |
| `updates_qrir_tav.json` | 12 | `increase_quality`/`decrease_quality`, MAM `tav` across the `exp<1/beta` boundary, `rl.update_classifier` |
| `subsumption.json` | 7 | `does_subsume`, `is_subsumer` (experience/quality/marked) |
| `numerosity_collapse.json` | 6 | `Classifier.__eq__` identity = (C,A,E), `alp.add_classifier` dedup |
| `learning_step.json` | 3 | `apply_alp` + `apply_reinforcement_learning` on a constructed action set: population before/after |

## Fixture schema

Each file is `{"category": <name>, "fixtures": [ ... ]}`. Each fixture:

```json
{
  "id": "unique string",
  "category": "matches the file",
  "operation": "the pyalcs operation under test",
  "source": "test: <path>::<method>  |  generated: deterministic | ...",
  "description": "one line",
  "wildcard": "#",
  "input": { ... operation-specific ... },
  "expected": { "mode": "exact" | "property", ... }
}
```

Conventions:

- **Symbols** are strings. Wildcard is `"#"`. Conditions/effects/perceptions are
  string arrays; the array length is the `classifier_length` for that fixture.
- **Marks** are arrays (one per attribute) of sorted string arrays; `[]` is an
  unspecified position. Compare as sets.
- **Classifiers** are flat objects: `condition, action, effect, mark, q, r, ir,
  num, exp, talp, tga, tav`.
- **Floats** are full precision. Consumers compare with tolerance **`1e-9`**.

### `expected.mode`

- **`exact`** — deterministic (no RNG on the path). Compare the stored value
  directly (floats within `1e-9`, marks/symbols exact).
- **`property`** — a genuine random branch is exercised (`get_differences`
  picking one of several candidate positions; `expected_case` building the child
  condition from that pick). The fixture stores the **full set of admissible
  outcomes**, sampled by running pyalcs many times under varying RNG:
  `allowed_specialized_indices`, `allowed_specificity`, and the RNG-invariant
  fields (effect, action, quality, marked-ness). A correct implementation's
  outcome must fall within these sets regardless of its own RNG. Only the two
  lifted `get_differences`/`expected_case` multi-candidate cases use this mode;
  everything else is deterministic by construction (single-candidate marks).

## What "verifiable against pyalcs" means here

Three independent layers:

1. **Generation** computes every `expected` by calling unmodified pyalcs — no
   hand-written scalars.
2. **Literal cross-check** — where a pyalcs unit test hard-codes a value (the
   `expected_case` quality `0.54`, `rl.update_classifier` `r=97.975`/`ir=50`,
   the exact `unexpected_case`/`cover` conditions, every `does_subsume`
   result, …), the generator asserts the live pyalcs value against that test
   literal. Two independent sources (the test author's hand computation and the
   live call) must agree.
3. **Verification** (`verify_fixtures.py`) re-reads the JSON, reconstructs the
   `input` into pyalcs objects, re-runs the operation, and asserts agreement
   (exact within `1e-9`). For the 5 property-mode fixtures it additionally runs a
   non-sampling **analytic** cross-check independent of the sampler: the allowed
   specialized indices must equal pyalcs's `possible_idx`
   (`get_differences_candidates`), and the allowed specificity must equal
   `orig_spec + (condition[pos]=='#')` over those candidates. It also validates,
   independently of the generator's serializer, that every classifier object
   carries exactly the 12 expected keys with no nulls outside `talp`.

   Scope of this loop: because generate and verify share `cl_to_dict`/`dict_to_cl`,
   the exact-mode pass proves pyalcs-agreement + round-trip + determinism, and the
   key-set pass proves serialization completeness — but a *foreign* reader's
   parse is first exercised by the Rust consumer in P3. The property-mode analytic
   check is the one place the oracle is validated without re-running pyalcs's RNG.

No instrumented copy of pyalcs was needed: every operation is reachable through
the public API, so pyalcs is imported unmodified rather than edited (the
"instrument a copy, never the original" rule is satisfied by not touching it).

## Regenerate / verify

Run with the baseline uv environment (it provides the pinned pyalcs):

```bash
cd baseline
uv run python ../fixtures/generate_fixtures.py   # writes *.json
uv run python ../fixtures/verify_fixtures.py     # asserts all against pyalcs
```

`fixtures_common.py` holds the shared (de)serialization and the property-mode
samplers used by both scripts.

## Notes

- **GA fixtures are deferred to P5.** GA is off in the benchmark protocol and its
  paths add RNG (roulette selection, mutation, crossover, victim deletion);
  generating those as deterministic vectors belongs with the GA implementation.
- `learning_step` is the deterministic-path oracle for the P8 differential test:
  it exercises the real `apply_alp` + `apply_reinforcement_learning` orchestration
  on constructed inputs chosen so no random branch fires (covering only;
  unexpected-case child that covering then merges; correct-anticipation
  quality bump).
