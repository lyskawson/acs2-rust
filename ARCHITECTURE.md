# Rust workspace architecture decisions

Decisions that are not recoverable from reading the (P2-scaffold) signatures and
that P3–P8 depend on. PROJECT_CONTEXT.md remains the curated source of truth;
this file records implementation-level choices made while executing the build
plan.

## Crate layout and dependency direction

```
acs2-bench (bin)  ──▶ acs2-core, acs2-envs        (composition root)
acs2-envs  (lib)  ──▶ acs2-core
acs2-core  (lib)  ──▶ (no workspace deps; only rand/rand_chacha)
```

`acs2-core` depends on neither `acs2-envs` nor `acs2-bench` (P2 gate). Verified
with `cargo tree -p acs2-core`.

## The Environment port lives in `acs2-core`, not `acs2-envs`

PROJECT_CONTEXT §7's module map lists `environment` under `acs2-envs`, but the
**trait** (`Environment<N>`, `StepOutcome<N>`) is defined in
`acs2-core::environment`. This is forced, not a preference:

- §7/P6 place the single shared trial loop (`Agent`) in `acs2-core`.
- The loop calls `env.reset()` / `env.step()`, so it must name the `Environment`
  trait.
- The P2 gate forbids `acs2-core` depending on `acs2-envs`.

Therefore the port must live in core (Dependency Inversion: the domain defines
the interface; the outer layer implements it). `acs2-envs::maze::Maze`
implements `acs2_core::environment::Environment<8>`. A generic `reset`/`step`
over `Perception<N>` carries no maze specifics, so core keeps "zero environment
knowledge". This mirrors how RNG is already injected via a core trait (§8).

## `ClassifierRef` is a positional index — deletion contract

Match sets and action sets are `Vec<ClassifierRef>` where
`ClassifierRef = usize` indexes into `Population`'s internal `Vec<Classifier<N>>`
(`population.rs`). This deliberately avoids pyalcs's shared-mutable aliasing of
the same classifier objects across population/match-set/action-set.

Consequence that must be honored in P4/P5: a positional index is invalidated by
any mid-`Vec` removal (every later index shifts). **Append never shifts; only
deletion does.** ALP deletion (incorrectly-anticipating ∧ inadequate, SPEC D.6)
*does* fire within the 500 explore episodes, so:

- `apply_alp` is the **sole owner** of population mutation and set consistency.
- Deletion is **deferred to after** the per-item ALP loop (collect victims,
  delete once), never interleaved with iteration over the action set.
- After `apply_alp` returns, the agent **re-derives the match set** before
  selecting an action (match is a pure function of state + population, so this is
  equivalent to pyalcs's "extend match_set with new matchers" and also absorbs
  any deletions). This is the chosen resolution ("option b": plain `Vec` +
  re-scan) over a tombstoning/slotmap store.

A naive `Vec::remove` mid-loop reintroduces an index-aliasing bug that surfaces
only as a P8 differential mismatch, not a compile error — hence this note.

## Other choices

- **No `cfg` field in `Classifier`** (unlike pyalcs). `beta` and the `theta_*`
  thresholds are threaded as method parameters. This makes `copy_from` a pure
  field copy and keeps the classifier free of a shared-config God object.
- **Alloc-free seams.** `ActionSelector::select` and `BootstrapEstimator::estimate`
  take `(&Population<N>, &[ClassifierRef])` rather than `&[&Classifier<N>]`, to
  avoid building a `Vec<&Classifier>` on every action selection / bootstrap
  estimate (§8: minimize per-step heap churn).
- **Const-generic genome length `N`.** Core carries no environment dimension;
  the maze fixes `N = 8` at the edge (`MAZE_PERCEPTION_LEN`).
- **Subsumption is a relational strategy module** (`subsumption.rs`, added in P3),
  not methods on `Classifier`: free `is_subsumer(&Classifier, theta_exp, theta_r)`
  and `does_subsume(cl, other, theta_exp, theta_r)`, mirroring pyalcs
  `lcs/strategies/subsumption.py` (SPEC F.4). ALP (P4/P5) and GA (P5) reuse it.
  `Condition::subsumes` itself is the **symmetric** per-position permissive rule
  (`a==# || b==# || a==b`, identical to `does_match`) exactly as pyalcs
  `acs/Condition.py`; classifier-level subsumption is made asymmetric by the
  separate `is_more_general` (strict `specificity <`) guard inside `does_subsume`,
  not by `Condition::subsumes`.
- **`Symbol::Token(u8)` assumes a single-byte alphabet** (holds for maze codes
  `'0'/'1'/'9'`; fixtures also use `'2'`). Revisit if a non-maze environment with a
  multi-byte/wider alphabet is ever added.
- **`Symbol = enum { Wildcard, Token(u8) }`** (Copy/Ord). `Mark<N>` is
  `[BTreeSet<Symbol>; N]`. `Classifier` identity (`PartialEq`) is
  `(condition, action, effect)` only — implemented manually, no `Hash`.
- **RNG.** `acs2-core::rng::RandomSource` is the injected trait (object-safe:
  `gen_bool`, `gen_range`; `shuffle` is a free function). `ChaChaRandomSource`
  wraps `ChaCha8Rng` seeded via `seed_from_u64`. `rand`/`rand_chacha` use
  `default-features = false` so `getrandom`/OS entropy is **not** linked into the
  domain — determinism comes from the seed (§8), not entropy.
