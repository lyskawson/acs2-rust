# The Multiplexer arc in Rust ACS2 — a scientific report

**Scope.** This report tells the whole Multiplexer (MPX) story for the Rust ACS2 port as
one narrative: how far canonical ACS2 learns the Boolean multiplexer, how compactly, and
where — and why — it stops. It is a supervisor-facing summary; the terse implementation
record lives in `docs/ARCHITECTURE.md` (M1 / M2a / M2b sections).

**Why the multiplexer.** The k-bit multiplexer is the standard LCS scaling benchmark: `a`
address bits select one of `2^a` data bits, so the perception is `N = a + 2^a + 1` bits
(the `+1` is a validation/answer bit that flips on a correct action). The *ideal* rule
specifies exactly `a + 1` bits (the `a` address bits + the one selected data bit) and
wildcards the rest — so **reliable-condition specificity converging to `a+1` is the direct,
measurable signature of correct generalization**, and the size of the reliable population
measures compactness. Knowledge = fraction of the `2^k × 2` (input, action) pairs the
reliable population anticipates correctly; at k ≤ 20 it is computed EXHAUSTIVELY (every
pair), so "knowledge = 1.0" means literally every pair, not a sample.

**Protocol (identical across all phases).** Single-step episodes; reward **1000** on a
correct action, **0** otherwise; explore at ε = 0.8; the measured metric is taken on the
frozen population. Reach runs use a 4-way verdict (SUCCESS / TRIALS- / MEMORY- /
TIME-LIMITED) with RSS cap 5.6 GB and a trials cap = **M1-empirical specialize-only
extrapolation × 10** (`20000·2^(k−6)·10`), a deliberately loose non-convergence proxy —
NOT a literature budget — so that MEMORY or TIME binds before TRIALS. Seeds are base 42 +
repeat.

---

## Act I — Specialize-only caps at ~k=11 (M1)

Under the shipped pyalcs maze config (GA off, `u_max = 100000` → ALP only specializes),
structural learning is specialize-only. It solves the small multiplexers but cannot scale:

| k  | knowledge      | reliable pop | reliable spec (ideal a+1) |
|----|----------------|--------------|---------------------------|
| 6  | 1.0            | ~260         | 5.63 / 7   (ideal 3)      |
| 11 | 1.0            | ~4 380       | 9.01 / 12  (ideal 4)      |
| 20 | cannot reach   | explodes     | 20.5 / 21  (ideal 5)      |

The specificity tells the story: with no generalization pressure, reliable rules specialize
to nearly full length (9/12, 20.5/21 — one classifier per input), so the population scales
with the `2^k` input space. k=20 would need ~`2^21` reliable classifiers — infeasible.
**Specialize-only caps at ~k=11.** Compactness is a generalization problem, which motivates
Act II.

---

## Act II — Canonical GA-on compactifies to k=37, then loses a race (M2a)

Turning genetic generalization ON (`do_ga = true`, all pyalcs-default GA parameters, still
`u_max = 100000`) transforms the picture. GA drives reliable specificity toward the ideal
`a+1` and collapses the reliable population by orders of magnitude:

| k  | knowledge | reliable pop (GA) | reliable spec (ideal a+1) | vs M1 specialize-only |
|----|-----------|-------------------|---------------------------|-----------------------|
| 6  | 1.0       | 28.1              | 3.04 (ideal 3)            | 260 → 28              |
| 11 | 1.0       | 56.7              | 4.15 (ideal 4)            | 4380 → 57             |
| 20 | 1.0       | 88.6              | 5.24 (ideal 5)            | infeasible → 89       |
| 37 | 1.0       | ~166              | 7.91 / 38 (ideal 6)       | reach boundary        |

GA-on reaches **k=37** at knowledge = 1.0 with a compact population — proof by counting
that the ~166 reliable rules each generalize over ~`2^29` inputs. But at **k ≥ 70 the race
is LOST**: the few reliable rules that stabilize are heavily OVER-specialized (spec
46–68/71 at k=70, 135.9/136 at k=135), cover ≈ 0 % of the space, and knowledge collapses to
≈ 0. The mechanism is action-set revisitation: ALP specializes a rule toward full length
*before* GA generalizes it; once near-fully-specialized it matches ≈ `1/2^k` inputs, so its
action set is essentially never revisited, GA's trigger (`time − mean(tga) > theta_ga`)
almost never fires on it, and it can never be generalized back. **This is a race-lost
freeze**, not a memory or trials limit (peak RSS 0.2–0.4 GB, far below the cap). It is the
motivation for Act III — and, ultimately, for the thesis's prioritization / experience-
replay mechanism.

---

## Act III — Canonical ALP-generalization breaks the freeze (M2b)

M2b turns on the canonical ALP over-specialization generalization branch — DEAD in M1/M2a
because `u_max = 100000` — that generalizes INSIDE `expected_case`, on every action-set
visit, directionally (mark-driven), with no `theta_ga` gate. Two faithful variants were
implemented and measured (see "The pyalcs-vs-Butz divergence" below). GA stays ON
throughout (required — see Provenance).

### Cleaner generalization: specificity converges to the ideal `a+1`

Both ALP-gen variants hold knowledge = 1.0 (10/10, exhaustive) on the gate sizes AND drive
reliable specificity CLOSER to the ideal than GA's random mutation:

**Specificity trajectory (reliable-condition mean specificity; ideal = a+1):**

| k  | ideal a+1 | M1 specialize-only | M2a GA-on | M2b Pyalcs | M2b Butz |
|----|-----------|--------------------|-----------|------------|----------|
| 6  | 3         | 5.63               | 3.04      | **3.00**   | **3.02** |
| 11 | 4         | 9.01               | 4.15      | **4.00**   | **4.00** |
| 20 | 5         | 20.5               | 5.24      | **5.02**   | **5.00** |
| 37 | 6         | —                  | 7.91      | **6.03**   | **6.00** |

At k=37 the gap is stark: ALP-gen sits at the ideal 6.0, GA-alone at 7.91 (~2 redundant
specializations). **Canonical ALP-gen approaches the ideal more tightly than GA**, exactly
as theory predicts for a directional (vs random) generalization operator.

### The k=70 boundary changes CHARACTER: race-lost → compact-but-time-bound

This is the M2b headline. At k=70, GA-alone froze fully-specialized with knowledge ≈ 0.
ALP-gen keeps reliables COMPACT near the ideal a+1 = 8 and lifts knowledge by two orders of
magnitude — though it remains TIME-LIMITED short of 1.0 in the 600 s cap:

**Reach delta vs the GA-on k=37 baseline (verdicts are 3-seed, ALL AGREE unless noted):**

| k   | M2a GA-alone                    | M2b Pyalcs                        | M2b Butz                          |
|-----|---------------------------------|-----------------------------------|-----------------------------------|
| 37  | SUCCESS · spec 7.91 · know 1.0  | **SUCCESS · spec 6.03 · know 1.0**| **SUCCESS · spec 6.00 · know 1.0**|
| 70  | TIME-LIM · spec ~57 · know ≈0   | TIME-LIM · spec ~9.2 · **know ~0.17** | TIME-LIM · spec ~8.6 · **know ~0.24** |
| 135 | TIME-LIM · 62 rel frozen spec 135.9 | TIME-LIM · 0 reliable formed  | TIME-LIM · 0 reliable formed      |

The **largest k at knowledge = 1.0 is k=37 for all three** — ALP-gen does NOT push the
reach-to-1.0 boundary further in k. What it does is change the *nature* of the k=70
boundary: from GA's race-lost freeze (reliables fully specialized at spec ~57, matching
≈ `1/2^70`, knowledge ≈ 0) to ALP-gen's compact-but-time-bound state (reliables at spec
~`a+1`, knowledge climbing to ~0.2 in the same wall-clock). The freeze that made GA's
boundary a hard wall is REMOVED, because ALP-gen generalizes a specializing rule back
before it can freeze. At k=135 neither variant forms any reliable rule in budget (the space
2^135 is too large to accumulate reliability in 188–324k trials); this is TIME-bound, not
memory-bound (RSS ≤ 0.38 GB).

**Honest limit.** ALP-gen runs on the PREVIOUS action set, so it is NOT free of action-set
revisitation — which is why k=70/135 remain time-bound rather than clean successes. Its
advantage over GA is that it fires on *every* visit and generalizes *directionally*, which
removes the freeze but does not remove the revisitation cost. That residual cost is the
empirical case for the thesis's prioritization / experience-replay mechanism.

---

## The pyalcs-vs-Butz divergence — a first-class finding

The ALP generalization branch has a substantive divergence between the reference
implementation and the canonical algorithm, in the same class as the four documented pyalcs
bugs (though here BOTH sides are legitimate — one is the faithful port, one is the paper):

- **Pyalcs (`alp.py:78-94`, the port anchor)** generalizes the **PARENT** classifier in
  place and leaves the offspring fully specialized; it counts only *unchanging* specified
  attributes.
- **Butz (`butz_algorithm.pdf`, EXPECTED CASE)** generalizes the **CHILD** (offspring),
  counting *full* condition specificity.

Because the reliable classifiers the knowledge metric measures are the *children*, the two
variants could behave differently on exactly the metric that matters. A confound-controlled
k=70 isolation (seed 42, single repeat per cell) separates the variant from its `u_max`:

| config (k=70, seed 42) | knowledge | reliable | reliable spec (/71) |
|------------------------|-----------|----------|---------------------|
| Pyalcs @ a+2 (its correct threshold) | 0.241 | 110 | 8.63 |
| Pyalcs @ a+3                          | 0.246 | 257 | 9.80 |
| Butz @ a+2                            | **0.024** | **8** | 7.38 (collapsed) |
| Butz @ a+3 (its correct threshold)    | 0.256 | 203 | 8.58 |

Read carefully, this is a more honest — and more interesting — picture than "Butz wins":

1. **The freeze-break is VARIANT-INDEPENDENT.** Every valid config lifts knowledge to
   ~0.24–0.26 (vs GA-alone's ≈0). The earlier "Butz 0.24 vs Pyalcs 0.17" gap was a
   3-seed-mean artifact (Pyalcs' other seeds ran lower); at matched seed and threshold the
   two are equal. The headline M2b result — ALP-gen breaks the race-lost freeze — does NOT
   depend on the parent-vs-child choice.
2. **Butz CANNOT use the tighter a+2 threshold — it collapses** (knowledge 0.024, 8 reliable
   rules): its full-specificity count fires branch A on its own compact change rule and
   strips a needed bit. This directly CONFIRMS the derivation (that is why Butz derives a+3).
   Pyalcs, counting only unchanging attributes, works at the tighter a+2. So **Pyalcs
   tolerates a tighter over-specialization bound.**
3. **The variant DOES affect specificity, but modestly and threshold-entangled.** At EQUAL
   `u_max = a+3`, Butz yields tighter reliable conditions (8.58 vs 9.80) but FEWER reliable
   rules (203 vs 257) — so the "Butz more reliable rules" reading flips once `u_max` is held
   equal; reliable count tracks the threshold, not the variant. At each variant's OWN correct
   threshold the specificity is close (Pyalcs 8.63 vs Butz 8.58 at seed 42; across 3 seeds
   Pyalcs is noisier, 8.63–9.79, while Butz is stable ~8.6).

Net: child-generalization (Butz) produces marginally tighter and more consistent reliable
conditions at equal threshold, but the two variants are comparable on the metric that
matters (knowledge), and the clean separable difference is **threshold tolerance**, not a
Butz sweep. The parent-vs-child divergence is a first-class *implementation* finding — the
port must choose one and derive its `u_max` accordingly — but it is NOT the source of the
M2b freeze-break.

---

## Provenance and honesty ledger

Every non-obvious choice, stated plainly so the numbers can be trusted:

- **`u_max` is DERIVED, not literature.** No citable pyalcs-semantic `u_max` exists: Butz's
  algorithmic description defines `umax` but gives no value; the Unold papers never mention
  it; the `u_max=1` in the ALCS repo is a *different* "max attributes in covering" semantic
  that never reaches `expected_case`. The MPX values are derived from the solution structure
  (the compact rule specifies `a+1` bits) to the tightest value that preserves the compact
  rule and corrects one-redundant: **Pyalcs `u_max = a+2`, Butz `u_max = a+3`** (Butz's full
  count includes the changing validation bit in the condition, which pyalcs's unchanging
  count excludes — hence one higher). This bakes in knowledge of the answer; a `u_max`
  **sweep is a separate study**, not part of this work. The maze path keeps `u_max = 100000`.
- **The A/B `u_max` confound — checked and resolved.** Because Pyalcs ran at `a+2` and Butz
  at `a+3`, the raw A-vs-B gap conflated the generalization target (parent vs child) with the
  `u_max` value. A confound-controlled k=70 isolation (both variants at both thresholds, seed
  42) resolves it (see "The pyalcs-vs-Butz divergence"): knowledge is variant-independent
  (~0.25 everywhere valid), Butz cannot use a+2 (collapses, confirming its derivation), and
  the reliable-count difference flips at equal `u_max`. The originally-reported "Butz clearly
  better" was largely the confound; the honest residual is that Butz gives modestly tighter
  conditions at equal threshold, while Pyalcs tolerates a tighter threshold.
- **GA-on is REQUIRED throughout M2b (both variants).** GA-off + ALP-gen is unstable: with no
  `theta_as` action-set deletion pressure, the population explodes. Measured at k=11 (5000
  trials): Pyalcs is catastrophic (does not complete even a few hundred trials); Butz
  completes but blows the population to ~95k with ~4k reliable rules (vs GA-on's bounded
  ~290 / ~57) — child-generalization does NOT confer GA-off stability. So M2b = GA-on +
  ALP-gen throughout, mirroring M2a; GA is constant and equal across the A/B comparison.
- **Trials cap = M1-empirical specialize-only extrapolation × 10** (`200000·2^(k−6)`), a
  deliberately loose non-convergence proxy so MEMORY/TIME binds first — NOT a GA-on or
  ALP-gen budget prediction and NOT literature.
- **Reward is 1000 / 0** (correct / incorrect), single-step episodes, ε = 0.8 explore.
- **Maze path untouched.** The `u_max` flag keeps mazes on `u_max = 100000` (ALP-gen branch
  dead, zero added RNG draws), so the maze correctness gate (P8: 761/761 differential cases,
  zero divergence) and the P9 benchmark (byte-identical learning metrics) are UNCHANGED.

---

## Bottom line for the thesis

1. **Specialize-only caps at ~k=11**; canonical **GA-on compactifies to k=37** but **loses a
   race at k ≥ 70** (reliables freeze fully-specialized, knowledge ≈ 0).
2. **Canonical ALP-generalization breaks that freeze**: it converges to the IDEAL `a+1`
   specificity (cleaner than GA at every k, strikingly so at k=37: 6.0 vs 7.91) and turns the
   k=70 boundary from race-lost (spec ~57, knowledge ≈ 0) into compact-but-time-bound (spec
   ~`a+1`, knowledge ~0.2).
3. It does NOT push the knowledge=1.0 reach past k=37, because ALP still runs on the previous
   action set and remains action-set-revisitation-bound — **which is precisely the empirical
   motivation for the prioritization / experience-replay mechanism** the thesis proposes.
4. The **pyalcs-vs-Butz (parent- vs child-generalization) divergence** is a first-class
   implementation finding, but a confound-controlled isolation shows the freeze-break itself
   is variant-INDEPENDENT (knowledge ~0.25 either way); the separable variant differences are
   threshold tolerance (Pyalcs works at the tighter a+2; Butz collapses there and needs a+3)
   and modestly tighter conditions for Butz at equal threshold. The port must pick one variant
   and derive its `u_max` accordingly — that choice, not a compactness sweep, is the finding.

*Raw data: `reports/mpx_m2b_pyalcs.csv`, `reports/mpx_m2b_butz.csv`,
`reports/mpx_m2b_reach{37,70,135}.log`. Implementation record: `docs/ARCHITECTURE.md` §M2b.*
