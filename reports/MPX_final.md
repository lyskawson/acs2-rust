# The Multiplexer arc in Rust ACS2 — a scientific report

**Scope.** This report tells the whole Multiplexer (MPX) story as one narrative: how far
ACS2 learns the Boolean multiplexer, how compactly, where it stops, and what turned out to
be stopping it. It is a supervisor-facing summary; the terse implementation record lives in
`docs/ARCHITECTURE.md`.

**Result, up front.** 70 bits is solved on every seed tried and 135 bits is solved — the
latter both by changing the problem encoding and, more importantly, by removing an
exploration bias while leaving the canonical encoding intact. Published ACS/ACS2 results
stop at 20–37 bits. Acts I–IV are the history that got there and remain as written; Act V
is the resolution, and it overturns this report's own earlier conclusion that k=135 was a
mechanism boundary.

**Why the multiplexer.** The k-bit multiplexer is the standard LCS scaling benchmark: `a`
address bits select one of `2^a` data bits, so the perception is `N = a + 2^a + 1` bits
(the `+1` is a validation/answer bit that flips on a correct action). A standard compact rule
specifies `a + 1` bits (the `a` address bits + the selected data bit) and wildcards
the rest. Other minimal rules can omit address bits when candidate data bits agree.
Specificity near `a+1` measures compactness; correctness requires checking predictions,
not specificity alone. The reliable population size is another compactness measure. Knowledge = fraction of the `2^k × 2` (input, action) pairs the
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
ALP-gen keeps reliables COMPACT near the ideal a+1 = 7 and lifts knowledge by two orders of
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

## Act IV — k=70 solved at knowledge = 1.0, on every seed tried (M3)

Given a budget large enough, canonical ALP-generalization does **not** stop at k=37. All
five seeds attempted at k=70 (pyalcs variant, `u_max` derived = 8, GA on) reach
knowledge = 1.0:

| seed | trials to success | reliable rules | reliable spec (/71) | wall | machine |
|------|-------------------|----------------|---------------------|------|---------|
| 42 | 17,880,000 | 277 | 7.04 | 6,105 s | M1 laptop |
| 43 | 17,820,000 | 269 | 7.00 | 13,692 s | Bem2 |
| 44 | 44,580,000 | 274 | 7.00 | 28,988 s | Bem2 |
| 45 | 21,300,000 | 271 | 7.00 | 11,529 s | Bem2 |
| 46 | 66,420,000 | 268 | 7.00 | 46,266 s | Bem2 |

This exceeds every published ACS/ACS2 result (literature maximum ~37-bit; see
`MPX_literature_review.md`). Two properties of the table matter more than the headline:

**The solution is invariant; the cost is not.** Every seed lands on the same structure —
268–277 reliable rules at the IDEAL `a+1` = 7 specificity — but trials-to-success spans
17.82 M to 66.42 M, a **3.73x spread** (median 21.30 M). Seed 42, the seed the earlier
sections report, is at the fast end of that range and must not be read as typical. Any
budget derived from it under-provisions by up to ~4x.

**The plateau is not a stall, and it can be enormous.** Every trajectory is an S-curve:
climb, plateau, sprint. The plateau length is what varies:

| seed | plateau level | plateau span | share of run |
|------|---------------|--------------|--------------|
| 42 | 0.669 | 5.0 M → 7.5 M (2.5 M) | 14 % |
| 43 | 0.351 | 6.3 M → 7.7 M (1.4 M) | 8 % |
| 44 | 0.723 | 14.0 M → 31.3 M (17.3 M) | 39 % |
| 45 | 0.636 | 8.2 M → 9.5 M (1.3 M) | 6 % |
| 46 | 0.745 | 27.5 M → 57.5 M (29.9 M) | 45 % |

Seed 46 held knowledge ≈ 0.745 for **29.9 M consecutive trials — 45 % of its run — and
then still converged to 1.0.** Any experiment that had stopped it at 50 M trials would
have reported a confident, wrong negative. This is the single most important
methodological caution the MPX arc produces: **at k ≥ 70, a flat knowledge curve is not
evidence of non-convergence at any budget short of tens of millions of trials.**
Specificity and reliable-count are the diagnostic channels — they keep moving while
knowledge looks frozen (`reports/figures/mpx70_anatomy_s46_pyalcs.pdf`).

### k=135 — where it stood, and why it looked like a wall

The k=135 run (seed 42, pyalcs @ `u_max` = 9, GA on) was given **105,621,000 trials over
250,000 s** — 1.6x the trials that solved k=70's slowest seed — and ended TIME-LIMITED at
knowledge = 0.0000. The trajectory contains no learning signal whatsoever:

- **reliable = 0 at every one of the 1,760 evaluation points.** Not a slow climb; never a
  single rule crossed the reliability threshold.
- **Population is stationary**: decile means 4,023–4,133 across the entire run (min 3,147,
  max 19,469), with no trend in either direction.
- For contrast, every k=70 seed formed its first reliable rule inside the first ~10⁵
  trials.

This is qualitatively unlike the k=70 plateau, where knowledge was flat but specificity
and population were visibly still converging. Here nothing moved, so more wall-clock did
not look like the lever.

**That reading was right about the budget and wrong about the wall.** k=135 is solved —
Act V. What follows is the diagnosis that got there, in the order it was established.

### The diagnosis, measured: ALP specializes blindly at k=135

Population-wide instrumentation (`--log-diagnostics`, read-only over the population —
the k=70 run below reproduces its trials-to-success exactly, 17,880,000) locates the
failure precisely.

First, what it is **not**. At k=135 mean population specificity sits at 8.8–9.6 against
an ideal `a+1` = 8, while the *solvable* k=70 run runs at 8.7 rising to 10.3 against its
ideal of 7. The k=135 population is, if anything, closer to ideal specificity than the
one that succeeds — so the "specialization outruns generalization" reading carried over
from M2a does **not** explain k=135. Mark density is likewise not the discriminator: the
k=70 run ends at 0.914 and solves, against k=135's 0.935.

One diagnostic is *which* attributes get specialized. A standard compact MPX rule
specifies the `a` address bits plus the data bit those bits select. Measuring how many
address positions each classifier specifies, against `specificity x a / k` — what the same
classifier would hit choosing attributes uniformly — gives a signal with a built-in null
hypothesis:

| | MPX-70 (solved at 17.88 M) | MPX-135 (9.51 M trials) |
|---|---|---|
| address-bit enrichment over blind choice | 1.12x -> **8.0x** | **0.81–1.08x, no trend** |
| share of population holding a complete address | 0 -> **0.227** | **0.0000 at every point** |
| standard complete-address rules (`correct`, `a+1` spec) | 0 -> **256** | **0 at sampled points** |
| mean classifier experience | 2.9 -> **9,245.8** | **2.8, flat** |

At k=70 ALP's attribute choice is enriched eightfold over chance and a fifth of the
population ends up holding a complete address. At k=135 the enrichment never leaves the
blind-choice line, and **no classifier specifies all seven address bits at the sampled
evaluations through 9.51 M trials** (`reports/figures/mpx_specialization_signal.pdf`).

The historical `correct` column retains its original complete-address predicate. It
does not count every correct minimal rule: in MPX-6, address `0#` and data `00##`
always imply answer 0 with specificity 3 = `a+1`. A rule can leave an address bit
unspecified if both candidate data bits agree. Thus zero `addr_full` or `correct`
does not establish the absence of correct rules. The observed low experience and
quality below `theta_r` = 0.9 describe this finite probe, not permanent failure.

**What remains open.** The measurement shows no address enrichment in this k=135 probe;
it does not establish the cause. Two candidates are entangled and this data cannot separate them:
a mark too saturated to indicate which attribute matters, and rules too short-lived for
the mark to sharpen — each feeds the other. Separating them needs a further experiment.

This suggests testing whether replay helps candidate rules accumulate reliable evidence.
The present measurements do not distinguish insufficient reinforcement from inaccurate
generalization, replacement, or conflicting updates.

### The `u_max` limit is one of two causes

The diagnosis pointed at its own test. Population specificity at k=135 sits at 8.8–9.6
against a derived `u_max` of 9 — rules were resting *on the limit*, so the limit itself
was a candidate cause: a blindly specializing rule may need to hold surplus attributes
before the right ones are among them, and `u_max` = 9 generalizes it back first. Sweeping
`u_max` at k=135 (seed 42) confirms the limit matters. The following are last sampled
readings from finite runs, not established ceilings:

| `u_max` | trials | knowledge | reliable | specificity |
|---|---|---|---|---|
| 8 | 5.54 M | 0.0000 | 0 | — |
| 9 *(derived)* | 105.6 M | 0.0000 | 0 | — |
| 10 | 294.8 M | 0.2717 | 170 | 8.97 |
| **11** | **655.4 M** | **0.7499** | **396** | **8.00** |
| 12 | 366.8 M | 0.5000 | 259 | 8.05 |
| 13 | 158.2 M | 0.4889 | 258 | 8.14 |
| 14 | 25.1 M | 0.1628 | 6 861 | 15.20 |
| 16 | 35.5 M | 0.3242 | 8 092 | 16.89 |
| 24 | 1.59 M | 0.0000 | 0 | — |

The canonical `a+2` = 9 is too tight at 135 attributes; 11 is the best value tried. But
the readings near 0.75 and 0.50 motivate the class-level diagnosis below. These
time-limited runs do not isolate every interaction with `u_max`.

### The exact fractions are missing classes, not partial learning

Splitting knowledge by action × whether the transition changes the validation bit turns
those fractions into a structural statement. At `u_max` = 11, seed 42:

| class | coverage |
|---|---|
| action 0, changes | 1.0000 |
| action 1, changes | 1.0000 |
| action 1, no change | 1.0000 |
| **action 0, no change** | **0.0000, at every evaluation across 583.8 M trials** |

Three of four classes have reliable sampled coverage, one has none: approximately
0.75. Seeds 43, 44 and 45 have both no-change classes at zero reliable coverage in
these runs and finish near 0.50.

Candidate classifiers do exist in the starved class. Across the **4,865** `qdetail:`
points in `slurm_mpx135_s42_qdetail_u11.out`, `a0nc_any` is zero only **16** times,
partial **2,600** times, and prints **1.0000 at 2,249** evaluations. The best quality
ever recorded in that class is **0.823**, below `theta_r = 0.9`. This is a failure to
reach reliability despite frequent candidate coverage, not evidence that candidate
rules are never discovered. Coverage and quality are logged at finite precision;
they do not establish that the same candidates persist between evaluations.

The no-change classes are exactly the wrong answers. Under the canonical encoding a wrong
answer leaves the perception unchanged, so its rule must anticipate identity — every
classifier's default effect, which has to be *narrowed* — whereas correct-answer rules are
built directly by ALP's unexpected case.

### `u_max` is not doing hidden work at the sizes that already solve

The honesty ledger flags that a *derived* `u_max` could smuggle in knowledge of the
solution. Sweeping it where the system converges shows it does not — every value tried
solves, at knowledge 1.0 and near-ideal specificity:

| k | ideal | `u_max` values tried | result | trials range |
|---|---|---|---|---|
| 20 | 5 | 5, 6, 7, 8, 9, 10 | **all SUCCESS**, spec 5.00–5.42 | 80 k – 160 k |
| 37 | 6 | 6, 7, 8, 9, 10, 12 | **all SUCCESS**, spec 6.00–6.85 | 690 k – 1.02 M |

The derived value is the *fastest* at both sizes (k=20 at 6, k=37 at 7), but nothing hinges
on picking it — the reach claim survives any value in the range. Which makes the k=135
result sharper rather than weaker: `u_max` is inert where the task is tractable and
decisive exactly where it is not.

---

## Act V — k=135 solved, by two independent routes

The starvation hypothesis makes two different predictions, and both hold.

### Route 1 — give the wrong answer an observable effect

Under `--encoding outcome` the validation attribute starts at 2 and becomes 1 on a hit or
0 on a miss, so every action changes the perception and no rule has to anticipate identity.

Tested first at k=70, where the canonical encoding already succeeds, so the comparison is
clean: starvation disappears and the problem solves **3.8x faster** (4.68 M trials against
17.88 M) at the same final structure (276 vs 277 reliable rules, specificity 7.01 vs 7.04).
Under the canonical encoding the `a1_wrong` class sits at exactly 0.0000 from 120 k to
5.88 M trials; under `outcome` all four classes climb together from the start.

At k=135 it closes the problem outright, on four of five seeds:

| seed | trials | reliable | specificity (ideal 8) | wall |
|---|---|---|---|---|
| 42 | 43.2 M | 539 | 8.00 | 50.5 h |
| 43 | 46.8 M | 539 | 8.00 | 78.8 h |
| 45 | 55.6 M | 532 | 8.00 | 80.9 h |
| 46 | 30.2 M | 535 | 8.01 | 57.6 h |
| 44 | — | — | — | cancelled before success; an earlier reading had population 64 579 at specificity 10.9 |

Seed 44 remains unfinished; its eventual convergence is unknown. **Results under `outcome` are not
comparable to the multiplexer literature** — it is a different problem.

### Route 2 — stop the exploration bias from avoiding the starved class

ACS2's greedy branch selects among classifiers that anticipate change, which under the
canonical encoding means classifiers anticipating a *correct* answer. It therefore
under-visits exactly the transitions that starve. Setting `epsilon = 1` makes action choice
uniform and removes that bias, changing nothing else.

**This solves k=135 under the canonical encoding**: seed 43 reaches knowledge 1.0000 at
427,920,000 trials, 532 reliable rules at specificity 8.02, all four coverage classes at
1.0000, in 152.3 h. Those numbers *are* comparable to the literature.

The shape of the climb is the interesting part, and both seeds share it. One wrong-answer
class opens early; the other holds at **exactly** 0.0000 for hundreds of millions of
trials, then opens abruptly and the run finishes within ~23 M more:

| seed | first class opens | second class opens | outcome |
|---|---|---|---|
| 43 | `a0_nochange` at 20.2 M | `a1_nochange` at 404.5 M | knowledge 1.0 at 427.9 M |
| 42 | `a1_nochange` at 80.6 M | still 0.0000 at 301.4 M | stopped by the wall-clock cap |

Seed 42 is not a seed that fails to respond. It was stopped 100 M trials before the point
where seed 43's second class opened. A run sitting at 0.7442 with one class at exactly zero
is mid-climb, and reading it as converged is the error this report has now made three
times — see the ledger.

**Status: one seed, confirmed; a second under way.** The canonical claim rests on seed 43
alone until that finishes.

### `accuracy` and `knowledge` are different criteria, and the gap is the whole story

`knowledge` demands anticipating every transition, the null ones included. Choosing the
right answer needs only the change-anticipating half. The literature scores the latter —
ExSTraCS reports classification accuracy, ACS2ER reports reward.

At k=135, canonical encoding, `u_max` = 11, the accuracy log spans **325.8 M trials**.
Its first reading is **0.4989**; only **769 of 2,715 evaluations (28.3%)** print
**1.0000**. The last sub-1.0000 reading occurs at **299,160,000 trials**, with an
uninterrupted sequence of printed 1.0000 readings starting at 299,280,000. Near the
end, knowledge is approximately 0.7499. There was no 325.8-million-trial plateau
at accuracy 1.0000.

This is **rounded sampled accuracy**, not an exact zero-error result: **49,999 of
50,000** correct answers also prints **1.0000** at four decimal places. Report it
beside knowledge without claiming exact or exhaustive classification success.
At `u_max` = 12 the final sampled accuracy is about 0.982 while knowledge is 0.4821;
high accuracy is not automatic when reliable coverage stalls.

### k=264 — measured, not extrapolated

The next size is 264 (`k = a + 2^a` admits nothing between). A 12-hour probe at
`u_max` = 12 under `outcome`:

| | measured |
|---|---|
| throughput | 27 trials/s average, decaying 88 → 19 as the population grew |
| peak RSS | **0.86 GB** — memory is not the constraint |
| population | 60 366 after 1.165 M trials, still growing linearly |
| reliable rules | 0 |

Scaling trials by the ~10x seen from 70 to 135 puts k=264 near 400 M trials, i.e.
**700–4500 CPU-hours per seed**. The range is wide because it is unknown whether the
population condenses as it did at 135, where it peaked near 80 k and then collapsed to
~2.5 k while throughput recovered. The longest queue available is 21 days (504 h), so
k=264 needs save/restore of the population before it is attemptable at all.

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
  sweep was outside the original M2b phase; this report now includes the later k=135
  sweep explicitly. The maze path keeps `u_max = 100000`.
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
- **Knowledge is sampled above k=20** — 50,000 inputs at a fixed evaluation seed, exhaustive
  only for k ≤ 20.
- **This report has called a mid-run reading a ceiling three times**, and each time it was
  wrong. k=70 seed 46 sat at ~0.745 for 29.9 M trials and then reached 1.0. Seed 43 under
  `epsilon = 1` was reported as capping at 0.7481 and reached 1.0000. Seed 42's starved class
  was read as a property of the seed when it was the same plateau seed 43 escaped 100 M
  trials later. A flat knowledge value is evidence of nothing until it is compared against
  how long the comparable run stayed flat before it moved.
- **Peak-memory figures from cluster runs before `f93b71e` are void.** `ru_maxrss` is bytes
  on macOS and kilobytes on Linux; the code assumed bytes, so those runs report 0.00 GB.

---

## Bottom line for the thesis

1. **Specialize-only caps at ~k=11**; canonical **GA-on compactifies to k=37** but **loses a
   race at k ≥ 70** (reliables freeze fully-specialized, knowledge ≈ 0).
2. **Canonical ALP-generalization breaks that freeze**: it converges to the IDEAL `a+1`
   specificity (cleaner than GA at every k, strikingly so at k=37: 6.0 vs 7.91) and turns the
   k=70 boundary from race-lost (spec ~57, knowledge ≈ 0) into compact-but-time-bound (spec
   ~`a+1`, knowledge ~0.2).
3. Given a large enough budget it pushes the knowledge=1.0 reach to **k=70 on all five seeds
   tried** (17.8 M–66.4 M trials, median 21.3 M; 268–277 reliable rules at the ideal `a+1`
   specificity) — beyond every published ACS/ACS2 result. The earlier "does not pass k=37"
   reading was a budget artifact of 600-second probes, not a property of the mechanism.
4. **k=135 reaches sampled knowledge 1.0, with strong configuration and budget dependence.**
   Wrong-answer classes can lack reliable coverage even while candidates cover much or
   all of the sampled class; in the seed-42 `u_max=11` diagnostic run their best recorded
   quality stays below `theta_r`. Two interventions relieve reliable-coverage starvation:
   giving the wrong answer an observable effect
   (`--encoding outcome`, 4 of 5 seeds, 30.2–55.6 M trials) and removing the greedy
   exploration bias that avoids those transitions (`epsilon = 1`, knowledge 1.0000 at
   427.9 M trials **under the canonical encoding**, one seed so far). Seed 42's canonical
   epsilon-1 run was cut at 301.4 M; its eventual outcome is unknown. These interventions
   support an exploration/encoding dependence, without identifying a unique cause.
5. **Report `accuracy` beside `knowledge` or the result reads as weaker than it is.** At
   k=135 later evaluations print sampled answer accuracy 1.0000 while knowledge is near
   0.7499. Four-decimal rounding prevents an exact zero-error claim.
6. **The motivation for prioritised replay survives, sharpened.** The failure was never
   uniform slowness; candidate coverage and reliable coverage diverge in particular
   transition classes. Test whether prioritisation improves candidate quality, retention,
   and reliable coverage at matched learning applications. The data does not yet prove
   which sampling criterion will do so; uniform ACS2ER is the baseline it has to beat.
7. The **pyalcs-vs-Butz (parent- vs child-generalization) divergence** is a first-class
   implementation finding, but a confound-controlled isolation shows the freeze-break itself
   is variant-INDEPENDENT (knowledge ~0.25 either way); the separable variant differences are
   threshold tolerance (Pyalcs works at the tighter a+2; Butz collapses there and needs a+3)
   and modestly tighter conditions for Butz at equal threshold. The port must pick one variant
   and derive its `u_max` accordingly — that choice, not a compactness sweep, is the finding.

*Raw data: every run at a given size is tabulated in `reports/MPX<k>_runs.md`, generated
from `reports/mpx_verdicts.csv` and `reports/mpx_trajectory.csv`, which are in turn parsed
from the cluster logs committed in `reports/` (`tools/README.md` documents the pipeline and
`tools/sync_runs.sh` rebuilds it). Every figure in `reports/figures/` regenerates from those
CSVs, never from the logs. Implementation record: `docs/ARCHITECTURE.md`.*
