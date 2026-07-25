# Literature Review: Scaling the Boolean Multiplexer in LCS (Phase 1)

Date: 2026-07-22. Scope: how large multiplexers (MPX) have been solved by learning
classifier systems, which mechanisms are credited with the scaling, and which of those
transfer to our Rust ACS2, whose k>=70 boundary is a specialize-vs-generalize race
(over-specialized rules match ~1/2^k inputs, so their niches are almost never revisited
and no generalization mechanism can reach them).

## 0. Source access report

Reachable and used: Google Scholar (result pages), arxiv.org (abstracts and PDFs;
PDFs read via local text extraction), PubMed Central (full text of ExSTraCS 2.0),
Springer abstract pages (via a cookie-redirect hop; abstracts only, no full text),
Semantic Scholar API (rate-limited, partially usable), GitHub (cloned ounold/ALCS).

Blocked or unusable: dl.acm.org (HTTP 403), ieeexplore.ieee.org (page renders empty),
Springer/Kluwer full texts (paywall), ResearchGate PDFs (not fetched). Consequence:
claims from ACM/IEEE-only venues (e.g. exact numbers inside Iqbal et al. TEVC 2014 and
Wilson's original papers) are verified through secondary sources (Knittel & Blair 2014,
Urbanowicz & Moore 2015, survey texts) rather than the primary PDFs. Each such case is
flagged below.

## 1. Largest multiplexers solved, by system

| System | Largest MPX solved | Cost (as reported) | Mechanism credited | Primary source |
|---|---|---|---|---|
| XCS (original) | 20-bit | N=400–2000 range, ~10^5 instances | accuracy-based fitness, niche GA, subsumption | Wilson 1995; Wilson 1998 [flag: numbers via secondary sources] |
| XCS (tuned) | 70-bit | "a handful of studies"; optimized settings cut instances to ~21% of standard | tournament selection, learning-bound-based population sizing | Butz, Goldberg, Lanzi 2005; Butz 2006 book; confirmed by Knittel & Blair 2014: "Standard XCS has been shown to solve problems up to size 70-bit" |
| XCS + layered/stepped reward | 135-bit (layered variant) | not extracted | reward shaping guides address-bit discovery — "essentially giving part of the solution to the algorithm" (Urbanowicz & Moore 2015) | Butz 2005/2006 |
| XCSCFC (code fragments) | 135-bit | ~5×10^5 instances (70-bit), ~2×10^6 (135-bit), per Knittel & Blair 2014 | transfer of code-fragment building blocks learned on smaller MPX | Iqbal, Browne, Zhang, IEEE TEVC 18(4):465–480, 2014 |
| ExSTraCS 2.0 (supervised, UCS lineage) | 135-bit, first *direct* reliable solution (no transfer, no reward shaping) | 37-bit: N=5000, 2×10^5 iters; 70-bit: N=10000, 5×10^5; 135-bit: N=10000, 1.5×10^6 | rule specificity limit (RSL) + expert-knowledge covering + attribute tracking/feedback + TuRF | Urbanowicz & Moore, Evolutionary Intelligence 8:89–116, 2015 (PMC4583133, full text read) |
| ADN (non-LCS baseline) | 135-bit | 9×10^6 instances (70-bit), 5×10^7 (135-bit) | feature re-use network + transfer | Knittel & Blair, arXiv:1412.4967, 2014 (full text read) |
| ACS (Butz/Goldberg/Stolzmann) | 20-bit, reportedly 37-bit | not extracted | ALP specialization + genetic generalization symbiosis | Butz, Goldberg, Stolzmann, Natural Computing 1(4), 2002; Butz 2002 book [flag: the "keeps up with XCS on 20- and 37-MPX" phrasing surfaced in indexed text tied to the 2002 book; primary text not directly verifiable online — verify via library copy] |
| ACS2 (pyalcs, Kozlowski & Unold) | 20-bit | 6/11/20-bit only in published experiments | canonical ACS2 | Kozlowski & Unold 2018 (OpenAI gym integration) and follow-ups (experience replay, 2022–2023) |

Bottom line: no published ACS/ACS2 result beyond ~37-bit exists; our k=37 already matches
the best documented ACS2 reach. Every LCS that got to 70/135-bit did it by **controlling
specificity**, not by strengthening after-the-fact generalization.

## 2. Mechanisms that made XCS-family systems scale

### 2.1 Rule specificity limit (RSL) — ExSTraCS 2.0
A hard cap on the number of specified (non-#) attributes a rule may ever have, enforced
in covering, crossover, and mutation. Set data-drivenly: increase candidate order n until
the training set can no longer statistically support ε^n attribute-state combinations.
Urbanowicz & Moore credit RSL (plus EK covering) as the change that let ExSTraCS solve
6→135-bit MPX "reliably... for the first time" without building blocks. Key observation:
an MPX(k) solution rule needs exactly addr+1 specified bits (log2-ish: 6-MPX→3, 135-MPX→8),
so a cap slightly above that loses nothing and makes over-specialization *impossible* —
it removes the losing side of the race instead of trying to win it back.

### 2.2 Covering generality — XCS P# / ALCS u_max
XCS covering specifies each attribute only with probability (1−P#); high P# (generality)
plus accuracy pressure is the classical XCS recipe (Wilson 1995; theory in Butz, Kovacs,
Lanzi, Wilson 2004). The supervisor's ALCS implements the mirror parameter `u_max`
("max attributes in covering", default 1): covering starts from an almost fully-#
condition and specializes on demand. Section 4 details his code.

### 2.3 Implicit generalization pressure via niche GA + subsumption
Butz, Kovacs, Lanzi, Wilson (IEEE TEVC 8(1), 2004) formalize XCS pressures: *set
pressure* (more general rules appear in more action sets, so reproduce more often),
mutation pressure, deletion pressure, and subsumption. GA subsumption + action-set
subsumption (Wilson 1998) condense the population onto maximally-general accurate rules.
Crucial structural fact: XCS has **no deterministic specialization operator** — the only
specializing forces are covering and mutation, both stochastic and weak, so the
equilibrium sits near maximal generality. ACS2 inverted this: ALP specializes
deterministically on every unexpected/expected-with-mark case, and the GA was bolted on
later precisely to push back (Butz, Goldberg, Stolzmann, GECCO 2000 parts 1+2; Natural
Computing 2002). Our k>=70 freeze is this asymmetry expressed at scale.

### 2.4 Tournament selection and population sizing
Butz, Sastry, Goldberg (GECCO 2003, "Tournament selection: stable fitness pressure in
XCS") made fitness pressure robust to fitness scaling; Butz, Goldberg, Lanzi 2005
("Computational complexity of the XCS classifier system") derive population bounds under
which XCS PAC-learns k-DNF, and report the 70-bit MPX with optimized settings. ExSTraCS
practice matches the theory: N grows 5000→10000 from 37-bit to 70/135-bit. Population
must scale with the niche count, or niches lose support and rules freeze or churn.

### 2.5 Mechanisms NOT transferable (noted for completeness)
- Code fragments / building-block transfer (Iqbal et al. 2014; Alvarez et al. 2016):
  representation change + curriculum of smaller MPX; out of scope for a canonical-ACS2 thesis.
- Layered/stepped reward (Butz 2005): changes the problem definition.
- Expert-knowledge covering, attribute tracking/feedback, TuRF (ExSTraCS): supervised-
  learning constructs needing class labels and epidemiological EK sources.
- Specify operator (Lanzi 1997): counter-pressure for *over-general* rules — the opposite
  failure mode; relevant only as precedent that explicit pressure-balancing operators are
  an accepted design move in LCS.

## 3. ACS/ACS2 specifically
The over-specialization problem is *documented from the beginning*: Butz, Goldberg,
Stolzmann introduced genetic generalization exactly because "the anticipatory learning
process... over-specializes" (GECCO 2000; Natural Computing 2002), reporting a symbiosis
that yields complete, accurate, maximally-general models on toy environments and MPX up
to 20 (reportedly 37). Nothing in the ACS2 literature addresses MPX at 70+; the
generalization symbiosis was never stress-tested at scales where niche revisit frequency
collapses. pyalcs-line publications (Kozlowski & Unold 2018–2023) stay at 6/11/20-bit.
Our boundary is therefore an open problem in the ACS2 literature, and the XCS/ExSTraCS
evidence says the fix applied elsewhere was always: **prevent specificity from exceeding
what the problem needs**.

## 4. The supervisor's ALCS implementation (code inspection, github.com/ounold/ALCS)
Read-only findings; his README demonstrates mux_11 and the repo contains no evidence of
runs beyond k=37 — treated purely as a source of parameter ideas.

- `u_max` (conf.py:30, default 1; logic.py:133): covering builds an all-# condition and
  specializes `min(u_max, l_len)` randomly sampled positions from the current state.
  With u_max=1, covering emits near-maximally-general rules — the ACS2 analog of high P#
  in XCS, and the covering-side half of an RSL.
- `alp_mark_only_incorrect` (default true; acs2CPU3.py:95–117, acs2GPU4.py:543):
  restricted marking. Canonical/full mode marks *every* action-set classifier with the
  previous state on every step; restricted mode marks *only* classifiers whose
  anticipation was wrong. Since marks are what drive expected-case specialization
  (off-condition specializes each # position whose mark disagrees with the state),
  restricted marking removes mark accumulation on already-correct classifiers and thus
  most of the ALP's specialization pressure. His parameter_guide.md states this
  "strongly favors generalization and keeps the population compact", vs. full marking's
  "rapid specialization... significantly larger, more specific population".

Both are anti-specialization levers, converging with the ExSTraCS/XCS story.

## 5. Ranked, testable levers for our Rust ACS2

1. **Specificity control at covering (u_max) + hard rule specificity limit (RSL).**
   Strongest external evidence (ExSTraCS 2.0's direct 135-bit; XCS P# theory) and present
   in the supervisor's code. Implementation: flag-gated `u_max` in covering, plus a cap on
   specified attributes enforced in ALP specialization and GA/crossover offspring. For
   MPX(k) the required specificity is addr+1, so an RSL of addr+1 (or a small margin) is
   principled, and a data-independent sweep (RSL ∈ {addr+1, addr+3, ∞}) is a clean gated
   experiment. Directly removes the frozen-niche failure: no rule can specialize into a
   1/2^k niche.
2. **Restricted marking (`alp_mark_only_incorrect`).** Cuts ALP specialization pressure
   at the source; one boolean flag; matches the supervisor's default and the historical
   diagnosis (Butz et al. 2000/2002) that ALP over-specializes. Risk: slower model
   convergence on small k — must re-verify knowledge=1.0 at k≤37.
3. **Genetic-generalization pressure tuning + subsumption audit.** Grounded in Butz,
   Goldberg, Stolzmann 2000/2002 and Wilson 1998: verify GA subsumption and action-set
   subsumption fire on our MPX path, then sweep theta_ga down / mu up (generalizing
   mutation only) so genetic generalization outpaces residual specialization. Cheapest to
   try, but by itself cannot reach frozen 1/2^k niches — pair with lever 1 or 2.
4. **Population sizing with k (and optionally tournament-style GA selection).** Theory
   (Butz, Goldberg, Lanzi 2005) and practice (ExSTraCS N=10000 at 70/135-bit) agree N
   must grow with niche count; our k=70 runs should not reuse k=37's N. Tournament
   selection is a secondary, XCS-proven stabilizer.

Recommendation: pursue levers 1+2 as the primary pair (both flag-gated, both mirrored in
the supervisor's code, both aimed exactly at the specialize-vs-generalize race), with 3
as the supporting audit and 4 as an experimental control (N scaled per k) rather than a
standalone fix.

## References
- Wilson, S.W. (1995). Classifier Fitness Based on Accuracy. Evolutionary Computation 3(2):149–175.
- Wilson, S.W. (1998). Generalization in the XCS Classifier System. Proc. Genetic Programming 1998. [Semantic Scholar](https://www.semanticscholar.org/paper/8248c70075a62c9ca6aedb02415bad8b9c4839cf)
- Lanzi, P.L. (1997). A Study of the Generalization Capabilities of XCS. Proc. ICGA 1997.
- Butz, M.V., Goldberg, D.E., Stolzmann, W. (2000). Introducing a Genetic Generalization Pressure to the Anticipatory Classifier System, Parts 1–2. GECCO 2000. [Part 1](https://www.semanticscholar.org/paper/e282d4d6257bdd44d8ec061c2d055f04ed16c7cd)
- Butz, M.V., Goldberg, D.E., Stolzmann, W. (2002). The Anticipatory Classifier System and Genetic Generalization. Natural Computing 1(4):427–467. [Springer](https://link.springer.com/article/10.1023/A:1021330114221)
- Butz, M.V. (2002). Anticipatory Learning Classifier Systems. Kluwer. [Springer](https://link.springer.com/book/10.1007/978-1-4615-0891-5) [20/37-MPX claim — verify in library copy]
- Butz, M.V., Stolzmann, W. (2002). An Algorithmic Description of ACS2. IWLCS 2001, LNAI 2321.
- Butz, M.V., Sastry, K., Goldberg, D.E. (2003). Tournament Selection: Stable Fitness Pressure in XCS. GECCO 2003. [Springer](https://link.springer.com/chapter/10.1007/3-540-45110-2_83)
- Butz, M.V., Kovacs, T., Lanzi, P.L., Wilson, S.W. (2004). Toward a Theory of Generalization and Learning in XCS. IEEE TEVC 8(1):28–46.
- Butz, M.V., Goldberg, D.E., Lanzi, P.L. (2005). Computational Complexity of the XCS Classifier System. In Foundations of Learning Classifier Systems, Springer.
- Butz, M.V. (2006). Rule-Based Evolutionary Online Learning Systems. Springer. [70-bit XCS; layered 135-bit]
- Iqbal, M., Browne, W.N., Zhang, M. (2014). Reusing Building Blocks of Extracted Knowledge to Solve Complex, Large-Scale Boolean Problems. IEEE TEVC 18(4):465–480. [numbers verified via Knittel & Blair 2014]
- Knittel, A., Blair, A. (2014). Sparse, Guided Feature Connections in an Abstract Deep Network. [arXiv:1412.4967](https://arxiv.org/abs/1412.4967) [full text read; source of 70-bit/135-bit XCS+XCSCFC quotes]
- Urbanowicz, R.J., Moore, J.H. (2015). ExSTraCS 2.0: Description and Evaluation of a Scalable Learning Classifier System. Evolutionary Intelligence 8:89–116. [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4583133)
- Alvarez, I.M., Browne, W.N., Zhang, M. (2016). Human-Inspired Scaling in Learning Classifier Systems: Case Study on the n-bit Multiplexer Problem Set. GECCO 2016.
- Kozlowski, N., Unold, O. (2018). Integrating Anticipatory Classifier Systems with OpenAI Gym. GECCO 2018 Companion.
- Kozlowski, N., Unold, O. (2022–2023). ACS2 with (Episode-Based) Experience Replay. [GECCO 2022 companion](https://dl.acm.org/doi/pdf/10.1145/3520304.3533996)
- ounold/ALCS repository. [GitHub](https://github.com/ounold/ALCS) [code inspected 2026-07-22; no evidence of k>37 runs]
