# Audit Report: Four Claimed Defects in `hendrykik/acs2vcp-python`

Scope: conclusions drawn solely from the cloned repository and the attached Butz
*"An Algorithmic Description of ACS2"* paper. No code was changed.

---

## Claim 1 — GA deletion tie-break uses bare `cl.is_marked` instead of `cl.is_marked()`

**(a) Does the code do what's described?** Yes.
`pyalcs/lcs/strategies/genetic_algorithms.py:221-226`:
```python
if abs(cl.q - cl_del.q) <= 0.1:
    if cl.is_marked() and not cl_del.is_marked():     # line 222 — called
        return True
    elif cl.is_marked or not cl_del.is_marked():      # line 224 — NOT called
        if cl.tav > cl_del.tav:
            return True
```
Line 224 references the bound method object `cl.is_marked` without `()`. `is_marked` is a
**plain method**, not a `@property` (`acs2/Classifier.py:280`, `acs/Classifier.py:331` —
both `def is_marked(self): return self.mark.is_marked()`, no decorator). A bound method is
always truthy, so `cl.is_marked or not cl_del.is_marked()` short-circuits to `True`
unconditionally. Line 222, in the same function, correctly calls `cl.is_marked()` — the
omission on line 224 is an inconsistency, not a style choice.

**(b) Genuine defect?** Yes — a Python-level slip. The paper (§2.8, *GA Deletion*)
specifies the intended ordering: *"marked classifiers are preferred for deletion before
unmarked classifiers and the least applied classifier is preferred among only marked or
only unmarked classifiers."* The intended condition `cl.is_marked() or not
cl_del.is_marked()` exists to gate the `tav` (least-applied) tie-break to same-markedness
cases, and to **exclude** the cross case (`cl` unmarked, `cl_del` marked) — where the
already-marked `cl_del` should remain the deletion candidate. The bug makes the `elif`
always fire.

**(c) Behavioural consequence:** Real but narrow. It changes deletion only when qualities
are within 0.1, `cl` is unmarked, and `cl_del` is marked. Correct code returns `False`
(keep preferring the marked one); buggy code falls through to the `tav` check and may
switch the deletion target to the **unmarked** classifier if `cl.tav > cl_del.tav` —
contradicting "marked preferred for deletion." This biases the GA tournament to
occasionally delete a good (unmarked) classifier over an inaccurate (marked) one.
Stochastic and confined to near-tie cases, so it perturbs population quality rather than
breaking learning.

**Confidence: High.** Mechanism, intent, and paper anchor all confirmed.

---

## Claim 2 — `ClassifiersList.copy()` copies `self.__slots__`, not the classifiers

**(a) Does the code do what's described?** Yes.
`pyalcs/lcs/agents/acs2/ClassifiersList.py:58-61`:
```python
def copy(self):
    new_copy = ClassifiersList()
    new_copy.items = self.__slots__.copy()  # lub inna logika kopiowania
    return new_copy
```
`self.__slots__` resolves to the base `TypedList.__slots__` = `['_items', 'oktypes']`
(`TypedList.py:10`). `.copy()` returns the list of those two **strings**. It is assigned
to `new_copy.items` — but the real backing store is `_items` (with underscore), and there
is no `items` property; subclasses lack `__slots__`, so this just creates a stray,
never-read instance attribute. The trailing Polish comment ("or other copy logic")
confirms this was a placeholder.

**(b) Genuine defect?** Yes. `new_copy` is returned with an empty `_items`; the
classifiers are never copied. No crash (`.copy()` on a list is valid), so it fails
silently.

**(c) Behavioural consequence — conditional, inert by default.** `copy()` is invoked only
in `ACS2VCPv2/v3/v4` (`ACS2HER(cfg, base_population.copy())`). In all three,
`base_population = population or ClassifiersList()`, and the experiment scripts instantiate
agents without a `population` argument, so `base_population` is **empty**. A correct copy
of an empty list is also an independent empty list — so in normal use the bug is
**behaviourally indistinguishable from correct** and inert. It would silently drop
classifiers (each ensemble head starting empty instead of seeded) **only if a non-empty
population were ever passed in and copied** — which does not happen in the present call
sites. (The main `ACS2VCP.py` doesn't call `copy()` at all — it shares one population
across heads, a separate non-claimed issue.)

**Confidence: High** that it's a defect; **High** that it's inert under current usage. To
raise the consequence to "harmful," one would need a call site that copies a populated
list.

---

## Claim 3 — `_run_trial_exploit` does `range(self.ensemble_heads)` over a list → TypeError

**(a) Does the code do what's described?** Yes.
`pyalcs/lcs/agents/acs2vcp/ACS2VCP.py:133`: `for i in range(self.ensemble_heads):`.
`self.ensemble_heads` is a **list** built at `ACS2VCP.py:35-38`
(`[ACS2HER(...) for _ in range(ensemble_size)]`). `range()` requires an integer; passing a
list raises `TypeError: 'list' object cannot be interpreted as an integer` on loop entry,
before any exploit step runs.

**(b) Genuine defect?** Yes, unambiguously. The sibling method `_run_trial_explore` does it
correctly: `for i in range(len(self.ensemble_heads))` (`ACS2VCP.py:69, 122`). So the
missing `len(...)` is a clear slip, not intent.

**(c) Behavioural consequence:** The exploit path of the **base** `ACS2VCP` class crashes
immediately whenever reached (via `Agent.exploit` / `explore_exploit`, which call
`_run_trial_exploit` — `Agent.py:70,93`). Note the maze experiment scripts instantiate
`ACS2VCPv10`, not the base `ACS2VCP`, so this particular crash is not exercised by those
scripts — but the base class's exploit path is non-functional.

**Confidence: High.**

---

## Claim 4 — `apply_alp` removes from the action set while iterating it, skipping the next classifier

**(a) Does the code do what's described?** Yes.
`pyalcs/lcs/agents/acs2/ClassifiersList.py:146-163`:
```python
for cl in action_set:
    cl.increase_experience()
    cl.update_application_average(time)
    if cl.does_anticipate_correctly(p0, p1):
        ...
    else:
        new_cl = alp_acs2.unexpected_case(cl, p0, p1, time)
        if cl.is_inadequate():
            ...
            lists = [x for x in [population, match_set, action_set] if x]
            for lst in lists:
                lst.safe_remove(cl)        # removes cl from action_set being iterated
```
Two precise corrections to the claim's wording, both of which strengthen rather than
weaken it:

- **It is not a "list iterator."** `action_set` is a `TypedList`
  (`collections.abc.MutableSequence`), and its `__iter__` (`TypedList.py:34`) delegates to
  `Sequence.__iter__`, the **index-based generator** (`i=0; while True: yield self[i];
  i+=1`). Removing the element at the current index `i` shifts the tail left; the generator
  then reads `self[i+1]`, so the element that slid into slot `i` is never yielded. Same
  skip the claim describes, by an even clearer mechanism.
- **`safe_remove` removes by value/equality, not identity.** `safe_remove` →
  `MutableSequence.remove` → `index`, which finds the first element where `self[k] == o`;
  `Classifier.__eq__` compares condition-action-effect (`acs/Classifier.py:54-60`). In the
  normal case (unique C-A-E within an action set) this removes `cl` itself and the skip
  falls on its immediate successor. If duplicate C-A-E existed it could remove a different
  equal element, but a one-element removal during index iteration still produces a skip.

**(b) Genuine defect?** Yes, and the paper is explicit. §2.6 (*Application of the ALP*)
prose: *"Deleted classifiers need to be deleted from the action set **without influencing
the update process**."* The `APPLY ALP` pseudocode loops *"for each classifier cl in [A]"*
(line 1) — every member is to be processed: `cl.exp++`, update application average,
anticipation check, and expected/unexpected handling. The code's in-place removal during
index iteration is exactly the "influence on the update process" the paper warns must be
avoided. The loop's evident intent (process all of `[A]`) and the spec both say every
member must be handled; the skip diverges. (The paper notes the pseudocode "does not
address such details" — but flags them as problems implementers must solve, which this
code does not.)

**(c) Behavioural consequence:** Real but bounded. When a classifier is found inadequate
(`q < theta_i`) and deleted, its successor in the action set is skipped for *this* ALP pass
only — it gets no experience increment, no application-average update, no anticipation
check, and no possible marking/specialization. The skipped classifier remains in the
population/action set, so it is processed in subsequent passes; the loss is one pass, not
permanent removal. Triggers only when (i) a classifier is inadequate (relatively rare —
requires quality below threshold) and (ii) it has a successor in the set. So it perturbs
learning at the margins rather than corrupting it.

**Confidence: High** on the mechanism and the paper-backed intent; **Medium-High** on it
being a "defect" versus an accepted simplification — I cannot consult upstream pyalcs, but
the attached paper's "without influencing the update process" makes the code's behaviour a
divergence from the specified algorithm. What would raise it to fully High: a unit test
demonstrating a known-inadequate-then-successor action set where the successor's
`exp`/quality is observably stale after one ALP pass.

---

## Summary

| # | Claim | Verdict | Consequence | Confidence |
|---|-------|---------|-------------|------------|
| 1 | Bare `cl.is_marked` in GA tie-break | **Confirmed defect** | Marked-preference tie-break subtly broken; can delete unmarked over marked | High |
| 2 | `copy()` copies `__slots__` not classifiers | **Confirmed defect** | Returns empty list; **inert in default use** (sources are empty); harmful only if a populated list is copied | High |
| 3 | `range(self.ensemble_heads)` on a list | **Confirmed defect** | `TypeError` crashes base-class exploit path on entry (not hit by maze scripts, which use v10) | High |
| 4 | Remove-during-iteration skips next classifier | **Confirmed defect** | Successor of a deleted inadequate classifier misses one ALP pass; bounded, transient; paper requires every member processed | High (mechanism) / Med-High (defect-vs-accepted) |

All four claims are accurate at the code level. The only material corrections are to
claim 2's *consequence* (inert under current usage, not "heads get empty populations") and
to claim 4's *mechanism wording* ("Sequence index-based iteration," and `safe_remove` by
value/equality, not "list iterator"/identity) — neither changes the verdict.
