---

name: adversarial-audit
description: Perform an independent adversarial audit of a completed experiment, code change, mathematical derivation, or research claim. Use before declaring substantive work complete.
argument-hint: "[experiment|code|theory|idea] [target files or claim]"
----------------------------------------------------------------------

# Adversarial Audit

Audit the completed work as if the goal were to disprove its conclusion.

## Rules

* Do not trust the main worker's interpretation without checking files, code, results, and assumptions.
* Separate: verified fact / inference / assumption / unsupported claim.
* Do not edit canonical documentation during the audit.
* Produce a concise audit report before any user-facing conclusion.

## Universal checks

1. Identify the exact claim being made.
2. Identify the evidence required for that claim.
3. Inspect whether the provided evidence actually supports it.
4. Search for contradictory evidence, missing controls, leakage, hidden assumptions, or implementation mismatches.
5. State whether the claim is:

   * supported,
   * partially supported,
   * unsupported,
   * contradicted,
   * not evaluable from available evidence.
6. List the minimum corrective action for every critical issue.

## Experiment audit

Check:

* Dataset split, calibration split, held-out split, and possible leakage.
* Model checkpoint, seed, preprocessing, quantizer configuration, and action definition.
* Whether oracle, held-out oracle, and proxy are distinguished correctly.
* Whether reported numbers match actual result files.
* Whether accuracy, flip rate, mean risk, p95 risk, saving ratio, and rank metrics are interpreted correctly.
* Whether a single run, model, or split is being overgeneralized.
* Whether baselines are fair and use the same resource budget.
* Whether mean behavior is hiding tail-risk failures.
* Whether the selected plan was tuned on the same split used for evaluation.

## Code audit

Check:

* Diff correctness and unintended file changes.
* Unit tests, smoke tests, and reproducibility.
* Metric implementation against `docs/formalization_notes.md`.
* Silent fallback behavior, default settings, device differences, and numerical edge cases.
* Whether a claimed feature is genuinely implemented rather than only partially wired.

## Theory audit

Check:

* Definitions and notation consistency.
* Equality versus approximation versus sufficient condition.
* Hidden assumptions: fixed ReLU masks, differentiability, independence, boundedness, or ignored cross terms.
* Counterexamples or boundary cases.
* Whether the theorem/claim is actually proved or only motivated.
* Whether the conclusion overstates what the derivation establishes.

## Research-idea audit

Check:

* Precise problem definition.
* Falsifiability and measurable success criterion.
* Existing baseline explanation.
* Novelty versus merely renaming an existing objective.
* Whether the idea needs a new algorithm, evidence, or theory to become a contribution.

## Required output

Write a report containing:

1. Claim audited
2. Evidence inspected
3. Critical findings
4. Non-critical concerns
5. Unsupported or overstated statements
6. Corrected conclusion
7. Required follow-up experiment, proof, or code check
8. Final verdict: PASS / PASS WITH CAVEATS / FAIL / INCOMPLETE
