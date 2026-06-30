---

name: research-close
description: Close a substantive experiment, code change, mathematical derivation, or research idea with evidence inspection, adversarial audit, a persistent session report, and a concise Korean briefing for the researcher.
disable-model-invocation: true
argument-hint: "[experiment|code|theory|idea] [task ID or short description]"
-----------------------------------------------------------------------------

# Research Close Protocol

Close the task described by:

$ARGUMENTS

Do not declare the task complete until this protocol is finished.

## 1. Identify the task and evidence

First identify:

* task type: experiment / code / theory / idea
* exact claim or question addressed
* relevant changed files
* relevant config, seed, checkpoint, split, command, and result-file paths
* whether the evidence is oracle, held-out oracle, proxy, or only a hypothesis

Read these before interpreting the result:

1. `docs/ACTIVE_RESEARCH_STATE.md`
2. `docs/formalization_notes.md`
3. Relevant scripts, configs, diffs, logs, CSVs, and session reports

Do not infer numbers, configurations, or implementation details that are not present in files.

## 2. Produce an initial evidence-based conclusion

Before the audit, write an internal draft containing:

* the narrowest conclusion supported by the evidence;
* the exact evidence supporting it;
* the strongest alternative explanation;
* what remains unverified.

For experiments, compare candidate and baseline under the same resource budget whenever possible.

## 3. Run the independent adversarial audit

Delegate an independent review to the `research-adversary` subagent.

Give the subagent:

* the task type;
* the exact claim being considered;
* relevant result files, config files, code paths, and diffs;
* the initial conclusion;
* any known limitations.

The audit must happen before the final user-facing conclusion.

If the subagent cannot run or lacks required evidence, mark the task as:

`INCOMPLETE — adversarial audit unavailable or insufficient evidence`

Do not silently skip the audit.

## 4. Reconcile the audit

After the audit:

* correct any false, overstated, or ambiguous conclusion;
* distinguish verified fact, inference, assumption, and open question;
* preserve critical audit findings even when they weaken the result;
* do not present a negative audit finding as a minor caveat if it invalidates the main claim.

## 5. Write the persistent session report

Create one file:

`docs/session_reports/YYYY-MM-DD_<short_slug>.md`

Use this format:

# Session Report

## Task

* Task type:
* Research question:
* Claim tested:

## What changed or was tested

* Code/config/formalization changes:
* Experiment command:
* Model/checkpoint:
* Dataset and split:
* Calibration / held-out protocol:
* Seed:

## Evidence

* Result files:
* Relevant code paths:
* Relevant config paths:
* Exact key values:

## Main result

* Narrow supported conclusion:
* What the result does not establish:

## Important metrics

| Metric | Baseline | Candidate | Difference | Interpretation |
| ------ | -------: | --------: | ---------: | -------------- |

Only include metrics necessary to judge the research question.

For decision-preservation experiments, include when available:

* requested and actual memory saving;
* accuracy;
* flip rate;
* signed-mean risk;
* signed-p95 risk;
* held-out oracle result;
* proxy-versus-oracle ranking metric;
* seed and split identity.

## Adversarial audit

* Verdict: PASS / PASS WITH CAVEATS / FAIL / INCOMPLETE
* Critical findings:
* Non-critical concerns:
* Unsupported or overstated claims:
* Required correction or follow-up:

## Recommended next action

* One highest-priority next action:
* Why it is the priority:
* Success criterion:

## Canonical-document update needed?

* `ACTIVE_RESEARCH_STATE.md`: Yes / No
* `formalization_notes.md`: Yes / No
* `COMPRESSION_PLANNING_AGENT_HANDOFF.md`: Yes / No
* Reason:

Do not edit canonical documents in this skill unless the current session is explicitly assigned as the integrator.

## 6. Give the researcher a concise Korean briefing

After writing the session report, respond in Korean using exactly this structure:

# 실험/작업 결과

## 한 줄 결론

* [Supports / Does not support / Inconclusive] + the narrowest valid conclusion.

## 무엇을 했는가

* 최대 3개 bullet.
* 구현·실험·수식 변경을 평이한 말로 설명.

## 중요한 결과만

* 최대 5개 수치.
* 반드시 baseline 대비값 또는 차이를 함께 제시.
* 숫자마다 어느 결과 파일에서 나왔는지 짧게 표시.

## 이 결과가 의미하는 것

* 연구 가설과 직접 연결해 설명.
* accuracy와 decision preservation을 혼동하지 말 것.
* oracle, held-out oracle, proxy를 명확히 구분할 것.

## 아직 말하면 안 되는 것

* 과장 가능성, 실험 한계, audit에서 발견된 위험을 명시.

## 다음 행동

* 가장 우선순위가 높은 실험 또는 검증 1개만 제안.
* 성공/실패를 어떻게 판정할지 함께 제시.

Never bury a critical failure under optimistic wording.
Never claim generalization from one model, one seed, one split, or one pilot result.
