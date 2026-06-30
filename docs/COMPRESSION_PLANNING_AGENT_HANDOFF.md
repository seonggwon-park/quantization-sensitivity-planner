# Compression Planning Research Handoff

작성 시점: 2026-06-30.  
범위: 이 repository 안에서 확인 가능한 문서, 코드, 결과 파일, git history만 근거로 정리했다. 확인되지 않은 대화 export, 외부 논문, 사용자의 비공개 의도는 만들지 않았다.

## 0. Agent Start Here

이 연구는 **binary CIFAR-10 ResNet-18에서 mixed-precision weight-only fake quantization action을 layer별로 선택해 parameter memory saving을 얻되, FP32 모델의 binary decision을 최대한 보존하는 compression planning 문제**로 현재 정식화되어 있다.

현재 가장 신뢰할 수 있는 주장은 다음이다.

| Claim | 상태 | 근거 |
| --- | --- | --- |
| Uniform int4는 약 87.4% parameter-memory saving을 주지만 binary decision flip이 크게 증가한다. | 실험됨 | `results/uniform_baselines.csv`; `docs/experiment_log.md`, uniform fake quantization baseline |
| 초기 pilot에서는 conv1 int4가 가장 위험한 single-layer action이며, conv1을 FP32로 보호하면 uniform int4 대비 flip/risk가 크게 줄었다. | 실험됨 | `results/validation_single_layer_sweep.csv`; `results/oracle_guided_mixed_controls.csv` |
| weight L2 norm은 layer/action decision risk ranking proxy로 약하다. 특히 old pilot에서 weight L2 top/bottom control이 oracle-like ranking과 어긋났고, go/no-go ranking에서도 decision-risk target과 correlation이 낮다. | 실험됨 | `results/weight_l2_mixed_controls.csv`; `results/go_no_go/metric_ranking_summary.csv` |
| held-out oracle single-action benchmark에서 score-set `decision_risk_p95`는 `oracle_decision_risk_p95` ranking과 매우 잘 맞았다. | 부분 검증 | `results/go_no_go/metric_ranking_summary.csv`, row: `score_metric=decision_risk_p95`, `oracle_target=oracle_decision_risk_p95` |
| frozen 3-seed protocol의 descriptive aggregate에서 `vector_signed_mean_risk` planner는 18개 seed-budget cell 모두에서 six scalar baselines보다 locked-test p95 decision risk가 낮았다. | 가장 강한 현재 실험 결과, 단 통계적 유의성 아님 | `results/repro_v1/aggregate_v1/aggregate_summary.csv`; `results/repro_v1/aggregate_v1/per_seed_budget_results.csv`; `experiments/go_no_go/aggregate_frozen_repro_results.py` |

현재 구현 상태:

- 모델/데이터: ImageNet-pretrained ResNet-18을 CIFAR-10 class 0/1, 즉 airplane/automobile binary classifier로 fine-tune한다. 근거: `config.py`, `data.py`, `model.py`, `train.py`.
- Compression action: Conv2d/Linear weight만 대상으로 `fp32`, `fp16`, `int8`, `int4` fake quantization을 적용한다. Bias, BatchNorm, 기타 parameter는 FP32 memory로 계산한다. 근거: `quantization.py`, `additive_planner.py`, `experiments/go_no_go/adapters.py`.
- Planner: old additive knapsack planner, go/no-go scalar planner, samplewise score-delta vector beam planner가 구현되어 있다. 근거: `additive_planner.py`, `experiments/go_no_go/planner_eval.py`, `experiments/go_no_go/vector_beam_planner.py`.
- Evaluation split: score/calibration set, development oracle, confirmation split, locked test split의 역할이 분리되어 있다. 근거: `experiments/go_no_go/splits.py`, `experiments/go_no_go/make_confirmation_split.py`, `experiments/go_no_go/evaluate_plans_on_confirmation.py`, `experiments/go_no_go/evaluate_plans_on_test.py`.
- 아직 구현되지 않은 것: structured pruning action, hardware latency measurement, real quantized inference backend, multi-class extension, formal perturbation bound, sequential/interaction recalibration.

가장 큰 미해결 문제:

1. `vector_signed_mean_risk` 우위가 binary CIFAR-10 ResNet-18 fake-quant setting 밖에서도 유지되는가. 상태: 불명.
2. vector planner가 실제 layer interaction을 설명하는가, 아니면 score-set에서 signed delta cancellation을 이용한 좋은 heuristic인가. 상태: 부분 검증.
3. fake quantization memory saving ratio가 실제 hardware latency/memory saving으로 이어지는가. 상태: 미구현.
4. decision preservation objective가 accuracy-only objective보다 어떤 사용 사례에서 실질적으로 중요한가. 상태: 실험 맥락상 타당, 일반 주장은 보류.
5. score/calibration set 크기와 seed에 대한 planner 안정성이 충분한가. 상태: score seed 0/1/2 및 frozen seed 101/202/303까지 부분 검증.
6. theoretical bound, perturbation propagation, last-layer margin bound가 구현된 planner와 어떻게 연결되는가. 상태: formalization note에 analytical derivation은 있으나 active planner guarantee로는 미검증.
7. structured pruning까지 action set을 넓힐 때 risk/memory/latency objective가 같은 방식으로 작동하는가. 상태: 미구현.

다음 에이전트가 먼저 읽을 파일:

1. `CLAUDE.md` - research protocol. 단, 여기서 mandatory first로 지정한 `docs/ACTIVE_RESEARCH_STATE.md`는 현재 존재하지 않는다.
2. `docs/formalization_notes.md` - 수학적 source-of-truth, 단 code와 다른 signed-risk 정의가 있으니 section 4의 충돌 note와 함께 읽을 것.
3. `experiments/go_no_go/README.md` - 현재 go/no-go protocol의 목적과 adapter map.
4. `experiments/go_no_go/metrics.py` - 현재 code-level decision risk 정의.
5. `experiments/go_no_go/run_single_action_benchmark.py` - single-action benchmark와 score/oracle split 사용.
6. `experiments/go_no_go/planner_eval.py` - scalar additive planner evaluation.
7. `experiments/go_no_go/collect_score_delta_vectors.py` 및 `experiments/go_no_go/vector_beam_planner.py` - vector proxy와 vector planner.
8. `experiments/go_no_go/evaluate_plans_on_confirmation.py` 및 `experiments/go_no_go/evaluate_plans_on_test.py` - confirmation/test policy.
9. `experiments/go_no_go/aggregate_frozen_repro_results.py` - 현재 가장 강한 aggregate claim의 계산 방식.
10. `results/repro_v1/aggregate_v1/aggregate_summary.csv` 및 `results/repro_v1/aggregate_v1/per_seed_budget_results.csv` - 최종 숫자.
11. `docs/experiment_log.md` - 2026-06-23 초기 연구 흐름.

작업 전 위험 요소:

- `margin_normalized_risk`와 현재 `decision_risk`는 다르다. 이전 결과를 현재 risk claim으로 섞지 말 것.
- `oracle`, `held-out oracle`, `confirmation`, `locked test`, `proxy/score set`을 혼동하지 말 것.
- planner output의 proxy risk와 post-action full forward 결과를 같은 값처럼 쓰지 말 것.
- `vector_signed_mean_risk`는 globally optimal solver가 아니라 deterministic heuristic beam search다.
- `actual_saving`은 parameter storage proxy이며 latency가 아니다.
- locked test는 terminal evaluation으로 기록되어 있다. test result로 method/hyperparameter를 다시 고르면 protocol 위반이다.

## 1. Source Map and Evidence Coverage

| Source | Type | Research relevance | Used sections | Notes |
| ------ | ---- | ------------------ | ------------- | ----- |
| `CLAUDE.md` | Markdown research protocol | mandatory reading order, source-of-truth hierarchy, parallel-session rules, integrity rules | 0, 1, 14, 15 | References `docs/ACTIVE_RESEARCH_STATE.md`, but that file is missing in current workspace. |
| `docs/experiment_log.md` | Markdown experiment log | 2026-06-23 초기 training, uniform quantization, single-layer sweep, oracle controls, metric controls, activation proxy, additive planner 흐름 | 0, 2, 3, 7, 8, 9 | 자동 기록된 command/config/result가 포함됨. `results/experiment_history.jsonl`와 중복된다. |
| `docs/formalization_notes.md` | Markdown formalization note | 수학적 source-of-truth 역할. binary decision preservation, signed margin loss, last-layer bound, perturbation recurrence, oracle/proxy distinction, structured pruning/latency deferred status | 0, 2, 3, 4, 5, 9, 10, 11, 15 | Verification 중 untracked file로 발견되어 읽고 반영했다. 일부 apostrophe/dash 문자가 mojibake로 깨져 있으나 equations/status tags는 해석 가능했다. |
| `results/experiment_history.jsonl` | JSONL experiment log | `docs/experiment_log.md`의 machine-readable duplicate | 1, 15 | 원시 event log로 존재하지만 문서 본문에는 `docs/experiment_log.md`를 주 근거로 사용했다. |
| `config.py`, `data.py`, `model.py`, `train.py` | Python source | 모델, binary dataset, train/validation/test split, checkpoint protocol | 4, 6, 13 | CIFAR-10 class 0/1, image_size 224, ImageNet normalization, ResNet-18 binary head. |
| `quantization.py`, `metrics.py`, `activation_proxy.py` | Python source | fake quantization semantics, old margin risk, activation proxy | 4, 5, 6, 9 | old pipeline의 metric과 현재 go/no-go metric 차이를 구분해야 한다. |
| `additive_planner.py`, `run_additive_planner.py`, `risk_configurations.py` | Python source | old additive mixed-precision planner | 3, 4, 6, 7, 10 | Multiple-choice knapsack DP. Layer/action interaction은 additive risk로 단순화. |
| `run_baselines.py`, `sweep_single_layer.py`, `run_oracle_guided_controls.py`, `run_ranked_controls.py`, `run_activation_proxy.py`, `plot_results.py` | Python source | 초기 실험 재현 command와 metric 산출 방식 | 6, 7, 8, 9 | `plot_results.py`는 figure 생성 쪽이며 본 handoff의 핵심 numeric claim에는 사용하지 않았다. |
| `experiments/go_no_go/README.md` | Markdown protocol doc | go/no-go benchmark 목적, adapter map, quantizer semantics, output contract | 0, 1, 3, 6, 14 | checkpoint 변경 없음, planner 변경 없음이라는 benchmark scope를 명시한다. |
| `experiments/go_no_go/metrics.py` | Python source | 현재 decision risk, output KL, ranking helper 정의 | 4, 5, 9 | `decision_risk = max(0, -direction*delta_score)/(margin+eps)`. |
| `experiments/go_no_go/splits.py`, `make_confirmation_split.py` | Python source | score/oracle/confirmation split generation | 4, 6, 7 | Class-balanced, overlap-free split. |
| `experiments/go_no_go/run_single_action_benchmark.py`, `analyze_rankings.py` | Python source | held-out single-action benchmark와 rank consistency analysis | 6, 7, 8, 9 | Spearman, Kendall tau-b, top-k recall. |
| `experiments/go_no_go/planner_eval.py` | Python source | scalar proxy planner evaluation on oracle set | 6, 7, 8, 9 | Rank normalization 후 additive solver 사용. |
| `experiments/go_no_go/collect_score_delta_vectors.py`, `analyze_vector_additive_proxy.py`, `vector_beam_planner.py` | Python source | samplewise score-delta vector proxy와 vector beam planner | 4, 5, 6, 7, 8, 9 | vector proxy는 interaction update를 모델링하지 않는다고 payload/comment에 명시. |
| `experiments/go_no_go/evaluate_plans_on_confirmation.py`, `evaluate_plans_on_test.py` | Python source | fixed plan confirmation/test evaluation policy | 6, 7, 8, 11 | confirmation/test data는 plan generation이나 method selection에 사용하지 않는다고 명시. |
| `experiments/go_no_go/train_frozen_binary_seed.py`, `run_frozen_repro_pipeline.py`, `aggregate_frozen_repro_results.py` | Python source | frozen multi-seed training, reproducibility pipeline, aggregate claim | 6, 7, 8, 11, 13 | 현재 가장 강한 result의 protocol과 caveat. |
| `results/*.csv` | CSV results | 초기 baseline/sweep/control/planner numeric results | 7, 8, 9 | 75 CSV 중 flat results와 go/no-go/repro summaries를 중심으로 수치 보존. |
| `results/go_no_go/**/*.csv`, `results/go_no_go_planner_*/*.csv`, `results/go_no_go_vector_*/*.csv` | CSV results | go/no-go benchmark, scalar planner, vector analysis/plans | 7, 8, 9, 11 | pilot result. Frozen repro보다 claim strength 낮음. |
| `results/repro_v1/**/*.csv`, `results/repro_v1/**/*.json`, `pipeline_commands.txt`, `pipeline_manifest.json` | CSV/JSON/TXT results | frozen 3-seed protocol, locked-test aggregate | 0, 6, 7, 8, 11, 13 | `seed_101/202/303` and `aggregate_v1`. |
| `results/go_no_go/split_indices.json`, `results/go_no_go_confirmation_v1/confirmation_split_indices.json` | JSON metadata | split size/count/overlap policy | 4, 6, 7 | Raw arrays are long; this handoff records summary metadata. |
| `.git` history | Git log | chronological branch/commit trace | 2, 15 | Local git history accessible. No commit/branch changes made for this handoff. |
| `private/` | Directory | possible ChatGPT export/personal notes | 1 | Directory exists but is empty. No private conversation evidence found. |
| `.agents/` | Directory | possible agent notes | 1 | Directory exists but is empty. |

Input-source coverage counts from repository scan excluding `.git`, `.venv`, `data`, and this generated handoff file: `.py 34`, `.csv 75`, `.json 280`, `.jsonl 1`, `.md 4`, `.txt 5`, `.yml 1`. 근거: local file inventory command; `rg --files`; `Get-ChildItem` counts.

읽지 못했거나 접근하지 못한 자료:

- `data/cifar-10-batches-py` traversal에서 access denied가 발생했다. 이 디렉터리는 raw dataset artifact로 보이며 연구 문서/결과 파일은 아니다. 상태: 접근 불가.
- `docs/ACTIVE_RESEARCH_STATE.md`는 `CLAUDE.md`가 canonical current task state로 지정하지만 현재 workspace에 존재하지 않는다. 상태: referenced but missing.
- `.venv/`는 의도적으로 제외했다. 상태: 비연구 dependency directory.
- `checkpoints/*.pt`는 binary checkpoint라 직접 decode하지 않았다. training/evaluation metadata는 manifest/log/CSV에서 확인했다. 상태: binary artifact 미해석.
- `.npz` score/split/vector binary artifacts는 JSON sidecar와 code schema로 검증했지만 전체 array를 본문에 풀어 쓰지는 않았다. 상태: 부분 해석.
- `private/`와 `.agents/`는 빈 폴더였다. ChatGPT export, conversation JSON, 개인 메모는 발견되지 않았다.
- `claude_quantization_research_handoff/` 또는 유사한 handoff directory는 발견되지 않았다.
- root `README.md`, notebooks, `.tex` 연구 문서는 발견되지 않았다.

손상되었거나 형식을 해석할 수 없었던 자료:

- 손상된 CSV/JSON은 발견하지 못했다.
- `results/go_no_go/split_indices.json`는 정상 JSON이나 raw index array가 길어 본문에는 summary만 사용했다.
- `docs/formalization_notes.md`는 일부 영어 punctuation/possessive가 mojibake로 깨져 있다. 예: apostrophe 주변 문자가 `?셲`처럼 보인다. 수식, section title, status tag는 해석 가능했으며 본문에는 깨진 문자를 그대로 인용하지 않았다.

중복 자료:

- `docs/experiment_log.md`와 `results/experiment_history.jsonl`는 초기 실험 기록이 중복된다.
- `results/go_no_go_analysis_v1/metric_ranking_*.csv`는 `results/go_no_go/metric_ranking_*.csv`와 같은 ranking analysis 계열이다.
- `results/go_no_go_smoke*`는 smoke/small runs로 full benchmark보다 신뢰도가 낮다.
- pilot `results/go_no_go_*`와 frozen `results/repro_v1/seed_*`는 같은 pipeline 계열이지만 checkpoint/protocol strength가 다르다.

서로 다른 파일 간 충돌 또는 해석상 주의:

1. Old `margin_normalized_risk`와 current `decision_risk`는 수식이 다르다. 근거: `metrics.py`, `experiments/go_no_go/metrics.py`.
2. `oracle`이라는 단어가 초기 문서에서는 validation/test single-layer empirical sweep을 뜻하고, go/no-go에서는 train-pool held-out development oracle set을 뜻한다. 근거: `docs/experiment_log.md`; `experiments/go_no_go/run_single_action_benchmark.py`.
3. weight L2 dense planner는 특정 budget에서 나쁜 spike를 보이지만 neighboring budgets는 낮은 risk를 보인다. 일반화된 단순 결론으로 쓰면 위험하다. 근거: `results/weight_l2_dense_results.csv`.
4. Pilot confirmation/test result와 frozen repro aggregate는 claim strength가 다르다. 최종 주장은 frozen aggregate를 우선해야 한다. 근거: `results/go_no_go_test_eval_v1/test_primary_summary.csv`; `results/repro_v1/aggregate_v1/aggregate_summary.csv`.
5. `memory_saving_ratio`와 `actual_saving`은 parameter-storage proxy이지 latency나 hardware memory footprint가 아니다. 근거: `experiments/go_no_go/README.md`; `quantization.py`; `additive_planner.py`.
6. `vector_signed_p95_risk`는 intuitive tail-risk objective처럼 보이지만 frozen aggregate에서 `vector_signed_mean_risk`보다 우수하다고 말할 수 없다. 근거: `results/repro_v1/aggregate_v1/aggregate_summary.csv`.
7. `docs/formalization_notes.md`의 signed margin-loss risk \(R^{sgn}\)는 음수도 허용하고 \(R \ge 1\)을 flip threshold로 두는 수학적 working definition이다. 현재 go/no-go code의 `decision_risk`는 harmful erosion만 `clamp_min(0)`으로 남기는 nonnegative metric이다. 두 수식을 같은 이름으로 쓰면 안 된다. 근거: `docs/formalization_notes.md`, sections 3.2 and 12.2; `experiments/go_no_go/metrics.py`.
8. `CLAUDE.md` says `docs/ACTIVE_RESEARCH_STATE.md` is canonical current task state, but the file is missing. This handoff therefore cannot incorporate that canonical current-state document. 근거: `CLAUDE.md`; `Test-Path docs/ACTIVE_RESEARCH_STATE.md` returned false.

## 2. Chronological Research History

### 2026-05-11 — Last-layer binary decision bound

- 문제/질문: binary classifier의 final decision이 final-layer quantization perturbation 아래에서 언제 보존되는가.
- 당시 가설: binary score sign이 바뀌지 않으면 decision이 보존되며, final-layer weight perturbation이 score margin보다 작으면 no-flip sufficient condition을 줄 수 있다.
- 제안된 접근: final score \(s(x)=w^\top h_{L-1}(x)+b\), quantized final weight \(\tilde w=w+\Delta w\), score perturbation \(\delta s(x)=\Delta w^\top h_{L-1}(x)\)로 두고 \(|\delta s(x)|<|s(x)|\)이면 sign이 보존된다는 bound를 정리.
- 실제로 한 구현 또는 실험: 구현/실험이 아니라 analytical derivation으로 기록됨.
- 결과: \(|\delta s(x)| \le \|\Delta w\|_q\|h_{L-1}(x)\|_p\), \(|\delta s(x)|<|s(x)| \Rightarrow \operatorname{sgn}(\tilde s)=\operatorname{sgn}(s)\).
- 해석: decision preservation을 accuracy가 아니라 binary margin/sign preservation으로 다루는 초기 수학적 근거.
- 이후에 바뀐 결정: final layer만 다루면 earlier layer quantization에서 activation 자체가 변한다는 한계가 드러남.
- 현재 상태: Analytical derivation; active planner objective나 guarantee는 아님.
- 근거: `docs/formalization_notes.md`, sections 1.0 and 1.1.

### 2026-05-11 to 2026-05-21 — Earlier-layer perturbation recurrence

- 문제/질문: final-layer argument가 earlier-layer quantization에는 왜 그대로 적용되지 않는가.
- 당시 가설: earlier layer quantization은 local weight error와 propagated activation error를 동시에 만들며, cross term 때문에 단순 additive bound가 어렵다.
- 제안된 접근: \(u_\ell=W_\ell h_{\ell-1}+b_\ell\), \(\tilde W_\ell=W_\ell+\Delta W_\ell\), \(\tilde h_{\ell-1}=h_{\ell-1}+\delta h_{\ell-1}\)로 두고 exact recurrence를 전개.
- 실제로 한 구현 또는 실험: 구현/실험이 아니라 analytical derivation으로 기록됨.
- 결과: \(\delta u_\ell = W_\ell\delta h_{\ell-1}+\Delta W_\ell h_{\ell-1}+\Delta W_\ell\delta h_{\ell-1}\). Local linearization으로 \(\delta h_\ell\approx A_\ell(x)\delta h_{\ell-1}+e_\ell(a_\ell;x)\), ReLU fixed-mask special case \(A_\ell=D_\ell W_\ell\)가 제시됨.
- 해석: weight L2만으로 layer sensitivity를 판단하기 어렵고, downstream propagation/interaction을 고려해야 한다.
- 이후에 바뀐 결정: forward-only proxy와 interaction problem이 핵심 설계 문제로 올라옴.
- 현재 상태: Analytical derivation; active planner guarantee 아님.
- 근거: `docs/formalization_notes.md`, sections 1.2, 1.3.

### 2026-05-21 — Binary pilot and layer-level action abstraction

- 문제/질문: compression planning을 첫 pilot에서 tractable하게 만들려면 어떤 scope가 필요한가.
- 당시 가설: binary classification으로 decision boundary를 \(s(x)=0\) 하나로 두고, layer-level action units와 parameter memory accounting을 쓰면 검증 가능한 pilot이 된다.
- 제안된 접근: \(s(x)=z_1(x)-z_0(x)\), FP32 prediction \(\hat y_0=\mathbb 1[s_\theta(x)\ge 0]\), plan \(A=(a_1,\dots,a_L)\), action space \(\mathcal A=\prod_\ell\mathcal A_\ell\).
- 실제로 한 구현 또는 실험: later code가 binary CIFAR-10 ResNet-18 and layer-level Conv/Linear actions로 구현됨.
- 결과: binary scope and plan notation fixed in formalization note.
- 해석: multi-class/general pruning을 바로 주장하지 않고, binary decision preservation부터 검증하는 방향.
- 이후에 바뀐 결정: 2026-06-23 code pipeline에서 CIFAR-10 class 0/1 ResNet-18로 구현됨.
- 현재 상태: Fixed scope decision; 구현됨 in code.
- 근거: `docs/formalization_notes.md`, sections 1.4, 2.1, 2.2; `config.py`; `data.py`.

### 2026-06-23 — Forward-only proxy and interaction problem formalized

- 문제/질문: gradient/Jacobian/exhaustive re-evaluation 없이 planner가 어떻게 decision risk를 예측할 수 있는가.
- 당시 가설: analytical propagation은 동기를 주지만 active planner는 forward-only proxy여야 하며, single-layer risk addition에는 interaction residual이 있다.
- 제안된 접근: oracle risk \(R_A^{oracle}\), planner proxy \(\widehat R_A\), single-layer risk \(R_{\ell,a}^{single}\), additive surrogate \(\sum_\ell R_{\ell,a_\ell}^{single}\), interaction residual \(I_A(x)=R_A^{oracle}(x)-\sum_\ell R_{\ell,a_\ell}^{single}(x)\) 구분.
- 실제로 한 구현 또는 실험: 같은 날 code pipeline에서 activation proxy/additive planner가 구현되기 시작했고, 다음날 vector proxy가 구현됨.
- 결과: formalization note는 additive aggregation, conservative max-like aggregation, sequential recalibration, learned calibration map, vector propagation을 후보로 기록하되 confirmed method로 부르지 말라고 명시.
- 해석: 이후 vector signed proxy는 이 interaction problem에 대한 구현된 heuristic으로 볼 수 있다.
- 이후에 바뀐 결정: scalar additive planner의 한계를 vector score-delta planner로 보완.
- 현재 상태: Analytical/conceptual plus later partial implementation.
- 근거: `docs/formalization_notes.md`, sections 5.3, 8.1-8.4; `experiments/go_no_go/analyze_vector_additive_proxy.py`.

### 2026-06-23 19:58-20:03 — Automatic experiment logging and FP32 binary ResNet baseline

- 문제/질문: compression planning 실험을 재현 가능하게 기록할 수 있는 baseline pipeline이 필요한가.
- 당시 가설: binary CIFAR-10 ResNet-18 checkpoint를 먼저 안정적으로 만들면 quantization risk를 이후 단계에서 비교할 수 있다.
- 제안된 접근: ImageNet-pretrained ResNet-18의 final FC를 2-class head로 바꾸고 CIFAR-10 class 0/1만 사용한다.
- 실제로 한 구현 또는 실험: `train.py --epochs 10` 실행, automatic experiment logging 추가.
- 결과: validation accuracy 0.996, test accuracy 0.996, test loss 0.01517276841099374, test samples 2000, checkpoint `checkpoints\resnet18_binary_best.pt`.
- 해석: binary task에서는 FP32 teacher가 매우 높고 stable한 기준점이 되었다.
- 이후에 바뀐 결정: accuracy 자체보다 FP32 teacher decision을 보존하는지 평가할 수 있게 됐다.
- 현재 상태: 구현됨/실험됨.
- 근거: `docs/experiment_log.md`, section "train_binary_resnet18"; commit `be4ab74 docs: log FP32 ResNet training run`; `config.py`; `model.py`; `data.py`.

### 2026-06-23 20:13-20:16 — Uniform fake quantization baseline

- 문제/질문: uniform bit-width compression이 얼마나 많은 memory saving과 decision damage를 만드는가.
- 당시 가설: FP16/INT8은 거의 안전하고 INT4는 위험할 수 있다.
- 제안된 접근: 모든 quantizable Conv2d/Linear weight에 같은 bit-width fake quantization을 적용하고 FP32 model과 비교한다.
- 실제로 한 구현 또는 실험: `run_baselines.py --max-samples 2000`; bits 32/16/8/4.
- 결과: FP32 saving 0, accuracy 0.996, flip 0. FP16 saving 0.4995704778637299, flip 0. INT8 saving 0.7493557167955949, flip 0.0005. INT4 saving 0.8742483362615274, accuracy 0.846, flip 0.153, mean margin risk 0.7086520195007324, p95 1.3554645299911498.
- 해석: high compression regime에서 uniform INT4는 decision preservation을 크게 깨뜨린다. Mixed precision planning이 필요한 문제로 바뀐다.
- 이후에 바뀐 결정: uniform bit-width 선택이 아니라 layer별 bit allocation/action planning으로 이동.
- 현재 상태: 실험됨.
- 근거: `results/uniform_baselines.csv`; `docs/experiment_log.md`, "uniform_fake_quantization_baseline"; `run_baselines.py`; `quantization.py`.

### 2026-06-23 20:20-20:35 — Single-layer sensitivity sweep

- 문제/질문: INT4 uniform damage가 모든 layer에서 균등하게 발생하는가, 특정 layer가 더 위험한가.
- 당시 가설: layer별 sensitivity가 다르고, 위험한 layer만 FP32/고정밀로 보호하면 saving을 유지하면서 flip을 줄일 수 있다.
- 제안된 접근: 한 번에 하나의 layer만 quantize하여 FP32 teacher와 full-forward 비교한다.
- 실제로 한 구현 또는 실험: `sweep_single_layer.py` smoke 128 samples, 500 samples, full 2000 test samples.
- 결과: full test sweep에서 `conv1 int4`가 highest p95/flip: p95 1.2427469491958618, flip 0.098. 다른 layers는 대체로 훨씬 낮았다.
- 해석: first convolution이 binary decision에 매우 민감하다는 empirical signal이 생겼다.
- 이후에 바뀐 결정: test로 고른 layer를 다시 test에서 평가하는 leakage를 피하기 위해 validation sweep으로 옮김.
- 현재 상태: 실험됨; old metric 기반.
- 근거: `docs/experiment_log.md`, "single_layer_sweep"; `results/single_layer_sweep.csv`; commit `9631b39 docs: log full layer sensitivity sweep`.

### 2026-06-23 20:50-20:54 — Validation sensitivity and oracle-guided mixed controls

- 문제/질문: validation에서 고른 risky layers를 test에서 보호하면 uniform INT4보다 decision preservation이 좋아지는가.
- 당시 가설: validation single-layer oracle ranking top layers를 FP32로 유지하고 나머지를 INT4로 두면 flip/risk를 줄일 수 있다.
- 제안된 접근: validation sweep으로 top/bottom risk layers를 고르고 test set에서 mixed controls 평가.
- 실제로 한 구현 또는 실험: validation sweep 1000 samples; oracle top-k/bottom-k FP32 controls.
- 결과: validation `conv1 int4` p95 1.1971958875656128, flip 0.082. Test controls: uniform INT4 flip 0.153/p95 1.355465. `oracle_top1_fp32` saving 0.873511859230539, flip 0.0125, p95 0.628788. `oracle_top4_fp32` saving 0.867099, flip 0.0055, p95 0.353377.
- 해석: layer sensitivity ranking을 이용한 selective protection이 효과적이라는 초기 evidence.
- 이후에 바뀐 결정: oracle-like full forward sweep은 expensive하므로 proxy metric이 필요하다는 방향으로 이동.
- 현재 상태: 실험됨; 단 "oracle"은 현재 go/no-go의 held-out oracle과 의미가 다르다.
- 근거: `results/validation_single_layer_sweep.csv`; `results/oracle_guided_mixed_controls.csv`; `docs/experiment_log.md`, "oracle-guided mixed precision controls".

### 2026-06-23 21:29-21:31 — Weight L2 baseline controls

- 문제/질문: weight perturbation magnitude alone이 risky layer를 찾는 proxy가 될 수 있는가.
- 당시 가설: quantized weight의 relative L2 change가 크면 decision risk도 클 수 있다.
- 제안된 접근: weight relative L2 ranking top/bottom layers를 FP32로 보호하고 uniform INT4 baseline과 비교.
- 실제로 한 구현 또는 실험: metric-ranked mixed controls.
- 결과: `weight_l2_top1`은 `layer1.0.conv1`을 보호해 saving 0.871363, flip 0.134, p95 1.2759로 나쁨. 반대로 `weight_l2_bottom2`가 `conv1,fc`를 보호해 saving 0.873432, flip 0.012, p95 0.631028로 좋게 나왔다.
- 해석: weight L2 ranking은 위험도 proxy로 misaligned. 일부 bottom control이 우연히 critical layer를 보호했다.
- 이후에 바뀐 결정: activation/output/decision-aware proxy로 이동.
- 현재 상태: 실험됨; baseline으로 유지.
- 근거: `results/weight_l2_mixed_controls.csv`; `docs/experiment_log.md`, "weight L2 ranked mixed controls".

### 2026-06-23 21:43-21:50 — Local activation reconstruction proxy

- 문제/질문: full model forward oracle 없이 local forward만으로 risky layer를 찾을 수 있는가.
- 당시 가설: layer input을 FP32 model에서 얻고 quantized local module output reconstruction error를 측정하면 downstream decision risk와 더 잘 맞을 수 있다.
- 제안된 접근: hooks로 target layer input/output을 수집하고 local quantized module만 재실행해 per-sample relative activation L2 error를 계산한다.
- 실제로 한 구현 또는 실험: `activation_proxy.py`; smoke와 validation all-bits activation proxy; activation p95 ranking controls.
- 결과: `conv1 int4` local activation p95 0.23875348269939423로 highest. `local_activation_p95_top1`은 conv1 보호와 동일하여 flip 0.0125. `local_activation_p95_top4` saving 0.864213, flip 0.0065, p95 0.427720.
- 해석: activation proxy는 weight L2보다 oracle-like ranking에 가까운 early proxy로 보였다.
- 이후에 바뀐 결정: proxy metric을 planner objective에 넣는 additive mixed precision planner로 이동.
- 현재 상태: 구현됨/실험됨; 현재 final claim의 main metric은 아님.
- 근거: `activation_proxy.py`; `run_activation_proxy.py`; `results/validation_local_activation_proxy_all_bits.csv`; `results/local_activation_p95_mixed_controls.csv`.

### 2026-06-23 21:56-22:15 — Additive mixed-precision planner and dense budget comparison

- 문제/질문: layer별 action `{fp32,fp16,int8,int4}`를 memory budget 아래에서 자동 선택할 수 있는가.
- 당시 가설: risk를 layer/action별 additive cost로 두고 memory saving constraint를 걸면 knapsack planning으로 좋은 mixed precision plan을 찾을 수 있다.
- 제안된 접근: multiple-choice knapsack DP. Metric 후보는 weight L2, local activation p95, empirical margin p95.
- 실제로 한 구현 또는 실험: `additive_planner.py`; `run_additive_planner.py`; dense budget grid; allocation/result CSV.
- 결과: `weight_l2_dense_results.csv`에서 requested 0.8725, actual 0.872519 plan이 flip 0.1435, p95 1.316518로 매우 나빴다. 같은 budget 근처에서 `local_activation_dense_results.csv` requested 0.8725, actual 0.872570, flip 0.0055, p95 0.348429. `empirical_margin_dense_results.csv` requested 0.8725, actual 0.872524, flip 0.0040, p95 0.324819.
- 해석: planner framework 자체는 작동하지만 scalar metric 선택과 interaction/additivity 문제가 중요하다. Weight L2는 특히 brittle.
- 이후에 바뀐 결정: proxy metric이 held-out oracle ranking과 일관되는지 go/no-go benchmark로 검증하는 방향으로 이동.
- 현재 상태: 구현됨/실험됨; old pipeline.
- 근거: `additive_planner.py`; `results/weight_l2_dense_results.csv`; `results/local_activation_dense_results.csv`; `results/empirical_margin_dense_results.csv`; commit `92b914c docs: log dense planner comparison`.

### 2026-06-24 11:40-12:24 — Go/No-Go single-action decision-risk benchmark

- 문제/질문: proxy metric이 held-out oracle risk ranking과 일관되는가. Planner를 더 키우기 전에 go/no-go 기준이 필요한가.
- 당시 가설: decision preservation을 직접 반영한 score-set metric, 특히 tail risk,가 held-out oracle single-action risk와 잘 맞을 것이다.
- 제안된 접근: fixed class-balanced score sets와 disjoint development oracle set을 만들고, 모든 layer/action candidate 63개를 평가한다.
- 실제로 한 구현 또는 실험: `experiments/go_no_go/run_single_action_benchmark.py`; score seeds 0/1/2, score size 512, oracle size 2000; ranking analysis.
- 결과: vs `oracle_decision_risk_p95`, `decision_risk_p95` Spearman mean 0.998960, Kendall tau-b mean 0.984298, top5/top10 recall 1.0. `weight_rel_l2` tau 0.660010, top5 0.2, top10 0.4.
- 해석: decision-risk score metric은 held-out oracle single-action ranking proxy로 매우 강하다. Weight L2는 weak baseline.
- 이후에 바뀐 결정: scalar planner evaluation과 vector score-delta analysis로 이동.
- 현재 상태: 실험됨; metric go decision.
- 근거: `experiments/go_no_go/README.md`; `results/go_no_go/single_action_metrics_seed*.csv`; `results/go_no_go/metric_ranking_summary.csv`; commits `bf889ea`, `f7ea4b9`, `2f555c7`.

### 2026-06-24 12:30 — Scalar planner comparison including violation-aware metric

- 문제/질문: benchmark metrics를 additive planner objective로 쓰면 oracle set에서 어떤 plan이 나오는가.
- 당시 가설: decision-risk metrics가 scalar baselines보다 safer plans를 만들 것이다.
- 제안된 접근: metric별 rank normalization 후 additive solver로 budget별 plan을 만들고 development oracle에서 평가한다.
- 실제로 한 구현 또는 실험: `experiments/go_no_go/planner_eval.py`; `results/go_no_go_planner_v1`; `results/go_no_go_planner_v2_stress`.
- 결과: v1 budget 0.8 optimized plans는 모두 oracle flip 0, oracle accuracy 0.9995, p95 약 0.09279-0.09466로 비슷했다. v2 stress budgets 0.82-0.86에서도 many metrics가 close했고, violation-rate plan은 0.86에서 p95 0.288288/flip 0.001로 나빠졌다.
- 해석: scalar planning으로도 uniform INT4보다 안전하지만, metric 우열은 budget/constraint에서 미묘하다. Violation rate는 sparse/tie와 budget allocation 이슈가 있다.
- 이후에 바뀐 결정: scalar additive risk가 layer interactions/cancellations를 놓칠 수 있어 samplewise score-delta vector를 수집.
- 현재 상태: 실험됨; pilot.
- 근거: `results/go_no_go_planner_v1/planner_comparison_summary.csv`; `results/go_no_go_planner_v2_stress/planner_comparison_summary.csv`; `experiments/go_no_go/planner_eval.py`.

### 2026-06-24 13:11-13:35 — Samplewise score-delta vectors and vector proxy analysis

- 문제/질문: scalar additive risk는 여러 layer action의 score perturbation이 누적/상쇄되는 interaction problem을 설명하지 못하는가.
- 당시 가설: 각 candidate의 samplewise signed score delta vector를 합산하면 scalar risk sum보다 plan-level oracle p95 ranking을 더 잘 예측할 수 있다.
- 제안된 접근: score sets에서 layer/action별 `delta_score` vector를 저장하고, selected scalar plans에 대해 vector signed proxy와 scalar additive proxy를 비교한다.
- 실제로 한 구현 또는 실험: `collect_score_delta_vectors.py`; `analyze_vector_additive_proxy.py`.
- 결과: pooled within-budget primary ranking vs `oracle_decision_risk_p95`에서 `vector_signed_p95_risk` Kendall tau-b 0.735038, top1 0.857143, top3 0.904762. `scalar_additive_p95_risk` tau 0.367828, top1 0.428571, top3 0.714286.
- 해석: signed vector summation proxy는 scalar additive proxy보다 plan-level ranking에 더 가까운 signal을 준다. 단 code/comment가 명시하듯 interaction updates나 causality proof는 아니다.
- 이후에 바뀐 결정: vector proxy를 직접 objective로 쓰는 beam planner 구현.
- 현재 상태: 실험됨; proxy hypothesis.
- 근거: `results/go_no_go_vector_analysis_v1/vector_proxy_ranking_summary.csv`; `experiments/go_no_go/analyze_vector_additive_proxy.py`.

### 2026-06-24 13:44 — Vector-aware beam compression planner

- 문제/질문: signed samplewise vector proxy를 plan search objective로 직접 사용하면 scalar planners보다 better plan을 찾는가.
- 당시 가설: vector signed mean/p95 objective가 cancellation-aware risk를 낮출 수 있다.
- 제안된 접근: deterministic heuristic beam search over layer/action states, memory binning, objectives `vector_signed_mean_risk` and `vector_signed_p95_risk`.
- 실제로 한 구현 또는 실험: `experiments/go_no_go/vector_beam_planner.py`; pilot vector plans.
- 결과: pilot `vector_planner_summary.csv`는 budget 0.70-0.86에서 12 vector plans를 생성했다. 예: `vector_signed_mean` 0.86 actual saving 0.860248, objective_mean 0.055482, selected counts fp32 0/fp16 4/int8 10/int4 7.
- 해석: vector planner가 feasible plans를 만든다. 하지만 heuristic이며 global optimality guarantee는 없다.
- 이후에 바뀐 결정: vector vs scalar plans를 independent confirmation split과 locked test에서 fixed evaluation.
- 현재 상태: 구현됨/실험됨.
- 근거: `experiments/go_no_go/vector_beam_planner.py`; `results/go_no_go_vector_plans_v1/vector_planner_summary.csv`.

### 2026-06-24 13:54-14:00 — Confirmation evaluation and locked test evaluation

- 문제/질문: score/development oracle에 맞춘 plan이 unseen train-pool confirmation과 held-out test에서도 유지되는가.
- 당시 가설: vector plans should generalize better than scalar baselines at same budgets.
- 제안된 접근: canonical development oracle과 score sets를 제외한 train-pool confirmation split 생성; fixed plans evaluation; locked test evaluation.
- 실제로 한 구현 또는 실험: confirmation split은 13:35에 `make_confirmation_split.py`로 생성했고, 이후 `evaluate_plans_on_confirmation.py`와 `evaluate_plans_on_test.py`로 fixed plans를 평가했다.
- 결과: confirmation split size 2000, class counts 1000/1000, development oracle overlap 0, score sets overlap 0. Pilot test at budget 0.86: `vector_signed_p95` p95 0.121620, flip 0.0025, accuracy 0.9945; `vector_signed_mean` p95 0.123798, flip 0.002; weight p95 0.148395; decision_risk_p95 p95 0.161308; output_kl p95 0.175905.
- 해석: pilot test는 vector objectives가 scalar baselines보다 lower tail risk인 signal을 제공한다. 단 pilot result로 objective/hyperparameter를 retune하면 안 된다.
- 이후에 바뀐 결정: stronger evidence를 위해 frozen multi-seed protocol 구축.
- 현재 상태: 실험됨; pilot.
- 근거: `results/go_no_go_confirmation_v1/confirmation_split_indices.json`; `results/go_no_go_confirmation_eval_v1/confirmation_primary_summary.csv`; `results/go_no_go_test_eval_v1/test_primary_summary.csv`; `experiments/go_no_go/evaluate_plans_on_test.py`.

### 2026-06-24 14:19-16:22 — Frozen multi-seed training, reproducibility pipeline, aggregate analysis

- 문제/질문: single checkpoint/pilot result가 아니라 frozen protocol에서 current planner claim이 유지되는가.
- 당시 가설: `vector_signed_mean_risk` planner가 scalar baselines보다 lower locked-test decision-risk p95를 보일 것이다.
- 제안된 접근: seeds 101/202/303으로 final checkpoint를 만들고, canonical score/oracle split과 confirmation split을 고정한 pipeline을 실행한 뒤 descriptive aggregate를 산출한다.
- 실제로 한 구현 또는 실험: frozen training protocol, frozen repro pipeline, aggregate. Budgets 0.70/0.80/0.82/0.84/0.85/0.86, score seeds 0/1/2, beam width 512, memory quantum 1 KB.
- 결과: `vector_signed_mean_risk` vs each scalar baseline: 18/18 locked-test p95 wins. Relative p95 reduction mean: vs weight L2 0.529225, activation 0.534884, output KL 0.545131, abs delta 0.543547, decision risk mean 0.526099, decision risk p95 0.549915. `vector_signed_p95_risk` vs `vector_signed_mean_risk`: p95 win rate 7/18, relative_p95_reduction_mean -0.000568.
- 해석: 현재 가장 강한 claim은 vector signed mean planner가 frozen binary CIFAR-10 ResNet-18 fake-quant setting에서 scalar baselines보다 lower test p95 risk를 보였다는 descriptive result다. 통계적 유의성이나 generalization claim은 아직 안 된다.
- 이후에 바뀐 결정: next step은 새로운 대규모 feature보다 current claim의 minimal verification, calibration sensitivity, interaction audit.
- 현재 상태: 실험됨; current strongest evidence.
- 근거: `experiments/go_no_go/train_frozen_binary_seed.py`; `experiments/go_no_go/run_frozen_repro_pipeline.py`; `experiments/go_no_go/aggregate_frozen_repro_results.py`; `results/repro_v1/aggregate_v1/aggregate_summary.csv`; `results/repro_v1/aggregate_v1/per_seed_budget_results.csv`; commits `e16578d`, `08ff4f6`, `35dc2c4`, `21b76e5`, `e276d00`.

### 2026-06-24 to 2026-06-30 — Risk aggregation, evidence reporting, and narrow claim discipline

- 문제/질문: 평균 behavior만으로 충분한가, 어떤 evidence를 보고해야 과장되지 않는가.
- 당시 가설: mean and p95 risk must both be retained; oracle/held-out/proxy must be separated; signed-mean and signed-p95 should be reported with flip/accuracy/saving.
- 제안된 접근: \(J_{mean}\), \(J_{p95}\), signed mean/p95 notation, held-out oracle evaluation, Kendall tau-b/Spearman ranking evaluation, minimum reporting tuple을 정리.
- 실제로 한 구현 또는 실험: go/no-go ranking, confirmation/test evaluation, frozen aggregate에서 mean/p95/flip/accuracy/saving and rank correlations reported.
- 결과: formalization note explicitly says current proxy is not a proven upper bound, certified robustness guarantee, or globally optimal mixed-precision solution.
- 해석: 현재 claim은 narrow pilot evidence에 묶어야 한다.
- 이후에 바뀐 결정: Section 11의 can/cannot claim table이 필요.
- 현재 상태: Fixed reporting rule/conceptual discipline; implemented partially in result pipeline.
- 근거: `docs/formalization_notes.md`, sections 4, 5, 9.5, 10, 12, 13, 15; `results/repro_v1/aggregate_v1/aggregate_summary.csv`.

### [날짜 불명] — Structured pruning, latency, theory-bound expansion ideas

- 문제/질문: compression planning action space를 quantization 외 structured pruning/latency-aware planning까지 확장할 수 있는가.
- 당시 가설: decision preservation risk proxy가 compression action 일반으로 확장될 수 있다.
- 제안된 접근: [대화에서만 언급됨] 또는 user request에서 핵심 추적 대상. Repository 코드에는 structured pruning action, latency measurement, theoretical bound implementation이 확인되지 않는다.
- 실제로 한 구현 또는 실험: 확인된 구현 없음.
- 결과: [확인 불가].
- 해석: 현재 문서에서는 future direction으로만 취급해야 한다.
- 이후에 바뀐 결정: 우선 current quantization-only claim을 검증해야 한다.
- 현재 상태: 미구현/보류.
- 근거: repository keyword/file inspection; no `pruning` action implementation found in `*.py`; latency는 memory proxy와 구분됨.

## 3. Research Problem Definition Evolution

### Stage 0 — Mathematical decision-preservation framing

- 문제 정의: trained FP32 binary classifier의 decision sign을 compression 이후에도 보존하는 조건과 risk notation을 만든다.
- 입력: FP32 model \(f_\theta\), binary score \(s_\theta(x)\), compressed plan \(A\), compressed score \(s_{\tilde\theta(A)}(x)\).
- action space: abstract layer-level compression actions \(\mathcal A_\ell\); quantization/pruning/hybrid action까지 수식상 가능하지만 active pilot은 quantization.
- objective: FP32 decision preservation, i.e. \(\hat y_A(x)=\hat y_0(x)\), and memory/resource saving.
- constraints: risk proxy는 planner-side이고 oracle/full-forward evaluation과 구분해야 한다. Latency constraint는 valid latency model이 있을 때만 포함.
- 성공 기준: low oracle/held-out flip and risk at high memory saving; proxy ranking agrees with held-out oracle.
- 당시 한계: 수식은 analytical/conceptual이며 code implementation and experiments가 필요.
- 다음 단계로 바뀐 이유: binary scope and layer-level actions를 실제 ResNet/CIFAR-10 mixed-precision quantization pipeline으로 검증해야 했다.
- 근거: `docs/formalization_notes.md`, sections 0, 1, 2, 7, 10.

### Stage 1 — Accuracy and uniform compression baseline

- 문제 정의: trained binary ResNet-18을 uniform bit-width fake quantization했을 때 accuracy와 FP32 decision이 얼마나 유지되는지 측정한다.
- 입력: FP32 checkpoint, CIFAR-10 binary test loader, uniform bit-width `{32,16,8,4}`.
- action space: one global bit-width.
- objective: high accuracy / low flip / high memory saving 관찰.
- constraints: Conv2d/Linear weight-only fake quantization, no hardware latency.
- 성공 기준: high saving at low flip.
- 당시 한계: uniform INT4가 risk를 크게 만들며 layer별 선택 문제를 해결하지 못한다.
- 다음 단계로 바뀐 이유: uniform INT4 result가 flip 0.153으로 나빠 mixed precision이 필요해졌다.
- 근거: `results/uniform_baselines.csv`; `run_baselines.py`.

### Stage 2 — Layer-wise sensitivity and oracle-style protection

- 문제 정의: 어떤 layer/action이 decision risk를 크게 만드는지 single-layer full-forward oracle로 찾는다.
- 입력: FP32 checkpoint, validation/test subsets, layer list, action bits.
- action space: one layer at a time quantized, or top-k protected in otherwise INT4 model.
- objective: risky layers를 FP32로 보호해 flip/risk를 줄이는지 확인.
- constraints: oracle sweep은 full forward가 필요해 expensive하고 planner proxy가 아니다.
- 성공 기준: validation ranking top layers 보호가 test risk를 낮춘다.
- 당시 한계: validation oracle을 매번 쓰면 planning cost가 크고 interaction을 다루지 못한다.
- 다음 단계로 바뀐 이유: cheaper proxy metric 필요.
- 근거: `sweep_single_layer.py`; `run_oracle_guided_controls.py`; `results/oracle_guided_mixed_controls.csv`.

### Stage 3 — Proxy metric exploration

- 문제 정의: weight L2, activation reconstruction error, empirical margin risk 등 후보 metric이 oracle-like sensitivity를 대신할 수 있는가.
- 입력: FP32 model/layer activations/quantized local module outputs/score deltas.
- action space: layer별 precision assignment.
- objective: oracle full-forward 없이 risky layer/action ranking을 얻는다.
- constraints: proxy가 downstream full-model decision을 완벽히 보장하지 않는다.
- 성공 기준: proxy ranking controls가 oracle-guided controls와 가까운 risk/flip을 보인다.
- 당시 한계: weight L2가 명백히 misaligned; activation proxy는 promising but not final.
- 다음 단계로 바뀐 이유: proxy ranking을 실제 planner objective로 넣어야 한다.
- 근거: `activation_proxy.py`; `results/weight_l2_mixed_controls.csv`; `results/local_activation_p95_mixed_controls.csv`.

### Stage 4 — Additive mixed-precision planning

- 문제 정의: memory-saving target 아래에서 layer/action 선택을 자동화한다.
- 입력: layer/action risk table, per-action memory cost, target memory saving ratio.
- action space: each Conv2d/Linear layer chooses one of `{fp32,fp16,int8,int4}`.
- objective: target saving을 만족하면서 additive risk sum을 minimize.
- constraints: additive risk assumption; no interaction/cancellation; no real latency.
- 성공 기준: uniform INT4와 비슷한 saving에서 lower flip/p95 risk.
- 당시 한계: scalar metric/budget에 따라 brittle; interaction problem.
- 다음 단계로 바뀐 이유: held-out oracle ranking consistency를 먼저 검증해야 했다.
- 근거: `additive_planner.py`; `results/*dense_results.csv`.

### Stage 5 — Decision preservation and go/no-go metric validation

- 문제 정의: accuracy maximization 대신 FP32 model의 binary decision boundary를 보존하는 risk를 직접 측정한다.
- 입력: score/calibration set, held-out development oracle set, layer/action candidates.
- action space: single-action candidates for benchmark; scalar planner actions for planning.
- objective: score-set proxy ranking이 held-out oracle risk ranking과 일치하는지 검증.
- constraints: binary score margin에 특화; multi-class는 미구현.
- 성공 기준: high Spearman/Kendall tau-b and top-k recall.
- 당시 한계: single-action ranking은 multi-layer plan interaction을 보장하지 않는다.
- 다음 단계로 바뀐 이유: interaction/cancellation-aware vector proxy 필요.
- 근거: `experiments/go_no_go/metrics.py`; `results/go_no_go/metric_ranking_summary.csv`.

### Stage 6 — Vector proxy and vector beam planning

- 문제 정의: plan-level risk를 scalar additive sum이 아니라 samplewise signed score delta summation으로 approximate한다.
- 입력: score-set baseline margins, candidate signed score delta vectors, memory cost, budget.
- action space: per-layer `{fp32,fp16,int8,int4}`.
- objective: `vector_signed_mean_risk` or `vector_signed_p95_risk` minimize under memory-saving target.
- constraints: heuristic beam search, no exact optimality; still linear delta summation, no recalculated activations after combined actions.
- 성공 기준: confirmation/test post-action forward에서 scalar baselines보다 lower risk.
- 당시 한계: true interaction updates는 여전히 미구현; vector proxy can be wrong.
- 다음 단계로 바뀐 이유: stronger protocol needed.
- 근거: `experiments/go_no_go/collect_score_delta_vectors.py`; `experiments/go_no_go/vector_beam_planner.py`.

### Stage 7 — Frozen reproducibility and locked-test aggregate

- 문제 정의: current planner claim이 여러 trained checkpoints와 fixed protocol에서 유지되는지 검증한다.
- 입력: seeds 101/202/303 frozen checkpoints, canonical score/oracle split, confirmation split, locked binary test.
- action space: scalar baselines and vector objectives at fixed budgets.
- objective: descriptive locked-test p95 risk comparison without retuning on confirmation/test.
- constraints: only 3 seeds x 6 budgets; repeated measurements not independent statistical samples.
- 성공 기준: vector main objective lower p95 risk at comparable saving.
- 당시 한계: generalization/statistical significance/hardware still open.
- 다음 단계로 바뀐 이유: minimal verification and ablations before new features.
- 근거: `results/repro_v1/aggregate_v1/aggregate_summary.csv`; `experiments/go_no_go/aggregate_frozen_repro_results.py`.

구분 요약:

- Accuracy maximization: 초기 training/checkpoint quality 확인용. 현재 compression objective의 primary가 아니다.
- Memory saving: planner constraint/target. `actual_saving`은 parameter-storage proxy.
- Latency: 논의 대상이지만 구현/측정 없음.
- Prediction flip: FP32 prediction과 compressed prediction이 다른 sample fraction.
- Decision preservation: flip뿐 아니라 margin erosion/tail risk까지 보는 objective.
- Risk minimization: current formalization에서는 score margin erosion normalized by margin, 또는 vector summed delta risk.
- Oracle evaluation: full post-action forward on held-out oracle/confirmation/test.
- Proxy-based planning: score/calibration set에서 metric/vector를 계산해 plan을 선택.

## 4. Current Formalization and Notation

### Model, layers, actions

수식/정의:

```text
f_theta(x) -> z(x) in R^2
L = quantizable Conv2d/Linear layers
a_l in A = {fp32, fp16, int8, int4}
W_l = FP32 layer weight
Q_b(W_l) = fake-quantized/dequantized weight at bit-width b
Delta W_l(b) = Q_b(W_l) - W_l
```

- 의미: trained FP32 model의 quantizable layer별 precision action을 선택한다.
- 사용 목적: compression planner action space 정의.
- 계산에 필요한 정보: model module names, parameter shapes, bit-width mapping.
- forward-only 여부: action application/evaluation은 forward-only. Weight quantization 자체는 no-grad.
- oracle인지 proxy인지: action notation 자체는 둘 다에서 사용.
- 현재 구현 상태: 구현됨. `quantization.py`, `additive_planner.py`, `experiments/go_no_go/adapters.py`.
- 한계: only Conv2d/Linear weights; activation quantization, bias quantization, BatchNorm quantization, pruning 미구현.
- 근거: `quantization.py`, functions `list_quantizable_layers`, `fake_quantize_weight_per_output_channel`, `build_mixed_quantized_model`; `experiments/go_no_go/README.md`.

### Binary logit score and decision

수식/정의:

```text
z(x) = (z_0(x), z_1(x))
s(x) = z_1(x) - z_0(x)
pred(x) = 1 if s(x) > 0 else 0
margin(x) = |s_base(x)|
```

- 의미: binary classifier decision을 single signed score로 표현한다.
- 사용 목적: margin, delta score, decision risk 계산.
- 계산에 필요한 정보: FP32 logits and candidate logits.
- forward-only 여부: yes.
- oracle인지 proxy인지: score set에서는 proxy, held-out/confirmation/test에서는 oracle/evaluation.
- 현재 구현 상태: 구현됨. Old root `metrics.py`는 `binary_score`, `binary_prediction`; go/no-go `metrics.py`는 argmax 기반 base/candidate predictions와 score를 사용.
- 한계: binary-specific. Multi-class margin/risk는 미구현.
- 근거: `metrics.py`; `experiments/go_no_go/metrics.py`.

### Formalization-note exact decision condition and signed margin loss

수식/정의:

```text
s_0(x) = s_theta(x)
s_A(x) = s_tilde_theta(A)(x)
m_0(x) = |s_0(x)|
mu_A(x) = sign(s_0(x)) * s_A(x)
preservation iff mu_A(x) > 0, for s_0(x) != 0 and documented tie convention

R_sgn_A(x) =
  (m_0(x) - mu_A(x)) / m_0(x)
  = - sign(s_0(x)) * (s_A(x) - s_0(x)) / |s_0(x)|

F_A(x) = 1[R_sgn_A(x) >= 1]
```

- 의미: compressed score를 FP32 decision direction으로 정렬해 margin consumption을 signed quantity로 표현한다.
- 사용 목적: decision preservation의 canonical mathematical working definition. Negative value means margin increased in the FP32 decision direction.
- 계산에 필요한 정보: FP32 score, compressed full-plan score, score-zero tie convention.
- forward-only 여부: oracle/evaluation으로 계산할 때 full forward only; no gradient.
- oracle인지 proxy인지: \(s_A\)가 실제 compressed full forward에서 오면 oracle/evaluation; \(\widehat R_A\)로 approximate하면 proxy.
- 현재 구현 상태: 수학 노트에 [Fixed working risk definition]로 기록됨. 다만 현재 go/no-go code의 `decision_risk`는 아래 "Current code decision risk"처럼 nonnegative clipped erosion metric이다.
- 한계: \(m_0(x)=0\) 처리, p95 quantile interpolation, tie convention, code-level sign-equivalent naming은 notes에서도 확인 필요로 남겨졌다.
- 근거: `docs/formalization_notes.md`, sections 0.3, 3.1, 3.2, 12.2.

### Old margin-normalized risk

수식/정의:

```text
old_margin_normalized_risk(x) =
  |s_quantized(x) - s_fp32(x)| / (|s_fp32(x)| + epsilon)
```

- 의미: score perturbation magnitude를 FP32 margin으로 normalize한 old pilot metric.
- 사용 목적: early uniform baseline, single-layer sweep, empirical margin p95 planner.
- 계산에 필요한 정보: FP32 score, quantized score.
- forward-only 여부: yes.
- oracle인지 proxy인지: full test/validation forward로 계산하면 oracle-style empirical metric; planner table에 쓰면 proxy/risk estimate.
- 현재 구현 상태: 구현됨 in root `metrics.py`.
- 한계: perturbation direction을 무시한다. Decision boundary에서 멀어지는 harmless change도 risk로 센다. Current go/no-go decision risk와 다르다.
- 근거: `metrics.py`, function `compare_binary_models`; `results/uniform_baselines.csv`; `results/empirical_margin_dense_results.csv`.

### Current code decision risk

수식/정의:

```text
base_score = z_base,1 - z_base,0
candidate_score = z_candidate,1 - z_candidate,0
delta_score = candidate_score - base_score
direction = +1 if base_pred == 1 else -1
margin_erosion = max(0, -direction * delta_score)
decision_risk = margin_erosion / (|base_score| + epsilon)
decision_risk_violation = decision_risk >= 1
flip = argmax(z_candidate) != argmax(z_base)
```

- 의미: FP32 decision 방향의 margin을 깎는 perturbation만 risk로 센다.
- 사용 목적: decision preservation proxy, ranking analysis, scalar planner metric.
- 계산에 필요한 정보: base/candidate logits per sample.
- forward-only 여부: yes.
- oracle인지 proxy인지: score set에서 proxy risk; held-out oracle/confirmation/test에서 evaluation risk.
- 현재 구현 상태: 구현됨. `experiments/go_no_go/metrics.py`, `binary_decision_metrics`.
- 한계: binary-specific; margin erosion threshold는 flip과 관련 있지만 exact flip rate와 완전히 동일하지 않다. Flip target은 sparse/tie-dominated일 수 있다. 또한 `docs/formalization_notes.md`의 signed margin-loss \(R^{sgn}\)와 달리 negative helpful movement를 0으로 clip하므로 signed statistics와 이름을 섞으면 안 된다.
- 근거: `experiments/go_no_go/metrics.py`; `results/go_no_go/metric_ranking_summary.csv`; `docs/formalization_notes.md`, section 3.2.

### Mean risk, p95 risk, signed statistics

수식/정의:

```text
mean_risk = mean_x risk(x)
p95_risk = percentile_95_x risk(x)

For vector proxy:
delta_plan(x) = sum_l delta_{l,a_l}(x)
vector_signed_risk(x) = |delta_plan(x)| / (margin(x) + epsilon)
vector_signed_mean_risk = mean_x vector_signed_risk(x)
vector_signed_p95_risk = percentile_95_x vector_signed_risk(x)

vector_abs_sum_risk(x) =
  sum_l |delta_{l,a_l}(x)| / (margin(x) + epsilon)
scalar_additive_risk = sum_l precomputed scalar risk(l,a_l)
```

- 의미: mean은 average preservation cost, p95는 tail risk. `vector_signed_*`는 individual deltas의 sign을 유지해 먼저 합산하고 이후 magnitude/risk를 계산한다.
- 사용 목적: planner objective와 evaluation summary.
- 계산에 필요한 정보: samplewise risk vector or samplewise score-delta vectors.
- forward-only 여부: vector collection/evaluation은 forward-only.
- oracle인지 proxy인지: vector signed stats는 score set proxy. Confirmation/test p95는 post-action full forward evaluation.
- 현재 구현 상태: 구현됨. `analyze_vector_additive_proxy.py`, `vector_beam_planner.py`, evaluation scripts.
- 한계: "signed"는 signed deltas의 cancellation을 반영한다는 뜻이지 final combined model forward를 재계산한 true interaction이 아니다. `vector_signed_p95_risk`가 final main objective라는 증거는 없다.
- 근거: `experiments/go_no_go/analyze_vector_additive_proxy.py`; `experiments/go_no_go/vector_beam_planner.py`; `results/repro_v1/aggregate_v1/aggregate_summary.csv`.

### Oracle, held-out oracle, proxy, confirmation, test

정의:

```text
score/calibration set = 512 samples per score seed, train-pool class-balanced, disjoint from development oracle
development oracle = 2000 train-pool samples, class-balanced, held out from score sets
confirmation split = 2000 train-pool samples, class-balanced, disjoint from development oracle and score sets
locked test = CIFAR-10 binary test set, 2000 samples
```

- 의미: plan selection용 data와 evaluation용 data를 분리한다.
- 사용 목적: metric/proxy selection leakage 방지.
- 계산에 필요한 정보: split indices/metadata, source split.
- forward-only 여부: all evaluation forward-only.
- oracle인지 proxy인지: score set is proxy/calibration; development oracle is held-out oracle for benchmark/planner eval; confirmation and test are fixed evaluation.
- 현재 구현 상태: 구현됨. canonical split summary: source split train, oracle size 2000, class counts 1000/1000, score seeds 0/1/2 each size 512 with class counts 256/256 and oracle_overlap 0. Confirmation split: requested size 2000, class counts 1000/1000, development_oracle overlap 0, score_set overlap 0, score_set_union 1451, total_excluded_union 3451.
- 한계: confirmation is still train-pool, not test. Test is terminal.
- 근거: `results/go_no_go/split_indices.json`; `results/go_no_go_confirmation_v1/confirmation_split_indices.json`; `experiments/go_no_go/splits.py`; `experiments/go_no_go/make_confirmation_split.py`.

### Planner objective and memory cost

수식/정의:

```text
minimize proxy_risk(plan)
subject to saving(plan) >= target_saving
plan = {a_l for l in L}

parameter_weight_bytes(l,a) ~= numel(W_l) * bits(a) / 8
constant parameters = FP32 bytes
memory_saving_ratio = 1 - compressed_parameter_bytes / fp32_parameter_bytes
```

- 의미: target parameter memory saving 아래에서 risk를 최소화한다.
- 사용 목적: additive planner/scalar planner/vector beam planner.
- 계산에 필요한 정보: layer weights numel, bit assignment, risk table/vector proxy, target budget.
- forward-only 여부: planner search itself no model forward except risk table/vector precompute.
- oracle인지 proxy인지: planner objective is proxy. Evaluation is oracle/confirmation/test.
- 현재 구현 상태: implemented for fake weight quantization. `additive_planner.py` uses exact DP; `planner_eval.py` uses scalar rank-normalized metrics; `vector_beam_planner.py` uses heuristic beam.
- 한계: no latency, no hardware packing overhead, no activation memory, no structured pruning cost.
- 근거: `additive_planner.py`; `experiments/go_no_go/planner_eval.py`; `experiments/go_no_go/vector_beam_planner.py`; `experiments/go_no_go/README.md`.

Formalization note also records two equivalent constrained forms:

```text
P1 resource-first:
min_A M(A)
s.t. J_hat_risk(A; S_cal) <= tau
     T(A) <= T_max only when a valid latency model exists

P2 saving-first:
max_A Save_M(A)
s.t. J_hat_risk(A; S_cal) <= tau
```

상태: conceptual/current formulation. Code currently implements target-saving constrained risk minimization variants rather than a full latency-constrained problem. 근거: `docs/formalization_notes.md`, sections 0.4 and 7.

## 5. Theory and Analytical Ideas

### 증명 또는 정식 구현된 것

- Binary decision risk는 margin erosion 관점으로 구현되어 있다. `direction`을 기준으로 score delta가 FP32 decision margin을 줄이는 경우만 positive risk다. 상태: 구현됨. 근거: `experiments/go_no_go/metrics.py`.
- Layer/action memory cost는 parameter weight bit-width로 계산된다. 상태: 구현됨. 근거: `additive_planner.py`, `quantization.py`.
- Vector signed proxy는 samplewise signed score deltas를 합산해 cancellation을 반영한다. 상태: 구현됨. 근거: `collect_score_delta_vectors.py`, `analyze_vector_additive_proxy.py`, `vector_beam_planner.py`.
- Multiple-choice knapsack DP는 old/scalar additive planning에 쓰인다. 상태: 구현됨. 근거: `additive_planner.py`, `planner_eval.py`.

### 강한 empirical hypothesis로 남아 있는 것

- Tail risk, 특히 `decision_risk_p95`,는 single-action held-out oracle ranking을 잘 예측한다. 상태: 부분 검증. 근거: `results/go_no_go/metric_ranking_summary.csv`.
- Plan-level에서는 scalar additive p95보다 vector signed proxy가 oracle p95 ranking에 더 가까울 수 있다. 상태: 부분 검증. 근거: `results/go_no_go_vector_analysis_v1/vector_proxy_ranking_summary.csv`.
- Frozen 3-seed setting에서 `vector_signed_mean_risk`가 scalar baselines보다 lower locked-test p95를 낸다. 상태: 실험됨, descriptive. 근거: `results/repro_v1/aggregate_v1/aggregate_summary.csv`.

### Analytical derivations that motivate but do not certify the planner

- Last-layer logit bound: final binary score \(s(x)=w^\top h_{L-1}(x)+b\), final-layer perturbation \(\delta s(x)=\Delta w^\top h_{L-1}(x)\), and sufficient no-flip condition \(|\delta s(x)|<|s(x)|\) are documented. 상태: analytical derivation, not active guarantee. 근거: `docs/formalization_notes.md`, section 1.1.
- Earlier-layer exact recurrence: \(\delta u_\ell=W_\ell\delta h_{\ell-1}+\Delta W_\ell h_{\ell-1}+\Delta W_\ell\delta h_{\ell-1}\) is documented, including the cross term that simple first-order analyses often drop. 상태: analytical derivation. 근거: `docs/formalization_notes.md`, section 1.2.
- Local linearization and ReLU fixed-mask special case: \(\delta h_\ell\approx A_\ell(x)\delta h_{\ell-1}+e_\ell(a_\ell;x)\), with ReLU fixed-region \(A_\ell=D_\ell W_\ell\), are documented. 상태: analytical derivation. 근거: `docs/formalization_notes.md`, section 1.3.
- Activation error propagation: local activation proxy implements only local reconstruction, not the full propagation bound. 상태: partial implementation/limitation. 근거: `activation_proxy.py`; `docs/formalization_notes.md`, section 1.3.
- Sensitivity/Jacobian discussion: formalization notes explain why Jacobian propagation motivates the problem but is not the active planner because it needs Jacobian information, fixed ReLU masks, residual graph handling, and drops nonlinear/cross effects. 상태: analytical motivation; gradient/Jacobian planner 미구현. 근거: `docs/formalization_notes.md`, section 1.3.
- Gradient-free / forward-only planner intent: formalization notes explicitly distinguish analytical propagation from a desired forward-only proxy; code path also uses inference/full-local forward rather than gradients. 상태: design intent + implementation fact. 근거: `docs/formalization_notes.md`, sections 1.3 and 8.3; `run_single_action_benchmark.py`; `planner_eval.py`.
- Interaction problem: formalization notes define additive surrogate \(\sum_\ell R_{\ell,a_\ell}^{single}\) and interaction residual \(I_A(x)=R_A^{oracle}(x)-\sum_\ell R_{\ell,a_\ell}^{single}(x)\). Implemented vector proxy partially addresses cancellation with score-delta summation, but does not recompute true nonlinear activation interactions. 상태: analytical definition + partial heuristic implementation. 근거: `docs/formalization_notes.md`, section 8; `analyze_vector_additive_proxy.py`; `vector_beam_planner.py`.
- Sequential recalibration / interaction recalibration: formalization notes list sequential recalibration as a possible forward-only composition strategy but warn it trades planning quality against planning cost. Code implementation not found. 상태: 보류/미구현. 근거: `docs/formalization_notes.md`, section 8.4.

## 6. System and Implementation Status

### Model and data

- Model: torchvision ResNet-18, ImageNet pretrained weights, final `fc` replaced with 2-class linear head. 상태: 구현됨. 근거: `model.py`.
- Dataset: CIFAR-10 class IDs `[0,1]`, class names airplane/automobile. Labels remapped to 0/1. 상태: 구현됨. 근거: `config.py`, `data.py`.
- Transform: Resize 224, ImageNet normalization; train uses RandomHorizontalFlip, eval does not. 상태: 구현됨. 근거: `data.py`.
- Baseline checkpoint: `checkpoints/resnet18_binary_best.pt` for pilot, `checkpoints/repro_v1/resnet18_binary_seed{101,202,303}_final.pt` for frozen repro. 상태: artifact exists; binary not decoded. 근거: logs/manifests.

### Binary classification 구성 방식

- CIFAR-10 original labels 0 and 1 only.
- `BinaryCIFAR10` maps class 0 to label 0 and class 1 to label 1.
- Test set has 2000 binary samples. Training/validation split uses fixed seed.
- 근거: `data.py`; `config.py`; `docs/experiment_log.md`.

### Quantization method

- Weight-only fake quantization.
- FP16 action: cast weight to `float16`, then back to original dtype.
- INT8/INT4 action: symmetric per-output-channel quantization with max-abs scale, round/clip/dequantize.
- Model remains floating-point after fake quantization.
- Bias/BatchNorm/other parameters remain FP32 for memory estimate.
- 근거: `quantization.py`; `experiments/go_no_go/README.md`.

### Action granularity

- Granularity: individual Conv2d or Linear module/layer.
- Candidates in go/no-go: 21 quantizable layers x 3 non-FP32 actions = 63 candidates.
- FP32 is included as planner action but not as risky single-action candidate.
- 근거: `quantization.py`; `run_single_action_benchmark.py`; `collect_score_delta_vectors.py`.

### Calibration and evaluation procedure

- Score/calibration set: train-pool class-balanced, 512 samples per seed, seeds 0/1/2, disjoint from development oracle.
- Development oracle: train-pool class-balanced 2000 samples, seed 2026.
- Confirmation split: train-pool class-balanced 2000 samples, seed 3030, excludes development oracle and score sets.
- Locked test: binary CIFAR-10 test set 2000 samples.
- 근거: `results/go_no_go/split_indices.json`; `results/go_no_go_confirmation_v1/confirmation_split_indices.json`.

### Planner 종류

- Old additive planner: exact multiple-choice knapsack using CSV risk table. 근거: `additive_planner.py`.
- Go/no-go scalar planner: rank-normalized score metrics averaged across score seeds, additive solver, evaluated on oracle. 근거: `planner_eval.py`.
- Vector beam planner: samplewise score-delta vectors, deterministic beam search, objectives `vector_signed_mean_risk`, `vector_signed_p95_risk`. 근거: `vector_beam_planner.py`.
- Implemented baselines: `weight_rel_l2`, `activation_rel_mse`, `output_kl_mean`, `abs_delta_score_mean`, `decision_risk_mean`, `decision_risk_p95`; old baselines include `local_activation_p95`, `empirical_margin_p95`.

### Memory and latency

- Memory saving: theoretical parameter storage ratio computed from per-layer weights and assigned bits.
- Actual saving in result CSV means actual achieved ratio under memory quantization/plan allocation, not measured hardware memory.
- Latency: no hardware timing or latency proxy implementation found.
- 근거: `quantization.py`, `additive_planner.py`, `experiments/go_no_go/README.md`.

### Reproducibility commands

Initial pipeline examples:

```powershell
python train.py --epochs 10
python run_baselines.py --max-samples 2000
python sweep_single_layer.py --split validation --max-samples 1000 --output results/validation_single_layer_sweep.csv
python run_oracle_guided_controls.py --sweep-csv results/validation_single_layer_sweep.csv
python run_activation_proxy.py --split validation --max-samples 1000 --bits 4
python run_additive_planner.py --risk-csv results/validation_local_activation_proxy_all_bits.csv
```

Go/no-go and frozen pipeline examples:

```powershell
python -m experiments.go_no_go.run_single_action_benchmark --score-size 512 --oracle-size 2000 --score-seeds 0 1 2 --oracle-seed 2026
python -m experiments.go_no_go.analyze_rankings --benchmark-dir results/go_no_go
python -m experiments.go_no_go.planner_eval --benchmark-dir results/go_no_go --memory-saving-ratios 0.70 0.80 0.82 0.84 0.85 0.86
python -m experiments.go_no_go.collect_score_delta_vectors --benchmark-dir results/go_no_go --score-seeds 0 1 2
python -m experiments.go_no_go.vector_beam_planner --vector-dir results/go_no_go_vectors_v1 --benchmark-dir results/go_no_go --beam-width 512 --max-states-per-memory-bin 8 --memory-quantum-kb 1
python -m experiments.go_no_go.evaluate_plans_on_confirmation --confirmation-split results/go_no_go_confirmation_v1/confirmation_split_indices.npz
python -m experiments.go_no_go.evaluate_plans_on_test --vector-plan-dir results/go_no_go_vector_plans_v1 --scalar-plan-dirs results/go_no_go_planner_v2_stress
```

Frozen seed pipeline command shape, from `results/repro_v1/seed_101/pipeline_commands.txt`:

```powershell
python -m experiments.go_no_go.run_frozen_repro_pipeline --checkpoint checkpoints\repro_v1\resnet18_binary_seed101_final.pt --run-name seed_101 --canonical-split results\go_no_go\split_indices.npz --canonical-metadata results\go_no_go\split_indices.json --confirmation-split results\go_no_go_confirmation_v1\confirmation_split_indices.npz --confirmation-metadata results\go_no_go_confirmation_v1\confirmation_split_indices.json --output-root results\repro_v1\seed_101 --batch-size 256 --score-seeds 0 1 2 --memory-saving-ratios 0.70 0.80 0.82 0.84 0.85 0.86 --beam-width 512 --max-states-per-memory-bin 8 --memory-quantum-kb 1 --stages scalar_plans vectors vector_plans confirmation_eval --resume
python -m experiments.go_no_go.run_frozen_repro_pipeline --checkpoint checkpoints\repro_v1\resnet18_binary_seed101_final.pt --run-name seed_101 --canonical-split results\go_no_go\split_indices.npz --canonical-metadata results\go_no_go\split_indices.json --confirmation-split results\go_no_go_confirmation_v1\confirmation_split_indices.npz --confirmation-metadata results\go_no_go_confirmation_v1\confirmation_split_indices.json --output-root results\repro_v1\seed_101 --batch-size 256 --score-seeds 0 1 2 --memory-saving-ratios 0.70 0.80 0.82 0.84 0.85 0.86 --beam-width 512 --max-states-per-memory-bin 8 --memory-quantum-kb 1 --stages test_eval --resume
```

### 코드와 문서 간 불일치

- User-level research topic includes structured pruning and latency; code implements only quantization and parameter memory proxy.
- 초기 `docs/experiment_log.md`의 "oracle"은 go/no-go protocol의 "development oracle"과 다른 의미로 쓰인다.
- `signed-p95`는 plausible objective였지만 frozen aggregate에서 main objective로 지지되지 않는다.
- Some smoke files exist but should not override full/pipeline results.

## 7. Experiment Inventory

| Experiment ID / File | Purpose | Setup | Metrics | Main result | Interpretation | Status | Evidence |
| -------------------- | ------- | ----- | ------- | ----------- | -------------- | ------ | -------- |
| `train_binary_resnet18` | FP32 baseline checkpoint | CIFAR-10 0/1, ResNet-18, epochs 10 | val/test accuracy/loss | test acc 0.996 | reliable teacher for decision preservation | 실험됨 | `docs/experiment_log.md`; `train.py` |
| `results/uniform_baselines.csv` | Uniform quantization baseline | full test 2000, bits 32/16/8/4 | saving, accuracy, flip, mean/p95 old risk | int4 saving 0.874248, flip 0.153 | uniform int4 risky | 실험됨 | CSV |
| `results/single_layer_sweep.csv` | Test single-layer sensitivity | one layer quantized, test | old margin risk, flip | conv1 int4 highest p95/flip | layer sensitivity exists | 실험됨; leakage-prone for selection | CSV |
| `results/validation_single_layer_sweep.csv` | Validation sensitivity | one layer quantized, validation 1000 | old risk, flip | conv1 int4 p95 1.197196, flip 0.082 | validation ranking usable for controls | 실험됨 | CSV |
| `results/oracle_guided_mixed_controls.csv` | Protect oracle top-k layers | validation top-k FP32, rest int4, test | saving, flip, p95 | top1 conv1 flip 0.0125 vs int4 0.153 | selective protection works | 실험됨 | CSV |
| `results/weight_l2_mixed_controls.csv` | Weight L2 baseline | weight L2 rank top/bottom controls | saving, flip, p95 | top1 bad, bottom2 good due conv1/fc | L2 misaligned | 실험됨 | CSV |
| `results/validation_local_activation_proxy_all_bits.csv` | Activation proxy ranking | local module reconstruction on validation | mean/p95 activation rel L2 | conv1 int4 top p95 0.238753 | activation proxy promising | 실험됨 | CSV; `activation_proxy.py` |
| `results/local_activation_p95_mixed_controls.csv` | Activation proxy controls | activation top-k FP32, rest int4 | saving, flip, p95 | top4 flip 0.0065, p95 0.427720 | better than weight L2 | 실험됨 | CSV |
| `results/*additive_planner_results.csv` | Old additive planner | risk CSV + memory target | actual saving, full-forward metrics | activation/empirical low risk; L2 brittle | additive framework works, metric matters | 실험됨 | CSV |
| `results/*dense_results.csv` | Dense budget stress | dense requested saving near high compression | flip, p95 | weight L2 spike at 0.8725 | scalar metric brittle | 실험됨 | CSV |
| `results/go_no_go/single_action_metrics_seed*.csv` | Single-action go/no-go benchmark | 63 candidates, score seeds 0/1/2, oracle 2000 | weight/activation/KL/decision risk/oracle risk | decision_risk_p95 ranks oracle p95 strongly | metric go | 실험됨 | CSV |
| `results/go_no_go/metric_ranking_summary.csv` | Ranking consistency | score metrics vs oracle targets | Spearman, Kendall tau-b, top-k recall | decision_risk_p95 tau 0.984298 vs oracle p95 | strongest single-action proxy | 실험됨 | CSV |
| `results/go_no_go_planner_v1/planner_comparison_summary.csv` | Scalar planner v1 | budgets .5-.8 | oracle risk/flip | budget .8 all optimized flip 0 and p95 ~0.093 | scalar planners safe at moderate budget | 실험됨 | CSV |
| `results/go_no_go_planner_v2_stress/planner_comparison_summary.csv` | Scalar planner stress | budgets .82-.86 | oracle risk/flip | violation-rate weak at .86 | high saving differentiates metrics | 실험됨 | CSV |
| `results/go_no_go_vectors_v1/score_delta_vectors_seed*.json/.npz` | Collect vector proxy inputs | samplewise signed delta vectors | delta validation, schema | 63 candidates x 512 samples per seed | enables vector proxy | 구현됨 | `collect_score_delta_vectors.py`; sidecar JSON |
| `results/go_no_go_vector_analysis_v1/vector_proxy_ranking_summary.csv` | Vector proxy analysis | existing scalar plans, vector vs scalar proxy | tau/top-k vs oracle p95 | vector_signed_p95 tau 0.735038 vs scalar p95 0.367828 | vector proxy better plan ranking | 실험됨 | CSV |
| `results/go_no_go_vector_plans_v1/vector_planner_summary.csv` | Vector beam plans | objectives mean/p95, budgets .70-.86 | objective, saving, action counts | feasible vector plans generated | planner implemented | 실험됨 | CSV |
| `results/go_no_go_confirmation_v1/confirmation_split_indices.json` | Confirmation split | train-pool, excludes score/oracle | class counts/overlaps | size 2000, overlap 0 | independent confirmation from train pool | 구현됨 | JSON |
| `results/go_no_go_confirmation_eval_v1/confirmation_primary_summary.csv` | Pilot confirmation eval | fixed vector/scalar plans | p95/flip/accuracy | vector lower p95 at budgets | support only | 실험됨 | CSV |
| `results/go_no_go_test_eval_v1/test_primary_summary.csv` | Pilot locked test eval | fixed vector/scalar plans on test | p95/flip/accuracy | vector lower p95 than examples | support only; no retune | 실험됨 | CSV |
| `results/repro_v1/aggregate_v1/training_summary.csv` | Frozen training summary | seeds 101/202/303 | train/val acc/loss | val acc 0.996/0.998/0.996 | checkpoints strong | 실험됨 | CSV |
| `results/repro_v1/seed_*/pipeline_manifest.json` | Pipeline provenance | fixed score seeds/budgets/beam settings | stage status/commands | all stages completed | reproducibility evidence | 구현됨 | JSON |
| `results/repro_v1/aggregate_v1/aggregate_summary.csv` | Final aggregate | 3 seeds x 6 budgets | p95 win counts/reductions | vector mean 18/18 vs scalar baselines | current strongest claim | 실험됨; descriptive | CSV |
| `results/repro_v1/aggregate_v1/per_seed_budget_results.csv` | Detailed aggregate cells | all selections/seed/budget | test p95/flip/accuracy/saving | vector mean mean p95 0.074972 | detailed numbers | 실험됨 | CSV |

Confounders/한계:

- Pilot single checkpoint와 frozen 3-seed results는 구분해야 한다.
- `results/go_no_go_smoke*`는 sanity/smoke artifacts로 full benchmark보다 약하다.
- Frozen aggregate는 18 cells를 반복 측정으로 다루며 independent statistical inference를 하지 않는다.
- All experiments are binary CIFAR-10 ResNet-18 fake weight quantization only.

## 8. Verified Results and Exact Numbers

### Uniform baseline

근거: `results/uniform_baselines.csv`.

| Bits | Memory saving ratio | Accuracy | Flip rate | Mean old margin risk | p95 old margin risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32/32 | 0 | 0.996 | 0 | 0 | 0 |
| fp16/16 | 0.4995704778637299 | 0.996 | 0 | 0.0013704068260267377 | 0.003032 |
| int8/8 | 0.7493557167955949 | 0.9955 | 0.0005 | 0.03685024008154869 | 0.083575 |
| int4/4 | 0.8742483362615274 | 0.846 | 0.153 | 0.7086520195007324 | 1.3554645299911498 |

의미: `memory_saving_ratio`는 parameter storage proxy, `flip_rate`는 FP32 prediction과 compressed prediction mismatch fraction.

### Validation single-layer sensitivity and activation proxy

근거: `results/validation_single_layer_sweep.csv`; `results/validation_local_activation_proxy_all_bits.csv`.

| Metric/source | Layer/action | Key values |
| --- | --- | --- |
| Validation full-forward old p95 risk | `conv1 int4` | p95 1.1971958875656128, flip 0.082 |
| Validation full-forward old p95 risk | `layer1.0.conv1 int4` | relative L2 0.275746, flip 0.002, p95 0.331825 |
| Validation full-forward old p95 risk | `layer1.1.conv1 int4` | relative L2 0.239079, flip 0, p95 0.263625 |
| Local activation p95 | `conv1 int4` | mean_rel_act 0.207279, p95 0.23875348269939423, max 0.248666 |
| Local activation p95 | `layer2.0.conv1 int4` | p95 0.178932 |
| Local activation p95 | `layer2.0.downsample.0 int4` | p95 0.164640 |
| Local activation p95 | `layer1.1.conv1 int4` | p95 0.163185 |

### Mixed controls

근거: `results/oracle_guided_mixed_controls.csv`; `results/weight_l2_mixed_controls.csv`; `results/local_activation_p95_mixed_controls.csv`.

| Plan | Saving | Accuracy | Flip | p95 risk | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| uniform_int4 | 0.874248 | 0.846 | 0.153 | 1.355465 | high saving, high damage |
| oracle_top1_fp32 (`conv1`) | 0.873511859230539 | 0.9845 | 0.0125 | 0.628788 | protecting conv1 strongly helps |
| oracle_top2_fp32 | 0.870626 | 0.99 | 0.007 | 0.600372 | more protection helps |
| oracle_top4_fp32 | 0.867099 | 0.9915 | 0.0055 | 0.353377 | best listed oracle control |
| weight_l2_top1 | 0.871362548711532 | 0.865 | 0.134 | 1.2758997678756714 | weight L2 top protection poor |
| weight_l2_bottom2 (`conv1,fc`) | 0.8734316984652613 | 0.985 | 0.012 | 0.6310280561447144 | bottom ranking accidentally protects critical layers |
| local_activation_top1 | 0.873511859230539 | 0.9845 | 0.0125 | 0.628788411617279 | activation top1 matches conv1 |
| local_activation_top2 | 0.8677402841305483 | 0.988 | 0.010 | 0.561469554901123 | activation ranking useful |
| local_activation_top4 | 0.8642132104583317 | 0.9905 | 0.0065 | 0.42771995067596436 | low flip with slightly less saving |

### Dense planner comparison

근거: `results/weight_l2_dense_results.csv`; `results/local_activation_dense_results.csv`; `results/empirical_margin_dense_results.csv`.

| Planner metric | Requested saving | Actual saving | Flip | p95 risk | Meaning |
| --- | ---: | ---: | ---: | ---: | --- |
| weight L2 | 0.8725 | 0.872519 | 0.1435 | 1.316518 | severe bad plan spike |
| weight L2 | 0.873 | 0.873032 | 0.0055 | 0.370520 | neighboring budget much safer |
| weight L2 | 0.872 | 0.872025 | 0.0045 | 0.319434 | neighboring budget much safer |
| local activation | 0.8725 | 0.872570 | 0.0055 | 0.348429 | stable low risk |
| empirical margin | 0.8725 | 0.872524 | 0.0040 | 0.324819 | stable low risk |
| empirical margin | 0.870 | 0.870096 | 0.003 | 0.239269 | lower target safer |

해석: weight L2 planner는 특정 budget에서 brittle. 이 단일 spike만으로 모든 weight L2 plan이 항상 나쁘다고 일반화하면 안 되지만, proxy로서 위험하다.

### Go/no-go single-action ranking

근거: `results/go_no_go/metric_ranking_summary.csv`.

Against `oracle_decision_risk_p95`:

| Score metric | Spearman mean | Kendall tau-b mean | Top5 recall | Top10 recall |
| --- | ---: | ---: | ---: | ---: |
| decision_risk_p95 | 0.998960 | 0.984298 | 1.0 | 1.0 |
| decision_risk_mean | 0.992576 | 0.959037 | 1.0 | 0.9 |
| abs_delta_score_mean | 0.976542 | 0.936849 | 1.0 | 1.0 |
| output_kl_mean | 0.9607387290786837 | 0.8709504599044093 | 0.9333333333333332 | 0.9 |
| activation_rel_mse | 0.9042018689196109 | 0.7545656255333673 | 0.8000000000000002 | 0.6999999999999998 |
| weight_rel_l2 | 0.8702956989247314 | 0.6600102406554018 | 0.20000000000000004 | 0.4000000000000001 |
| decision_risk_violation_rate | 0.3015198655960869 | 0.2493985803791753 | 0.3333333333333333 | 0.2333333333333333 |

Against secondary `oracle_flip_rate`:

| Score metric | Kendall tau-b mean | Top10 recall | Notes |
| --- | ---: | ---: | --- |
| decision_risk_violation_rate | 0.638876 | 0.866667 | best among noted flip-oriented metrics |
| output_kl_mean | 0.31897208495573803 | 0.3333333333333333 | weak due sparse/ties |
| decision_risk_mean/p95 | 0.30610781912228796 | 0.3 | weak against sparse flip target |
| weight_rel_l2 | 0.21912636232251217 | 0.20000000000000004 | weak |

Single-action top example, seed0, 근거: `results/go_no_go/single_action_metrics_seed0.csv`:

- `conv1 int4`: `weight_rel_l2=0.149985`, `activation_rel_mse=0.044066`, `output_kl_mean=0.192758`, `abs_delta_score_mean=5.391079`, `decision_risk_mean=0.443142`, `decision_risk_p95=1.112019`, `oracle_flip_rate=0.0835`, `oracle_accuracy=0.9160`, `oracle_decision_risk_mean=0.423356`, `oracle_decision_risk_p95=1.117103`, `oracle_decision_risk_violation_rate=0.0835`.
- `layer1.0.conv1 int4`: `oracle_decision_risk_p95=0.308553`, `oracle_flip_rate=0.001`, `oracle_accuracy=0.9985`.

### Scalar planner pilot

근거: `results/go_no_go_planner_v1/planner_comparison_summary.csv`; `results/go_no_go_planner_v2_stress/planner_comparison_summary.csv`.

- v1 budget 0.8: optimized plans all `oracle_flip_rate=0`, `oracle_accuracy=0.9995`; p95 range roughly 0.092791 to 0.094664.
- v1 anchors: uniform_fp16 p95 0.001910, flip 0, actual saving 0.499570; uniform_int8 p95 0.057044, flip 0, saving 0.749356; uniform_int4 p95 1.259235, flip 0.1405, accuracy 0.859, saving 0.874248.
- v2 budget 0.86: weight/abs_delta p95 0.136571; decision_risk_mean/p95 p95 0.147368; output_kl 0.155071; activation 0.155443; violation_rate 0.288288 with flip 0.001 and accuracy 0.9985.

### Vector proxy and planner pilot

근거: `results/go_no_go_vector_analysis_v1/vector_proxy_ranking_summary.csv`; `results/go_no_go_vector_plans_v1/vector_planner_summary.csv`.

- Pooled within-budget primary vs `oracle_decision_risk_p95`: `vector_signed_p95_risk` Spearman 0.810281, Kendall tau-b 0.735038, top1 0.857143, top3 0.904762. `scalar_additive_p95_risk` Spearman 0.465293, tau 0.367828, top1 0.428571, top3 0.714286.
- Per-budget primary `vector_signed_p95` tau: 1.0 at 0.50, NaN at 0.60 due constant budget, 0.714286 at 0.70, 0.333333 at 0.80, 1.0 at 0.82, 0.857143 at 0.84, 1.0 at 0.85, 0.230769 at 0.86.
- Pilot vector plan examples: `vector_signed_mean` budget 0.70 actual 0.700353, objective_mean 0.002101, counts fp32 1/fp16 13/int8 7/int4 0. `vector_signed_mean` budget 0.86 actual 0.860248, objective_mean 0.055482, counts fp32 0/fp16 4/int8 10/int4 7. `vector_signed_p95` budget 0.86 actual 0.860019, objective_mean 0.140974, counts fp32 1/fp16 5/int8 7/int4 8.

### Pilot confirmation/test fixed evaluation

근거: `results/go_no_go_confirmation_eval_v1/confirmation_primary_summary.csv`; `results/go_no_go_test_eval_v1/test_primary_summary.csv`.

- Confirmation 0.70: vector mean p95 0.003846 vs decision_risk_p95 0.015417, output_kl 0.060916, weight 0.077966.
- Confirmation 0.86: vector mean p95 0.119308, vector p95 0.116696 vs weight 0.137429, decision_risk_p95 0.147281, output_kl 0.154256.
- Test 0.70: vector mean p95 0.004179, flip 0, accuracy 0.996; vector p95 0.004258; decision_risk_p95 0.017044, flip 0.0005; output_kl 0.065336, flip 0.0005; weight 0.080118, flip 0.0015.
- Test 0.86: vector p95 p95 0.121620, flip 0.0025, accuracy 0.9945; vector mean p95 0.123798, flip 0.002, accuracy 0.995; weight p95 0.148395; decision_risk_p95 p95 0.161308; output_kl p95 0.175905.

### Frozen training and aggregate

근거: `results/repro_v1/aggregate_v1/training_summary.csv`; `results/repro_v1/aggregate_v1/aggregate_summary.csv`; `results/repro_v1/aggregate_v1/per_seed_budget_results.csv`.

Training:

| Seed | Final train acc | Final train loss | Final val acc | Final val loss |
| ---: | ---: | ---: | ---: | ---: |
| 101 | 0.999889 | 0.000750 | 0.996 | 0.007157 |
| 202 | 1.0 | 0.000469 | 0.998 | 0.005713 |
| 303 | 0.999889 | 0.000646 | 0.996 | 0.011570 |

Vector main `vector_signed_mean_risk` vs scalar baselines, locked-test descriptive:

| Baseline | n cells | p95 win count | p95 win rate | Relative p95 reduction mean | Mean p95 delta | Mean saving delta | Strict Pareto | Near Pareto |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight_rel_l2 | 18 | 18 | 1.0 | 0.529225 | -0.063089 | -0.000085 | 7 | 17 |
| activation_rel_mse | 18 | 18 | 1.0 | 0.534884 | -0.064305 | -0.000358 | 7 | 14 |
| output_kl_mean | 18 | 18 | 1.0 | 0.545131 | -0.065360 | -0.000791 | 7 | 14 |
| abs_delta_score_mean | 18 | 18 | 1.0 | 0.543547 | -0.065116 | -0.000710 | 5 | 14 |
| decision_risk_mean | 18 | 18 | 1.0 | 0.526099 | -0.063409 | -0.000697 | 5 | 14 |
| decision_risk_p95 | 18 | 18 | 1.0 | 0.549915 | -0.069047 | -0.000959 | 5 | 12 |

Vector signed-p95 ablation vs vector signed-mean:

- p95 win count 7/18, win rate 0.388889, relative_p95_reduction_mean -0.000568, mean_p95_delta +0.000048, near_pareto 7.
- 해석: current main objective should be `vector_signed_mean_risk`, not `vector_signed_p95_risk`.

Per-selection means from `per_seed_budget_results.csv`:

| Selection | Mean test p95 | Mean test flip | Mean test accuracy | Mean actual saving |
| --- | ---: | ---: | ---: | ---: |
| vector_signed_mean_risk | 0.074972 | 0.000917 | 0.996194 | 0.811983 |
| vector_signed_p95_risk | 0.075020772728344592 | 0.00077777781148645551 | 0.99611111813121367 | 0.81204087857073526 |
| weight_rel_l2 | 0.13806151184770793 | 0.00094444445373177775 | 0.99600000845061409 | 0.8120676783503965 |
| decision_risk_mean | 0.13838109084301525 | 0.0010555555862891721 | 0.99588889877001441 | 0.81268001752960073 |
| activation_rel_mse | 0.1392771852099233 | 0.0008611111326091111 | 0.99586111307144165 | 0.81234156096509524 |
| abs_delta_score_mean | 0.14008803024060196 | 0.00097222224253020525 | 0.99591667122311067 | 0.81269274146059711 |
| output_kl_mean | 0.14033203509946662 | 0.00094444446343305551 | 0.99599999851650667 | 0.8127741746189745 |
| decision_risk_p95 | 0.14401898222664988 | 0.0010277778007245055 | 0.99586111307144165 | 0.8129421305081278 |

Vector main per budget across seeds:

| Budget | Mean test p95 | Mean test flip | Mean test accuracy | Mean actual saving |
| ---: | ---: | ---: | ---: | ---: |
| 0.70 | 0.004394 | 0 | 0.996333 | 0.700300 |
| 0.80 | 0.026375449573000229 | 0.00033333334916579995 | 0.99633334080378211 | 0.80008173535173854 |
| 0.82 | 0.0613666785260041 | 0.00066666669833159991 | 0.99599999189376831 | 0.82068877779704252 |
| 0.84 | 0.0896607488393783 | 0.0015000000130384666 | 0.99616668621699012 | 0.84034247971243747 |
| 0.85 | 0.1181462854146957 | 0.0013333333966632332 | 0.99633334080378211 | 0.85018459938733082 |
| 0.86 | 0.14989098409811655 | 0.0016666666682188334 | 0.99600001176198327 | 0.8603015559717474 |

## 9. Baselines, Metrics, and Comparison Logic

### Weight L2 norm

- 무엇을 측정하는가: `||Q_b(W)-W||_2 / ||W||_2`류의 weight perturbation magnitude.
- 왜 baseline 또는 후보 metric인가: 가장 cheap하고 model forward 없이 계산 가능.
- planner에 직접 적용 가능한가: yes, scalar risk/cost table로 가능.
- oracle이 필요한가: 계산에는 oracle 불필요.
- 실험에서 실제로 사용되었는가: yes, controls/scalar planner/frozen baselines.
- decision preservation과 어떤 관계인가: 간접적. Weight change가 decision boundary effect로 이어진다는 보장은 없다.
- 관찰된 장점: cheap, deterministic.
- 관찰된 한계: mixed controls and go/no-go ranking에서 weak/misaligned. 근거: `results/weight_l2_mixed_controls.csv`; `results/go_no_go/metric_ranking_summary.csv`.

### Activation L2 / activation relative MSE

- 무엇을 측정하는가: target layer local output reconstruction error under fake quantization.
- 왜 baseline 또는 후보 metric인가: weight change보다 model computation/activation scale을 반영한다.
- planner에 직접 적용 가능한가: yes, scalar table로 가능.
- oracle이 필요한가: local forward/calibration activations 필요, full oracle은 불필요.
- 실험에서 실제로 사용되었는가: yes, early activation proxy and go/no-go `activation_rel_mse`.
- decision preservation과 어떤 관계인가: downstream decision effect의 indirect proxy.
- 관찰된 장점: early pilot에서 conv1 위험을 잘 포착.
- 관찰된 한계: current frozen aggregate에서는 vector main보다 lower p95를 보이지 못함.
- 근거: `activation_proxy.py`; `results/validation_local_activation_proxy_all_bits.csv`; `results/repro_v1/aggregate_v1/aggregate_summary.csv`.

### Empirical metric / old empirical margin p95

- 무엇을 측정하는가: single-layer full-forward old margin-normalized risk.
- 왜 baseline 또는 후보 metric인가: actual model output effect를 직접 측정하므로 oracle-like upper-quality proxy.
- planner에 직접 적용 가능한가: yes, but expensive because every candidate needs full forward.
- oracle이 필요한가: validation/test full forward required.
- 실험에서 실제로 사용되었는가: yes, early additive planner.
- decision preservation과 어떤 관계인가: margin-normalized perturbation magnitude; direction ignored.
- 관찰된 장점: dense planner에서 low p95/flip.
- 관찰된 한계: old risk definition, possible oracle leakage/cost.
- 근거: `results/empirical_margin_dense_results.csv`; `metrics.py`.

### KL divergence

- 무엇을 측정하는가: FP32 and candidate output distributions divergence.
- 왜 baseline 또는 후보 metric인가: common distillation/model-output preservation metric.
- planner에 직접 적용 가능한가: yes, scalar metric.
- oracle이 필요한가: score/calibration logits; full held-out oracle only for evaluation.
- 실험에서 실제로 사용되었는가: yes, go/no-go ranking/scalar planner/frozen baseline.
- decision preservation과 어떤 관계인가: output distribution preservation이 decision preservation과 관련될 수 있지만 binary margin direction을 직접 보지는 않는다.
- 관찰된 장점: ranking better than weight L2 in some settings.
- 관찰된 한계: frozen aggregate에서 vector main보다 p95가 높음.
- 근거: `experiments/go_no_go/metrics.py`; `results/repro_v1/aggregate_v1/aggregate_summary.csv`.

### Decision risk metric

- 무엇을 측정하는가: FP32 decision margin을 erosion시키는 score perturbation.
- 왜 baseline 또는 후보 metric인가: decision preservation objective와 직접 연결된다.
- planner에 직접 적용 가능한가: yes as scalar mean/p95.
- oracle이 필요한가: score-set proxy로 계산 가능; oracle은 validation/evaluation용.
- 실험에서 실제로 사용되었는가: yes.
- decision preservation과 어떤 관계인가: direct.
- 관찰된 장점: single-action held-out oracle p95 ranking에서 가장 강함.
- 관찰된 한계: scalar additive plan에서는 interaction/cancellation을 놓칠 수 있고 frozen aggregate에서 vector main보다 약함.
- 근거: `results/go_no_go/metric_ranking_summary.csv`; `results/repro_v1/aggregate_v1/aggregate_summary.csv`.

### Mean vs p95

- 무엇을 측정하는가: mean은 average risk, p95는 tail risk.
- 왜 baseline 또는 후보 metric인가: deployment에서 rare high-risk samples도 중요할 수 있다.
- planner에 직접 적용 가능한가: yes.
- oracle이 필요한가: proxy/evaluation 모두 가능.
- 실험에서 실제로 사용되었는가: yes.
- decision preservation과 어떤 관계인가: p95는 near-boundary/tail erosion을 더 강조한다.
- 관찰된 장점: single-action ranking에서 decision_risk_p95 strongest.
- 관찰된 한계: vector planner final에서는 signed mean objective가 signed p95보다 더 좋은 aggregate를 보였다.
- 근거: `results/go_no_go/metric_ranking_summary.csv`; `results/repro_v1/aggregate_v1/aggregate_summary.csv`.

### Signed-mean vs signed-p95

- 무엇을 측정하는가: signed score-delta summed plan risk의 average vs tail.
- 왜 baseline 또는 후보 metric인가: cancellation-aware plan objective.
- planner에 직접 적용 가능한가: yes in vector beam planner.
- oracle이 필요한가: score-delta vectors from calibration; no oracle for planning.
- 실험에서 실제로 사용되었는가: yes.
- decision preservation과 어떤 관계인가: summed score perturbation relative to margin approximates plan-level decision movement.
- 관찰된 장점: signed mean is current main winner in frozen aggregate.
- 관찰된 한계: signed p95 did not beat signed mean overall; beam heuristic not exact.
- 근거: `vector_beam_planner.py`; `results/repro_v1/aggregate_v1/aggregate_summary.csv`.

### Flip rate vs accuracy

- 무엇을 측정하는가: flip rate compares compressed prediction to FP32 prediction; accuracy compares compressed prediction to ground-truth label.
- 왜 baseline 또는 후보 metric인가: decision preservation research cares about teacher decision stability, not only labels.
- planner에 직접 적용 가능한가: flip as objective is sparse/non-smooth but evaluable; accuracy requires labels and may hide changed-but-still-correct/incorrect decisions.
- oracle이 필요한가: full forward and labels for accuracy; FP32/compressed predictions for flip.
- 실험에서 실제로 사용되었는가: yes.
- decision preservation과 어떤 관계인가: flip is direct but sparse; risk gives graded margin erosion signal.
- 관찰된 장점: intuitive deployment metric.
- 관찰된 한계: sparse/tie-dominated in ranking; go/no-go uses risk primary.
- 근거: `metrics.py`; `experiments/go_no_go/analyze_rankings.py`.

### Rank correlation

- 무엇을 측정하는가: proxy candidate ranking vs oracle target ranking consistency.
- 왜 baseline 또는 후보 metric인가: planning relies on ranking/action selection under budget.
- planner에 직접 적용 가능한가: not objective itself, but validates objective.
- oracle이 필요한가: yes for validation analysis.
- 실험에서 실제로 사용되었는가: Spearman, Kendall tau-b, top-k recall.
- decision preservation과 어떤 관계인가: ensures score proxy selects truly risky/safe actions.
- 관찰된 장점: go/no-go benchmark clearly separated metrics.
- 관찰된 한계: single-action ranking does not guarantee multi-layer plan ranking.
- 근거: `experiments/go_no_go/analyze_rankings.py`; `results/go_no_go/metric_ranking_summary.csv`.

### Oracle vs held-out oracle vs proxy

- 무엇을 측정하는가: proxy는 score/calibration set estimate; held-out oracle는 disjoint set full-forward evaluation; confirmation/test는 stronger fixed evaluation.
- 왜 baseline 또는 후보 metric인가: separates selection from validation.
- planner에 직접 적용 가능한가: proxy yes; oracle should not be used for final plan tuning.
- oracle이 필요한가: for validation/evaluation.
- 실험에서 실제로 사용되었는가: yes.
- decision preservation과 어떤 관계인가: prevents overfitting proxy to particular score samples.
- 관찰된 장점: frozen protocol has explicit no confirmation/test selection policy.
- 관찰된 한계: development oracle still train-pool; test only binary CIFAR-10.
- 근거: `results/repro_v1/seed_101/pipeline_manifest.json`; `evaluate_plans_on_test.py`.

## 10. Failed, Rejected, or Deferred Directions

### Gradient-based planner

- 제안 이유: sensitivity/Jacobian/gradient can approximate output perturbation cheaply.
- 시도 여부: repository implementation not found.
- 보류/폐기 이유: current implemented pipeline is forward-only and gradient-free; no evidence of gradient planner experiments.
- 다시 검토할 조건: need compare gradient proxy vs decision-risk/vector proxy on same held-out oracle ranking.
- 현재 판단: 미구현/[확인 불가].
- 근거: no gradient planner files in `*.py`; planning/evaluation scripts use inference/forward paths.

### 지나치게 비싼 recalibration

- 제안 이유: after selecting actions, recompute residual risks/interactions sequentially.
- 시도 여부: implementation not found.
- 보류/폐기 이유: would require repeated full/model forwards for many partial plans; current protocol uses fixed score-delta vectors/scalar tables.
- 다시 검토할 조건: vector proxy failure cases show systematic errors that recalibration could fix.
- 현재 판단: 보류/미구현.
- 근거: `vector_beam_planner.py` payload fields `interaction_updates=False`; `analyze_vector_additive_proxy.py`.

### 단순 layer-wise 독립 평가

- 제안 이유: easy to score all single actions and add risks.
- 시도 여부: implemented in old/scalar additive planners.
- 보류/폐기 이유: interaction/cancellation problem; vector proxy analysis suggests scalar additive p95 less aligned with plan oracle p95.
- 다시 검토할 조건: if action interactions are negligible in a target domain.
- 현재 판단: baseline으로 유지, main direction 아님.
- 근거: `additive_planner.py`; `results/go_no_go_vector_analysis_v1/vector_proxy_ranking_summary.csv`.

### Accuracy-only objective

- 제안 이유: standard compression metric.
- 시도 여부: accuracy logged but not current primary.
- 보류/폐기 이유: decision preservation asks whether compressed model preserves FP32 decisions, including cases where accuracy remains similar.
- 다시 검토할 조건: application only cares about label accuracy and has no teacher-decision consistency requirement.
- 현재 판단: secondary metric.
- 근거: `metrics.py`; `experiments/go_no_go/metrics.py`; result CSVs include both accuracy and flip/risk.

### Latency proxy

- 제안 이유: compression often targets latency as well as memory.
- 시도 여부: no latency measurement/proxy implementation found.
- 보류/폐기 이유: current quantization is fake weight quantization; no hardware backend.
- 다시 검토할 조건: introduce real quantized kernels or validated latency lookup model.
- 현재 판단: 미구현.
- 근거: no latency result columns in inspected primary CSVs; `experiments/go_no_go/README.md` says storage ratios.

### Multi-class 확장

- 제안 이유: broader applicability beyond binary classifier.
- 시도 여부: no implementation found.
- 보류/폐기 이유: current risk formula depends on binary score `z1-z0`.
- 다시 검토할 조건: define top-1 margin erosion or class-pair margin risk for multi-class.
- 현재 판단: 보류.
- 근거: `config.py` class_ids `[0,1]`; `metrics.py`; `experiments/go_no_go/metrics.py`.

### Structured pruning 확장

- 제안 이유: compression planning action set could include pruning as well as quantization.
- 시도 여부: no pruning action implementation found.
- 보류/폐기 이유: action cost/risk and model surgery not implemented.
- 다시 검토할 조건: define structured pruning candidates, memory/latency cost, and score-delta/risk evaluation adapter.
- 현재 판단: 미구현 future direction.
- 근거: repository code inspection; no pruning scripts/actions found.

### Reinforcement learning formulation

- 제안 이유: sequential action selection can be modeled as RL.
- 시도 여부: no implementation found.
- 보류/폐기 이유: current problem has deterministic finite action space and strong forward-only proxies; RL unnecessary at current scale.
- 다시 검토할 조건: action space becomes too large/non-differentiable with expensive delayed rewards.
- 현재 판단: 보류/[확인 불가].
- 근거: no RL code/results found.

### Theoretical bound의 실용성 문제

- 제안 이유: margin/perturbation bounds could justify decision preservation guarantees.
- 시도 여부: final-layer no-flip sufficient condition, exact earlier-layer recurrence, and first-order propagation derivations are documented in `docs/formalization_notes.md`; no implemented certified planner/bound found.
- 보류/폐기 이유: formalization notes themselves mark these as analytical derivations, not active planner objectives or guarantees. ReLU mask changes, residual graph handling, cross terms, and nonlinear effects limit direct practical use.
- 다시 검토할 조건: derive bound that predicts planner failures/successes better than empirical proxies.
- 현재 판단: analytical motivation으로 보존, claim/guarantee로는 보류.
- 근거: `docs/formalization_notes.md`, sections 1.1-1.3 and 13.

### Metric이 proxy로 충분하지 않은 문제

- 제안 이유: single-action proxy may not predict combined plan.
- 시도 여부: observed in scalar additive vs vector proxy analysis.
- 보류/폐기 이유: current vector planner addresses this partially but not with true interaction updates.
- 다시 검토할 조건: compare predicted vector risk vs actual post-action risk plan-by-plan, identify systematic residuals.
- 현재 판단: open P0/P1 issue.
- 근거: `results/go_no_go_vector_analysis_v1/vector_proxy_ranking_summary.csv`; `results/repro_v1/aggregate_v1/aggregate_summary.csv`.

## 11. Current Research Claims: What Can and Cannot Be Claimed

### 현재 근거로 주장 가능한 내용

| Claim | Supporting evidence | Strength | Caveat |
| ----- | ------------------- | -------- | ------ |
| Binary CIFAR-10 ResNet-18에서 uniform INT4 fake quantization은 high parameter saving but high decision flip/risk를 만든다. | `results/uniform_baselines.csv` | 강함 | one model/task; fake quant only |
| Layer sensitivity is highly nonuniform; `conv1 int4` is especially risky in pilot. | `results/validation_single_layer_sweep.csv`; `results/go_no_go/single_action_metrics_seed0.csv` | 강함 for pilot | may differ by checkpoint/model |
| Weight L2 is a weak decision-risk proxy compared with decision-aware metrics. | `results/weight_l2_mixed_controls.csv`; `results/go_no_go/metric_ranking_summary.csv` | 강함 in this repo | not universal proof |
| `decision_risk_p95` score metric strongly matches held-out oracle single-action p95 ranking. | `results/go_no_go/metric_ranking_summary.csv` | 강함 for single-action benchmark | does not guarantee multi-layer plan optimality |
| Vector signed score-delta proxy better captures plan-level risk ranking than scalar additive p95 in pilot analysis. | `results/go_no_go_vector_analysis_v1/vector_proxy_ranking_summary.csv` | 중간 | proxy still linear, no true interaction update |
| Frozen 3-seed descriptive aggregate supports `vector_signed_mean_risk` planner over six scalar baselines on locked-test p95. | `results/repro_v1/aggregate_v1/aggregate_summary.csv` | 현재 최강 | descriptive only, repeated cells not independent |
| Current implemented pipeline is forward-only/gradient-free for planning/evaluation. | Code in `experiments/go_no_go/*.py` and root scripts | 강함 | rationale vs gradients not formally proven |

### 현재 근거로 주장하면 안 되는 내용

| Overclaim risk | Why unsupported | What evidence is needed |
| -------------- | --------------- | ----------------------- |
| "이 방법은 모든 compression planning 문제에 일반화된다." | only binary CIFAR-10 ResNet-18 fake quantization tested | multiple datasets/models/tasks/actions |
| "실제 hardware latency를 줄인다." | no latency measurement; fake quantized float model | hardware backend/latency benchmark |
| "structured pruning까지 검증됐다." | pruning action not implemented | pruning candidates, cost model, experiments |
| "vector planner is globally optimal." | beam search heuristic; manifest says no optimality guarantee | exact solver or proof |
| "theoretical decision preservation guarantee가 있다." | no formal bound in repo | theorem + empirical validation |
| "test result를 보고 고른 objective이다." | protocol says no confirmation/test selection; but must preserve this discipline | preregistered/frozen protocol logs for future changes |
| "signed-p95 is the best objective." | aggregate shows signed-p95 does not beat signed-mean overall | new evidence across seeds/budgets |
| "accuracy preservation equals decision preservation." | flip/risk and accuracy differ conceptually and empirically | analyze both and define application objective |
| "oracle metric can be used in planner without leakage." | oracle/confirmation/test are evaluation splits | strict split policy and new calibration-only planner |

Literature/novelty positioning currently supported only as context: `docs/formalization_notes.md` lists HAWQ, HAWQ-V2, BRECQ, and HAWQ-V3 as related mixed-precision/PTQ/resource-constrained work. The repo does **not** yet contain a verified literature review proving novelty. A defensible provisional contribution is narrower: a binary decision-preservation compression-planning framing with forward-only proxy evaluation and a vector signed score-delta planner that beats scalar baselines in this pilot. 상태: 부분 검토/과장 주의. 근거: `docs/formalization_notes.md`, section 14; `results/repro_v1/aggregate_v1/aggregate_summary.csv`.

## 12. Open Questions and Decision Log

### P0 — 현재 vector_signed_mean claim이 재현 가능한가

- 질문: existing artifacts에서 aggregate를 재계산하거나 minimal rerun으로 같은 18/18 win result가 유지되는가.
- 왜 중요한가: 현재 연구 주장 전체의 중심이다.
- 현재까지의 근거: `aggregate_summary.csv` and `per_seed_budget_results.csv`; `pipeline_manifest.json`.
- 가능한 답: 그대로 재현 / 일부 mismatch / script bug.
- 가장 저렴한 다음 검증 실험: `aggregate_frozen_repro_results.py`를 existing result root에 대해 재실행하고 CSV diff.
- 성공/실패 판정 기준: aggregate summary exact or numerically identical; mismatch면 claim freeze.
- 의존성: existing CSV/JSON intact.
- 우선순위: P0.

### P0 — Vector proxy prediction vs actual post-action forward gap

- 질문: vector planner가 실제 combined action forward risk를 얼마나 잘 예측하는가.
- 왜 중요한가: novelty가 "interaction-aware planning"이면 proxy/actual relationship이 핵심이다.
- 현재까지의 근거: vector proxy ranking analysis and locked-test aggregate.
- 가능한 답: good ranking but biased scale / fails at high budget / only mean objective robust.
- 가장 저렴한 다음 검증 실험: for all selected plans, join vector predicted risk, scalar predicted risk, confirmation/test actual p95 by seed/budget.
- 성공/실패 판정 기준: monotonic/ranking consistency and residual analysis.
- 의존성: vector plan JSON, vector summaries, confirmation/test CSV.
- 우선순위: P0.

### P0 — Claim wording for paper/presentation

- 질문: "decision preservation compression planner"를 어떻게 과장 없이 표현할 것인가.
- 왜 중요한가: novelty claim 과장 위험이 크다.
- 현재까지의 근거: Section 11.
- 가능한 답: "for binary decision-preserving fake-quant planning"로 제한 / broader claim 보류.
- 가장 저렴한 다음 검증 실험: no experiment; write claim table and map each phrase to evidence.
- 성공/실패 판정 기준: every claim has file evidence and caveat.
- 의존성: this handoff.
- 우선순위: P0.

### P1 — Calibration set size/seed sensitivity

- 질문: score size 512와 seeds 0/1/2가 충분한가.
- 왜 중요한가: planner proxy quality depends on calibration set.
- 현재까지의 근거: score seeds 0/1/2 and frozen seeds 101/202/303.
- 가능한 답: robust / larger score needed / specific seed brittle.
- 가장 저렴한 다음 검증 실험: rerun benchmark/vector planner with score sizes 128/256/1024 on one frozen checkpoint, evaluate confirmation only.
- 성공/실패 판정 기준: ranking and selected plan p95 stable.
- 의존성: compute time.
- 우선순위: P1.

### P1 — More training seeds or datasets

- 질문: three frozen seeds are enough for descriptive confidence?
- 왜 중요한가: generalization/statistical strength.
- 현재까지의 근거: 3 seeds only.
- 가능한 답: add seeds; switch to another binary pair; use CIFAR-100 binary subset.
- 가장 저렴한 다음 검증 실험: add 2-3 seeds with same protocol; do not tune.
- 성공/실패 판정 기준: vector main win rate and p95 reduction remain positive.
- 의존성: training time.
- 우선순위: P1.

### P1 — Latency and real quantization

- 질문: parameter saving proxy maps to actual latency/memory benefits?
- 왜 중요한가: compression relevance.
- 현재까지의 근거: none beyond theoretical bytes.
- 가능한 답: no latency gain in fake model / real INT backend needed.
- 가장 저렴한 다음 검증 실험: implement measured CPU/GPU latency for exported/real quantized candidate models or clearly label memory-only.
- 성공/실패 판정 기준: reliable timing protocol and correlation with saving.
- 의존성: backend selection.
- 우선순위: P1.

### P2 — Multi-class risk formalization

- 질문: binary score margin risk를 multi-class로 어떻게 generalize?
- 왜 중요한가: broader impact.
- 현재까지의 근거: binary-only implementation.
- 가능한 답: top1-vs-runner-up margin erosion; all-pair margin risk; KL/decision flip hybrid.
- 가장 저렴한 다음 검증 실험: define formula and run small CIFAR-10 multi-class benchmark.
- 성공/실패 판정 기준: proxy ranking vs held-out oracle.
- 의존성: model training.
- 우선순위: P2.

### P2 — Structured pruning action set

- 질문: vector decision-risk planning works for structured pruning?
- 왜 중요한가: original broader compression planning direction.
- 현재까지의 근거: no implementation.
- 가능한 답: action deltas can be collected similarly / cost model more complex.
- 가장 저렴한 다음 검증 실험: add one simple structured channel/filter pruning candidate per layer as benchmark only, not full planner.
- 성공/실패 판정 기준: score/oracle risk table generated and comparable.
- 의존성: pruning implementation.
- 우선순위: P2.

## 13. Recommended Next Steps

### Step 1 — Minimal current-claim reproduction

- 목적: 현재 가장 강한 claim이 script와 artifacts로 재현되는지 확인.
- 변경할 파일: ideally none. If needed, only add a small verification script under `experiments/go_no_go/` after reproducing current pipeline.
- 실행 방법: rerun `experiments.go_no_go.aggregate_frozen_repro_results` on `results/repro_v1`.
- 필요한 실험: existing artifact aggregate recomputation.
- 기록할 지표: p95_win_count, relative_p95_reduction_mean, mean_p95_delta for vector main vs scalar baselines.
- 성공 기준: `aggregate_summary.csv`와 동일 or numerical tolerance 내 동일.
- 실패 시 해석: current strongest claim blocked until aggregation mismatch explained.
- 다음 분기: success면 Step 2; fail이면 bug audit.
- 예상 연구적 가치: highest. 주장 신뢰도의 바닥을 확인한다.

### Step 2 — Proxy-vs-actual interaction audit

- 목적: vector signed proxy가 왜 좋은지, 어디서 실패하는지 분석.
- 변경할 파일: new analysis script may be added, e.g. `experiments/go_no_go/analyze_plan_proxy_actual_gap.py`.
- 실행 방법: join vector plan summaries, scalar plan summaries, confirmation/test evaluation by seed/budget/selection.
- 필요한 실험: no new model forward if existing CSV/JSON enough.
- 기록할 지표: predicted vector mean/p95, actual confirmation/test p95, Spearman/Kendall by budget, residuals, high-budget failures.
- 성공 기준: vector proxy ranking correlates with actual p95 and explains scalar failures.
- 실패 시 해석: vector planner works empirically but mechanism claim must be weakened.
- 다음 분기: if gap systematic, consider recalibration; if stable, write contribution around cancellation-aware proxy.
- 예상 연구적 가치: high, directly supports novelty.

### Step 3 — Calibration size ablation

- 목적: score set size/seed sensitivity 확인.
- 변경할 파일: possibly pipeline argument wrapper only.
- 실행 방법: on one frozen checkpoint, rerun benchmark/vector plans for score sizes 128/256/512/1024; evaluate on confirmation only.
- 필요한 실험: single checkpoint, fixed budgets .80/.84/.86.
- 기록할 지표: ranking tau, selected plan overlap, confirmation p95/flip/actual saving.
- 성공 기준: 512 is stable or minimal sufficient size identified.
- 실패 시 해석: current claim depends on calibration size; protocol must specify larger set.
- 다음 분기: update planner defaults or caveat.
- 예상 연구적 가치: medium-high.

### Step 4 — Add more frozen seeds without changing planner

- 목적: descriptive result robustness.
- 변경할 파일: none or run manifest only; do not tune objectives.
- 실행 방법: train seeds 404/505/606 using `train_frozen_binary_seed.py`, run frozen pipeline with same budgets/settings.
- 필요한 실험: full pipeline for new seeds.
- 기록할 지표: same aggregate; vector main win rate; signed-p95 ablation.
- 성공 기준: vector main remains consistently lower p95.
- 실패 시 해석: claim becomes seed-dependent; investigate model-specific layer sensitivities.
- 다음 분기: analyze failed seeds.
- 예상 연구적 가치: high, but compute cost higher.

### Step 5 — Presentation/paper claim lock

- 목적: 과장 없는 contribution wording.
- 변경할 파일: docs/slides/paper draft if they exist later.
- 실행 방법: use Section 11 claim table as checklist.
- 필요한 실험: none unless claim lacks evidence.
- 기록할 지표: every claim mapped to evidence/caveat.
- 성공 기준: no novelty/generalization statement without support.
- 실패 시 해석: presentation risks overclaiming.
- 다음 분기: remove or mark speculative.
- 예상 연구적 가치: high for communication quality.

## 14. Agent Operating Rules for This Project

1. 실험 결과를 보기 전에 결론을 단정하지 말 것.
2. `oracle`, `held-out oracle`, `confirmation`, `test`, `proxy/score set`을 혼동하지 말 것.
3. Accuracy와 decision preservation을 같은 개념으로 취급하지 말 것.
4. Planner output/proxy objective와 실제 post-action full forward 결과를 구분할 것.
5. Mean risk와 tail risk/p95 risk를 함께 확인할 것.
6. Result file, seed, split, config, checkpoint, git commit을 연결해서 기록할 것.
7. 새로운 주장 전에 weight L2, activation, KL, abs delta, decision-risk scalar baselines와 ablation을 확인할 것.
8. 자료가 없는 내용은 `[추정]`, `[확인 불가]`, `[코드 구현 여부 미확인]`, `[대화에서만 언급됨]`으로 명시할 것.
9. 코드 변경 전 현재 pipeline을 existing artifacts로 재현할 것.
10. 논문 아이디어와 실제 검증된 결과를 분리할 것.
11. Locked test result로 method/objective/hyperparameter를 retune하지 말 것.
12. `memory_saving_ratio`를 latency나 deployed memory saving으로 말하지 말 것.
13. `vector_signed_mean_risk`의 current win을 "global optimum"이나 "proved interaction modeling"으로 표현하지 말 것.
14. Smoke results보다 full/frozen results를 우선할 것.
15. Old margin risk와 current decision risk를 한 표에 넣을 때 반드시 수식 차이를 설명할 것.

## 15. Appendix

### 15.1 전체 파일 경로 인덱스

Research docs:

- `CLAUDE.md`
- `docs/experiment_log.md`
- `docs/formalization_notes.md`
- `experiments/go_no_go/README.md`
- `docs/ACTIVE_RESEARCH_STATE.md` [referenced by `CLAUDE.md` but missing]

Root code:

- `activation_proxy.py`
- `additive_planner.py`
- `config.py`
- `data.py`
- `experiment_logger.py`
- `metrics.py`
- `model.py`
- `plot_results.py`
- `quantization.py`
- `risk_configurations.py`
- `run_activation_proxy.py`
- `run_additive_planner.py`
- `run_baselines.py`
- `run_oracle_guided_controls.py`
- `run_ranked_controls.py`
- `sweep_single_layer.py`
- `train.py`
- `utils.py`

Go/no-go code:

- `experiments/go_no_go/__init__.py`
- `experiments/go_no_go/adapters.py`
- `experiments/go_no_go/aggregate_frozen_repro_results.py`
- `experiments/go_no_go/analyze_rankings.py`
- `experiments/go_no_go/analyze_vector_additive_proxy.py`
- `experiments/go_no_go/collect_score_delta_vectors.py`
- `experiments/go_no_go/evaluate_plans_on_confirmation.py`
- `experiments/go_no_go/evaluate_plans_on_test.py`
- `experiments/go_no_go/make_confirmation_split.py`
- `experiments/go_no_go/metrics.py`
- `experiments/go_no_go/planner_eval.py`
- `experiments/go_no_go/run_frozen_repro_pipeline.py`
- `experiments/go_no_go/run_single_action_benchmark.py`
- `experiments/go_no_go/splits.py`
- `experiments/go_no_go/train_frozen_binary_seed.py`
- `experiments/go_no_go/vector_beam_planner.py`

Environment/dependency files:

- `environment.yml`
- `requirements.txt`
- `requirements-lock.txt`

Flat result CSV/JSON:

- `results/activation_proxy_smoke.csv`
- `results/empirical_margin_dense_allocations.csv`
- `results/empirical_margin_dense_results.csv`
- `results/empirical_margin_p95_additive_planner_allocations.csv`
- `results/empirical_margin_p95_additive_planner_results.csv`
- `results/experiment_history.jsonl`
- `results/local_activation_dense_allocations.csv`
- `results/local_activation_dense_results.csv`
- `results/local_activation_p95_additive_planner_allocations.csv`
- `results/local_activation_p95_additive_planner_results.csv`
- `results/local_activation_p95_mixed_controls.csv`
- `results/oracle_guided_mixed_controls.csv`
- `results/single_layer_sweep.csv`
- `results/uniform_baselines.csv`
- `results/validation_local_activation_proxy.csv`
- `results/validation_local_activation_proxy_all_bits.csv`
- `results/validation_single_layer_sweep.csv`
- `results/validation_sweep_smoke.csv`
- `results/weight_l2_additive_planner_allocations.csv`
- `results/weight_l2_additive_planner_results.csv`
- `results/weight_l2_dense_allocations.csv`
- `results/weight_l2_dense_results.csv`
- `results/weight_l2_mixed_controls.csv`

Go/no-go pilot result groups:

- `results/go_no_go/single_action_metrics_seed{0,1,2}.csv`
- `results/go_no_go/metric_ranking_by_seed.csv`
- `results/go_no_go/metric_ranking_summary.csv`
- `results/go_no_go/split_indices.json`
- `results/go_no_go_analysis_v1/metric_ranking_by_seed.csv`
- `results/go_no_go_analysis_v1/metric_ranking_summary.csv`
- `results/go_no_go_planner_v1/planner_comparison_summary.csv`
- `results/go_no_go_planner_v1/plan_{metric}_save_{5000,6000,7000,8000}bp.json`
- `results/go_no_go_planner_v1/plan_uniform_{fp16,int8,int4}_anchor.json`
- `results/go_no_go_planner_v2_stress/planner_comparison_summary.csv`
- `results/go_no_go_planner_v2_stress/plan_{metric}_save_{8200,8400,8500,8600}bp.json`
- `results/go_no_go_planner_v2_stress/plan_uniform_{fp16,int8,int4}_anchor.json`
- `results/go_no_go_vectors_v1/score_delta_vectors_seed{0,1,2}.json`
- `results/go_no_go_vectors_v1/score_delta_vectors_seed{0,1,2}.npz` [binary, sidecar only summarized]
- `results/go_no_go_vector_analysis_v1/vector_proxy_by_plan.csv`
- `results/go_no_go_vector_analysis_v1/vector_proxy_ranking_summary.csv`
- `results/go_no_go_vector_plans_v1/vector_planner_summary.csv`
- `results/go_no_go_vector_plans_v1/plan_vector_signed_{mean,p95}_save_{7000,8000,8200,8400,8500,8600}bp.json`
- `results/go_no_go_confirmation_v1/confirmation_split_indices.json`
- `results/go_no_go_confirmation_v1/confirmation_split_indices.npz` [binary, metadata summarized]
- `results/go_no_go_confirmation_eval_v1/confirmation_plan_results.csv`
- `results/go_no_go_confirmation_eval_v1/confirmation_plan_results.json`
- `results/go_no_go_confirmation_eval_v1/confirmation_primary_summary.csv`
- `results/go_no_go_test_eval_v1/test_plan_results.csv`
- `results/go_no_go_test_eval_v1/test_plan_results.json`
- `results/go_no_go_test_eval_v1/test_primary_summary.csv`
- `results/go_no_go_smoke*/*` [smoke artifacts, not primary]

Frozen repro result groups:

- `results/repro_v1/aggregate_v1/aggregate_summary.csv`
- `results/repro_v1/aggregate_v1/per_seed_budget_results.csv`
- `results/repro_v1/aggregate_v1/training_summary.csv`
- `results/repro_v1/aggregate_v1/vector_vs_scalar_pairwise.csv`
- `results/repro_v1/seed_{101,202,303}/training_log.csv`
- `results/repro_v1/seed_{101,202,303}/training_manifest.json`
- `results/repro_v1/seed_{101,202,303}/pipeline_manifest.json`
- `results/repro_v1/seed_{101,202,303}/pipeline_commands.txt`
- `results/repro_v1/seed_{101,202,303}/benchmark/single_action_metrics_seed{0,1,2}.csv`
- `results/repro_v1/seed_{101,202,303}/benchmark/split_indices.json`
- `results/repro_v1/seed_{101,202,303}/scalar_plans/planner_comparison_summary.csv`
- `results/repro_v1/seed_{101,202,303}/scalar_plans/plan_{metric}_save_{7000,8000,8200,8400,8500,8600}bp.json`
- `results/repro_v1/seed_{101,202,303}/scalar_plans/plan_uniform_{fp16,int8,int4}_anchor.json`
- `results/repro_v1/seed_{101,202,303}/vectors/score_delta_vectors_seed{0,1,2}.json`
- `results/repro_v1/seed_{101,202,303}/vectors/score_delta_vectors_seed{0,1,2}.npz` [binary, sidecar summarized]
- `results/repro_v1/seed_{101,202,303}/vector_plans/vector_planner_summary.csv`
- `results/repro_v1/seed_{101,202,303}/vector_plans/plan_vector_signed_{mean,p95}_save_{7000,8000,8200,8400,8500,8600}bp.json`
- `results/repro_v1/seed_{101,202,303}/confirmation_eval/confirmation_primary_summary.csv`
- `results/repro_v1/seed_{101,202,303}/confirmation_eval/confirmation_plan_results.csv`
- `results/repro_v1/seed_{101,202,303}/confirmation_eval/confirmation_plan_results.json`
- `results/repro_v1/seed_{101,202,303}/test_eval/test_primary_summary.csv`
- `results/repro_v1/seed_{101,202,303}/test_eval/test_plan_results.csv`
- `results/repro_v1/seed_{101,202,303}/test_eval/test_plan_results.json`

Binary/non-text artifacts:

- `checkpoints/resnet18_binary_best.pt`
- `checkpoints/repro_v1/resnet18_binary_seed{101,202,303}_final.pt`
- `results/**/*.npz`
- `results/figures/*` if present [not central to numeric claims]

### 15.2 주요 CSV/JSON column 및 field 설명

- Uniform/early result columns: `bits`, `memory_saving_ratio`, `accuracy`, `flip_rate`, `mean_margin_normalized_risk`, `p95_margin_normalized_risk`, `max_margin_normalized_risk`.
- Single-layer columns: `layer`, `bits`, `weight_relative_l2`, `memory_saving_ratio`, `flip_rate`, `accuracy`, old margin risk stats.
- Activation proxy columns: `layer`, `bits`, `mean_relative_activation_error`, `p95_relative_activation_error`, `max_relative_activation_error` or similar naming.
- Planner result columns: `requested_memory_saving_ratio`, `actual_memory_saving_ratio`/`actual_saving`, `accuracy`, `flip_rate`, risk stats.
- Go/no-go single-action columns: `score_seed`, `layer`, `action`, `bits`, `weight_rel_l2`, `activation_rel_mse`, `output_kl_mean`, `abs_delta_score_mean`, `decision_risk_mean`, `decision_risk_p95`, `decision_risk_violation_rate`, `oracle_accuracy`, `oracle_flip_rate`, `oracle_decision_risk_mean`, `oracle_decision_risk_p95`, `oracle_decision_risk_violation_rate`.
- Ranking summary columns: `score_metric`, `oracle_target`, `spearman_mean`, `spearman_std`, `kendall_tau_b_mean`, `kendall_tau_b_std`, `top5_recall_mean`, `top10_recall_mean`, warning fields.
- Vector proxy columns: `plan_id`, `budget`, `selection`, `vector_signed_mean_risk`, `vector_signed_p95_risk`, `vector_abs_sum_*`, `scalar_additive_*`, oracle target columns.
- Confirmation/test columns: `selection`, `requested_saving`, `actual_saving`, `confirmation_*` or `test_*` accuracy/flip/risk fields.
- Aggregate columns: baseline/selection identifiers, `n_cells`, `p95_win_count`, `p95_win_rate`, `relative_p95_reduction_mean`, `relative_p95_reduction_median`, `relative_p95_reduction_min`, `relative_p95_reduction_max`, `mean_p95_delta`, `mean_saving_delta`, `strict_pareto_count`, `near_pareto_count`.
- Pipeline manifest fields: `schema_version`, `checkpoint_path`, `run_name`, `git_commit_hash`, `canonical_split_paths`, `confirmation_split_paths`, `fixed_score_seeds`, `fixed_memory_saving_ratios`, `fixed_vector_settings`, `scalar_primary_metrics`, `commands`, `stage_completion_status`, `selection_policy`, `test_evaluation_is_terminal`.

### 15.3 핵심 용어 사전

- Compression planning: target saving/constraint 아래에서 layer별 compression action을 선택하는 문제.
- Mixed precision quantization: layer별 bit-width가 다를 수 있는 quantization.
- Fake quantization: quantize/dequantize된 float weight를 사용해 simulated quantization effect를 보는 방식.
- Decision preservation: compressed model이 FP32 teacher의 prediction/decision boundary behavior를 보존하는 것.
- Flip rate: compressed prediction이 FP32 prediction과 다른 비율.
- Margin: binary score `|z1-z0|`.
- Margin erosion: candidate score delta가 FP32 decision 방향의 margin을 줄이는 양.
- Decision risk: margin erosion divided by margin.
- Oracle: full forward evaluation target. 이 repo에서는 초기/현재 문맥에 따라 의미가 다르므로 split을 반드시 함께 적어야 한다.
- Held-out oracle: score set과 분리된 development oracle.
- Confirmation split: method selection 후 supportive evaluation용 disjoint train-pool split.
- Locked test: terminal evaluation; retuning 금지.
- Proxy: calibration/score set에서 계산해 planner가 직접 쓰는 estimate.
- Interaction problem: single-action risks를 단순 합산할 때 combined action의 cancellation/nonlinear effect를 놓치는 문제.
- Vector signed proxy: candidate별 signed score delta vectors를 합산해 plan risk를 estimate하는 proxy.
- Actual saving: selected plan이 달성한 parameter memory saving ratio.

### 15.4 약어 목록

- FP32: 32-bit floating point.
- FP16: 16-bit floating point.
- INT8/INT4: 8-bit/4-bit integer quantization action.
- KL: Kullback-Leibler divergence.
- L2: Euclidean norm.
- DP: dynamic programming.
- P95: 95th percentile.
- CSV/JSON/NPZ: result artifact formats.
- CNN: convolutional neural network.

### 15.5 날짜별 세부 사건 인덱스

- 2026-05-11: last-layer binary no-flip sufficient condition and scalar score/margin framing. 근거: `docs/formalization_notes.md`.
- 2026-05-11 to 2026-05-21: earlier-layer perturbation recurrence and first-order propagation derivation. 근거: `docs/formalization_notes.md`.
- 2026-05-21: binary pilot scope and layer-level action abstraction fixed. 근거: `docs/formalization_notes.md`.
- 2026-06-23: forward-only proxy and interaction problem formalized. 근거: `docs/formalization_notes.md`.
- 2026-06-23 19:58: automatic experiment logging added. Git: `8110751`.
- 2026-06-23 20:03: FP32 training logged. Git: `be4ab74`.
- 2026-06-23 20:13-20:16: uniform baseline logging and full result. Git: `508e875`, `e21d77a`, `24076f3`.
- 2026-06-23 20:20-20:35: single-layer sweep implementation and logs. Git: `2c18b22`, `0b77cee`, `1799d2e`, `9631b39`.
- 2026-06-23 20:50-20:54: oracle-guided controls and validation sweep. Git: `8d16f04`, `1ffab6c`, `d564839`.
- 2026-06-23 21:29-21:31: metric-ranked controls and weight L2 results. Git: `c90b3ba`, `b1df60e`.
- 2026-06-23 21:43-21:54: activation proxy and all-bit table. Git: `45e5b88`, `b9a942e`, `24b4911`, `5af2a23`, `5555bbb`.
- 2026-06-23 21:56-22:15: additive planner and dense comparison. Git: `75b73aa`, `0ad3c77`, `6c5dfba`, `ed521ba`, `92b914c`.
- 2026-06-24 11:40: go/no-go scaffold. Git: `bf889ea`.
- 2026-06-24 12:01: single-action decision-risk benchmark. Git: `f7ea4b9`.
- 2026-06-24 12:24: held-out action-risk ranking analysis. Git: `2f555c7`.
- 2026-06-24 12:30: violation-aware/scalar plan comparison. Git: `60c5a59`.
- 2026-06-24 13:11: score delta vectors. Git: `7404c5c`.
- 2026-06-24 13:27: signed vector plan risk analysis. Git: `c20818e`.
- 2026-06-24 13:35: confirmation split. Git: `5ca2b0a`.
- 2026-06-24 13:44: vector beam planner. Git: `1ce7658`.
- 2026-06-24 13:54: confirmation evaluation. Git: `8e3ebda`.
- 2026-06-24 14:00: test evaluation. Git: `606d583`.
- 2026-06-24 14:19: frozen multi-seed training protocol. Git: `e16578d`.
- 2026-06-24 14:33-14:44: frozen reproducibility pipeline and resume fix. Git: `08ff4f6`, `35dc2c4`.
- 2026-06-24 16:19-16:22: aggregate frozen results and pairwise key fix. Git: `21b76e5`, `e276d00`.
- 2026-06-24 to 2026-06-30: risk aggregation/reporting/no-overclaim discipline consolidated. 근거: `docs/formalization_notes.md`.

### 15.6 아직 해석되지 않은 결과 파일 목록

- All `.npz` binary arrays: split/vector content not fully printed in this document; JSON sidecars and code schema used.
- All individual plan JSON files under `results/go_no_go_planner_*`, `results/go_no_go_vector_plans_v1`, and `results/repro_v1/seed_*/{scalar_plans,vector_plans}`: summarized by CSVs, not individually expanded.
- `results/go_no_go_smoke*` files: smoke artifacts not used for central claims.
- Figure/image outputs if present: not used for numeric claims.
- Checkpoint `.pt` files: binary model states not decoded.

### 15.7 문서 작성 과정에서 발견된 모순 목록

1. Old risk vs current risk: `margin_normalized_risk` uses absolute score perturbation; `decision_risk` uses directional margin erosion.
2. Oracle terminology conflict: early validation/test oracle-style sweep vs go/no-go held-out development oracle.
3. Formalization-note signed risk vs code decision risk: `docs/formalization_notes.md` defines signed normalized margin loss that can be negative; `experiments/go_no_go/metrics.py` clips harmless margin movement to zero.
4. Weight L2 paradox: `weight_l2_top` controls can be bad while some bottom controls protect critical layers.
5. Dense budget nonmonotonicity: weight L2 plan at requested 0.8725 is much worse than neighboring requested budgets.
6. Pilot vs frozen strength: pilot vector test result supports direction but final claim should rely on frozen aggregate.
7. Signed-p95 intuition vs result: p95 objective sounds tail-aware but signed-mean wins in frozen aggregate.
8. Memory vs latency: files report parameter saving, not latency.
9. Structured pruning appears in research aspiration/formalization extension but not code/results.
10. `docs/formalization_notes.md` contains mojibake in punctuation; equations and status tags are readable, but exact prose quote should be avoided.
11. `CLAUDE.md` says `docs/ACTIVE_RESEARCH_STATE.md` is canonical current task state, but that file is absent in the current workspace.

### 15.8 Quality Verification Checklist

```text
[x] 연구 관련 대화/문서/결과 파일을 가능한 한 모두 조사했다.
[x] 시간 순서가 유지된다.
[x] 확정된 사실과 가설을 구분했다.
[x] 모든 핵심 주장에 파일 근거를 남겼다.
[x] oracle / held-out / proxy의 차이를 명확히 했다.
[x] 실험 결과의 숫자를 임의로 만들지 않았다.
[x] 실패한 방향과 보류된 방향을 삭제하지 않았다.
[x] 현재 코드 구현 상태와 문서상 계획을 분리했다.
[x] 다음 에이전트가 재현 가능한 실행 단서를 제공했다.
[x] 현재 주장 가능한 내용과 과장 위험을 분리했다.
[x] 개인적이거나 연구와 무관한 정보는 제외했다.
```
