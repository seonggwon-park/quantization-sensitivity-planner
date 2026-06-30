# Compression Planning Formalization Notes

> **Purpose.** This file is the current mathematical “source of truth” for the compression-planning project.  
> It separates (i) definitions that are already fixed for the binary-classification pilot, (ii) derivations used to motivate the project, and (iii) extensions that are **not** yet part of the implemented or experimentally established planner.
>
> **Scope.** The active pilot concerns **mixed-precision quantization for a binary classifier**. The decision-preservation formulation is defined relative to the original FP32 model’s prediction, not relative to the ground-truth label.
>
> **Status convention**
>
> - **[Fixed]**: agreed working definition; use this notation in future documents and code.
> - **[Evaluation definition]**: exact quantity computed by a full post-action forward pass.
> - **[Proxy / planner-side]**: quantity intended to be available to the planner; it is not automatically equal to the evaluation quantity.
> - **[Analytical derivation]**: useful for reasoning, but not the active planner objective unless explicitly implemented.
> - **[Deferred]**: future extension, not a current project claim.
>
> **Important non-claim.** This document does **not** claim that the current proxy is a proven upper bound, a certified robustness guarantee, or a globally optimal mixed-precision solution. Those would require additional assumptions and experiments.

---

## 0. Start here: current formalization in one page

### 0.1 Research question

Given an already trained FP32 binary classifier and a finite set of compression actions for each layer, choose a compression plan that achieves as much resource saving as possible **while preserving the original model’s decisions as reliably as possible**.

The key distinction is:

- **Accuracy** asks whether the compressed model agrees with the dataset label.
- **Decision preservation** asks whether the compressed model agrees with the original FP32 model.

These are related but not identical. A compressed model can preserve a wrong FP32 prediction; conversely, it can flip an FP32 prediction and become correct relative to the ground-truth label. The present project intentionally studies the first relation.

### 0.2 Fixed objects

- Baseline FP32 model: \(f_\theta\).
- Compression plan: \(A=(a_1,\dots,a_L)\), one action per layer.
- Compressed model induced by the plan: \(f_{\tilde\theta(A)}\).
- Binary decision score: \(s_\theta(x)\in\mathbb R\).
- FP32 prediction: \(\hat y_0(x)=\mathbb 1[s_\theta(x)\ge 0]\).
- Compressed prediction: \(\hat y_A(x)=\mathbb 1[s_{\tilde\theta(A)}(x)\ge 0]\).
- Preservation event: \(\hat y_A(x)=\hat y_0(x)\).
- Flip event: \(\hat y_A(x)\ne\hat y_0(x)\).

### 0.3 Core exact decision condition

Let

\[
m_0(x)=|s_\theta(x)|
\]

be the FP32 decision margin, and define the compressed score expressed in the **FP32 decision direction**

\[
\mu_A(x)
=
\operatorname{sgn}\!\big(s_\theta(x)\big)\,
s_{\tilde\theta(A)}(x).
\]

For all samples with \(s_\theta(x)\ne 0\),

\[
\hat y_A(x)=\hat y_0(x)
\quad\Longleftrightarrow\quad
\mu_A(x)>0.
\]

Equivalently,

\[
\hat y_A(x)\ne \hat y_0(x)
\quad\Longleftrightarrow\quad
\mu_A(x)\le 0,
\]

subject to the project’s explicit tie-breaking convention at score \(0\).

### 0.4 Main optimization form

The resource-first form is

\[
\begin{aligned}
\min_{A\in\mathcal A}
\quad & M(A) \\
\text{s.t.}\quad &
\widehat J_{\text{risk}}(A;\mathcal S_{\mathrm{cal}})
\le \tau,\\
& T(A)\le T_{\max}
\qquad\text{(only when a valid latency model is available).}
\end{aligned}
\tag{P1}
\]

The equivalent saving-first form is

\[
\begin{aligned}
\max_{A\in\mathcal A}
\quad & \operatorname{Save}_M(A) \\
\text{s.t.}\quad &
\widehat J_{\text{risk}}(A;\mathcal S_{\mathrm{cal}})
\le \tau.
\end{aligned}
\tag{P2}
\]

Here \(\widehat J_{\text{risk}}\) is a **planner-side proxy** unless the planner actually evaluates the fully compressed model. The true held-out evaluation must use an oracle/full-forward quantity, not the proxy alone.

---

# 1. Research evolution and why the notation looks this way

## 1.0 Chronological map

The dates below identify the order in which the working formulation was developed. They are internal research-record dates, not publication dates.

| Period | Main question | Formal outcome retained in this file |
|---|---|---|
| 2026-05-11 | Can a binary classifier’s final decision be preserved under final-layer quantization? | Scalar score, perturbation \(\delta s\), and the sufficient no-flip condition \(|\delta s|<|s|\). |
| 2026-05-11 to 2026-05-21 | Why does the final-layer argument fail when earlier layers are quantized? | Exact pre-activation recurrence and first-order propagation equation. |
| 2026-05-21 | How can the compression problem be made tractable for a first pilot? | Binary classification; parameter-level memory accounting; layer-level action units. |
| 2026-06-23 | How should a planner make decisions without gradients or exhaustive re-evaluation? | Separation of an analytical propagation model from a forward-only proxy; explicit interaction problem. |
| 2026-06-24 | Why is average behavior insufficient? | Mean and p95 aggregation of sample-wise decision risk. |
| 2026-06-25 | What is actually being evaluated, and how should rankings be checked? | Oracle vs. held-out oracle vs. proxy; Kendall \(\tau_b\) and Spearman rank evaluation. |
| 2026-06-28 | How should planner evidence be reported? | Retain both signed-mean and signed-p95; report values with flip rate, accuracy, saving, and held-out oracle results. |
| 2026-06-29 to 2026-06-30 | What is the narrow, defensible research claim? | Decision preservation is a target distinct from accuracy; claims must be limited to validated pilot evidence. |

The sections below retain this order rather than presenting the final notation as if it had appeared fully formed.

## 1.1 Stage 1 — last-layer binary decision bound

**Status: [Analytical derivation]**

The earliest clean formulation arose from the observation that binary classification reduces the final decision to the sign of one scalar score. Let the final binary score be

\[
s(x)=w^\top h_{L-1}(x)+b,
\]

where \(h_{L-1}\) is the penultimate representation and \(w,b\) are the final-layer parameters.

If only the final-layer weight is changed by quantization,

\[
\tilde w = w+\Delta w,
\]

then, holding \(h_{L-1}\) fixed,

\[
\tilde s(x)
=
(w+\Delta w)^\top h_{L-1}(x)+b
=
s(x)+\Delta w^\top h_{L-1}(x).
\]

Therefore

\[
\delta s(x)
:=
\tilde s(x)-s(x)
=
\Delta w^\top h_{L-1}(x).
\tag{1}
\]

By Hölder’s inequality,

\[
|\delta s(x)|
\le
\|\Delta w\|_q
\,
\|h_{L-1}(x)\|_p,
\qquad
\frac1p+\frac1q=1.
\tag{2}
\]

If

\[
|\delta s(x)|<|s(x)|,
\tag{3}
\]

then the sign of the score cannot change, hence the decision is preserved.

This gives a **sufficient** condition, not a necessary one:

\[
|\delta s(x)|<|s(x)|
\quad\Longrightarrow\quad
\operatorname{sgn}(\tilde s(x))
=
\operatorname{sgn}(s(x)).
\tag{4}
\]

It is only a final-layer result because \(h_{L-1}\) itself changes when earlier layers are compressed.

---

## 1.2 Stage 2 — exact perturbation recurrence for earlier layers

**Status: [Analytical derivation]**

For a general feed-forward layer, write

\[
u_\ell = W_\ell h_{\ell-1}+b_\ell,
\qquad
h_\ell=\phi_\ell(u_\ell),
\]

where \(u_\ell\) is the pre-activation and \(h_\ell\) is the post-activation.

Let quantization change the weight to

\[
\tilde W_\ell=W_\ell+\Delta W_\ell,
\]

and let the corresponding compressed activation be

\[
\tilde h_\ell=h_\ell+\delta h_\ell.
\]

The exact pre-activation perturbation is

\[
\begin{aligned}
\delta u_\ell
&:=
\tilde u_\ell-u_\ell\\
&=
(W_\ell+\Delta W_\ell)\tilde h_{\ell-1}+b_\ell
-
(W_\ell h_{\ell-1}+b_\ell)\\
&=
W_\ell\delta h_{\ell-1}
+
\Delta W_\ell h_{\ell-1}
+
\Delta W_\ell\delta h_{\ell-1}.
\end{aligned}
\tag{5}
\]

Equation (5) is exact. Its three terms have distinct meanings:

1. \(W_\ell\delta h_{\ell-1}\): propagated error arriving from earlier layers.
2. \(\Delta W_\ell h_{\ell-1}\): new local error injected by quantizing layer \(\ell\).
3. \(\Delta W_\ell\delta h_{\ell-1}\): a cross term between local weight perturbation and incoming perturbation.

The third term is often neglected in first-order analyses, but it is not identically zero. Removing it is an approximation and must never be presented as an exact derivation.

---

## 1.3 Stage 3 — local linearization and propagated activation error

**Status: [Analytical derivation]**

To express the effect of local compression errors on later layers, define the Jacobian of layer \(\ell\) with respect to its input, evaluated at the FP32 forward path:

\[
A_\ell(x)
:=
\left.
\frac{\partial h_\ell}{\partial h_{\ell-1}}
\right|_{\theta,x}.
\tag{6}
\]

Define \(e_\ell(a_\ell;x)\) as the direct activation-space error introduced by action \(a_\ell\) at layer \(\ell\) when its input is held at the FP32 activation. Then the first-order local model is

\[
\delta h_\ell
\approx
A_\ell(x)\delta h_{\ell-1}
+
e_\ell(a_\ell;x).
\tag{7}
\]

Recursively applying (7) gives

\[
\delta h_t
\approx
\sum_{k=1}^{t}
\left(
\prod_{r=k+1}^{t} A_r(x)
\right)
e_k(a_k;x).
\tag{8}
\]

The ordered product means

\[
\prod_{r=k+1}^{t} A_r
=
A_tA_{t-1}\cdots A_{k+1},
\]

and equals the identity map when \(k=t\).

Equation (8) says that each local perturbation \(e_k\) is transformed by all later layers before reaching depth \(t\). This is the formal reason that a layer cannot, in general, be judged only by its local weight error.

### ReLU special case

For a ReLU layer in an unchanged activation region,

\[
h_\ell = \operatorname{ReLU}(u_\ell),
\]

let

\[
D_\ell(x)
=
\operatorname{diag}
\left(
\mathbb 1[u_\ell(x)>0]
\right).
\]

Then

\[
A_\ell(x)=D_\ell(x)W_\ell.
\tag{9}
\]

If the ReLU activation mask is unchanged and the second-order cross term is ignored, then

\[
\delta h_\ell
\approx
D_\ell W_\ell\delta h_{\ell-1}
+
D_\ell\Delta W_\ell h_{\ell-1}.
\tag{10}
\]

Thus one possible first-order definition is

\[
e_\ell(a_\ell;x)
\approx
D_\ell(x)\Delta W_\ell(a_\ell)h_{\ell-1}(x).
\tag{11}
\]

### Why this derivation is not the active planner

Equation (8) is valuable as a conceptual explanation, but it has serious practical limitations:

- It needs Jacobian information or an equivalent local linear model.
- ReLU mask changes violate the fixed-region assumption.
- Residual connections require the computational graph, not merely a simple chain, to be handled correctly.
- The cross term in (5) and higher-order nonlinear effects are discarded.
- The project’s desired planner is **forward-only / gradient-free**. A Jacobian-based planner does not automatically satisfy that design goal.

Therefore, (8) motivates the need for a forward proxy; it is not itself the confirmed planner objective.

---

## 1.4 Stage 4 — restricting the pilot to binary classification

**Status: [Fixed scope decision]**

Binary classification was selected because it gives an unambiguous decision boundary:

\[
s(x)=0.
\]

For a two-logit classifier with logits \(z_0(x),z_1(x)\), use

\[
s(x)=z_1(x)-z_0(x).
\tag{12}
\]

Then

\[
\hat y_0(x)
=
\mathbb 1[s_\theta(x)\ge 0].
\tag{13}
\]

If the implementation emits a single logit, that scalar is already \(s(x)\), up to a possible sign convention.

This scope reduction does **not** imply that the eventual project is limited to binary classification. It makes the current preservation objective and its oracle evaluation clean enough to verify before attempting multi-class extensions.

---

# 2. Canonical notation

## 2.1 Models and plans

| Symbol | Meaning | Status |
|---|---|---|
| \(f_\theta\) | Original FP32 model | [Fixed] |
| \(L\) | Number of compression decision units; currently layer-level unless implementation says otherwise | [Fixed scope] |
| \(\ell\in\{1,\dots,L\}\) | Layer / compression unit index | [Fixed] |
| \(\mathcal A_\ell\) | Allowed actions at layer \(\ell\) | [Fixed] |
| \(a_\ell\in\mathcal A_\ell\) | Chosen action for layer \(\ell\) | [Fixed] |
| \(A=(a_1,\dots,a_L)\) | Full compression plan | [Fixed] |
| \(\mathcal A=\prod_{\ell=1}^{L}\mathcal A_\ell\) | Plan search space | [Fixed] |
| \(T_{\ell,a_\ell}\) | Parameter transformation caused by action \(a_\ell\) | [Fixed general form] |
| \(\tilde\theta(A)\) | Parameters after all transformations in plan \(A\) | [Fixed] |
| \(f_{\tilde\theta(A)}\) | Compressed model under plan \(A\) | [Fixed] |

For quantization,

\[
T_{\ell,a_\ell}(W_\ell)
=
Q_{a_\ell}(W_\ell),
\qquad
\tilde W_\ell(a_\ell)
=
W_\ell+\Delta W_\ell(a_\ell).
\tag{14}
\]

At the current stage, \(a_\ell\) is normally interpreted as a mixed-precision quantization action, such as a bit-width plus a fixed quantizer configuration.

---

## 2.2 Scores, predictions, and margins

For an input \(x\),

\[
s_0(x)
:=
s_\theta(x),
\qquad
s_A(x)
:=
s_{\tilde\theta(A)}(x).
\tag{15}
\]

The baseline and compressed predictions are

\[
\hat y_0(x)=\mathbb 1[s_0(x)\ge 0],
\qquad
\hat y_A(x)=\mathbb 1[s_A(x)\ge 0].
\tag{16}
\]

The baseline absolute margin is

\[
m_0(x)=|s_0(x)|.
\tag{17}
\]

The baseline-direction signed margin of the compressed model is

\[
\mu_A(x)
=
\operatorname{sgn}(s_0(x))\,s_A(x).
\tag{18}
\]

By construction,

\[
\mu_0(x)=m_0(x).
\tag{19}
\]

The direction-aligned margin change is

\[
\Delta\mu_A(x)
=
\mu_A(x)-m_0(x)
=
\operatorname{sgn}(s_0(x))
\big(s_A(x)-s_0(x)\big).
\tag{20}
\]

A negative \(\Delta\mu_A\) consumes decision margin; a positive \(\Delta\mu_A\) increases the margin in the original decision direction.

---

# 3. Exact decision-preservation quantities

## 3.1 Flip indicator

**Status: [Fixed] [Evaluation definition]**

Define

\[
F_A(x)
=
\mathbb 1\!\left[
\hat y_A(x)\ne\hat y_0(x)
\right].
\tag{21}
\]

For nonzero baseline scores and a documented convention at \(s_A(x)=0\),

\[
F_A(x)
=
\mathbb 1[\mu_A(x)\le 0].
\tag{22}
\]

This is the cleanest exact decision-preservation target.

### Split-level flip rate

For a sample set \(\mathcal S\),

\[
\operatorname{FlipRate}_{\mathcal S}(A)
=
\frac{1}{|\mathcal S|}
\sum_{x\in\mathcal S}F_A(x).
\tag{23}
\]

This is an **oracle evaluation metric** whenever \(s_A(x)\) comes from a real forward pass through the fully compressed model.

It is not the same as accuracy:

\[
\operatorname{Accuracy}_{\mathcal S}(A)
=
\frac1{|\mathcal S|}
\sum_{(x,y)\in\mathcal S}
\mathbb 1[\hat y_A(x)=y].
\tag{24}
\]

Neither metric determines the other.

---

## 3.2 Signed normalized margin-loss risk

**Status: [Fixed working risk definition; verify code-level sign convention before renaming columns]**

For all \(x\) with \(m_0(x)>0\), define the signed normalized margin loss

\[
R^{\mathrm{sgn}}_A(x)
=
\frac{m_0(x)-\mu_A(x)}{m_0(x)}
=
-\frac{
\operatorname{sgn}(s_0(x))
\big(s_A(x)-s_0(x)\big)
}{
|s_0(x)|
}.
\tag{25}
\]

Interpretation:

- \(R^{\mathrm{sgn}}_A(x)=0\): no score change.
- \(0<R^{\mathrm{sgn}}_A(x)<1\): the plan reduced the original-direction margin, but did not flip the decision.
- \(R^{\mathrm{sgn}}_A(x)=1\): the compressed score reaches the boundary.
- \(R^{\mathrm{sgn}}_A(x)>1\): the compressed score crosses the boundary, so the FP32 decision flips.
- \(R^{\mathrm{sgn}}_A(x)<0\): the compressed score moved farther into the original decision region.

The exact relation is

\[
F_A(x)=\mathbb 1
\left[
R^{\mathrm{sgn}}_A(x)\ge 1
\right],
\tag{26}
\]

again subject to the score-zero convention.

This relation is why a signed margin-loss statistic is more directly tied to decision preservation than raw output L2 error: it normalizes the displacement by the specific sample’s available decision margin and retains whether the displacement is harmful or helpful.

### Numerical zero-margin rule

Equation (25) is undefined if \(m_0(x)=0\). The implementation must specify one of the following rules:

1. Exclude exact-zero FP32 scores from risk aggregation and report their count.
2. Replace the denominator by \(\max(m_0(x),\varepsilon)\) and state that the threshold \(R=1\) is no longer exact.
3. Treat all near-zero-margin samples as a separate high-risk group.

Until code confirms one choice, the canonical mathematical definition remains (25) for \(m_0(x)>0\).

---

## 3.3 Absolute normalized score perturbation

**Status: [Fixed analytical comparison quantity, not the preferred signed objective]**

Define

\[
R^{\mathrm{abs}}_A(x)
=
\frac{|s_A(x)-s_0(x)|}{|s_0(x)|}.
\tag{27}
\]

If

\[
R^{\mathrm{abs}}_A(x)<1,
\]

then a decision flip is impossible. Therefore,

\[
R^{\mathrm{abs}}_A(x)<1
\quad\Longrightarrow\quad
F_A(x)=0.
\tag{28}
\]

But the converse is false: a sample can have \(R^{\mathrm{abs}}_A(x)>1\) and still avoid a flip if the score moves in the beneficial direction.

This is the key reason to preserve both quantities conceptually:

- \(R^{\mathrm{abs}}\) supports a simple sufficient no-flip condition.
- \(R^{\mathrm{sgn}}\) distinguishes margin consumption from margin expansion and maps exactly to the flip threshold in (26).

---

# 4. Risk aggregation: mean, p95, and signed statistics

## 4.1 Generic aggregation notation

Let \(R_A(x)\) denote a chosen sample-wise risk score. For a set \(\mathcal S\),

\[
J_{\mathrm{mean}}(A;\mathcal S)
=
\frac1{|\mathcal S|}
\sum_{x\in\mathcal S}R_A(x).
\tag{29}
\]

The tail-risk form is

\[
J_{\mathrm{p95}}(A;\mathcal S)
=
\operatorname{Quantile}_{0.95}
\left(
\left\{
R_A(x):x\in\mathcal S
\right\}
\right).
\tag{30}
\]

When \(R_A=R_A^{\mathrm{sgn}}\), write

\[
J_{\mathrm{signed\text{-}mean}}(A;\mathcal S)
=
\frac1{|\mathcal S|}
\sum_{x\in\mathcal S}R_A^{\mathrm{sgn}}(x),
\tag{31}
\]

and

\[
J_{\mathrm{signed\text{-}p95}}(A;\mathcal S)
=
\operatorname{Quantile}_{0.95}
\left(
\left\{
R_A^{\mathrm{sgn}}(x):x\in\mathcal S
\right\}
\right).
\tag{32}
\]

Some previous notation used a label such as

\[
J_{\mathrm{VS\text{-}mean}}(A)
=
\frac1{|\mathcal S|}
\sum_{x_i\in\mathcal S}\widehat R_A(x_i).
\tag{33}
\]

The symbol \(J\) denotes an aggregated objective / evaluation functional. The exact expansion of the historical label “VS” should not be guessed if it is not explicitly documented in the experiment code. In future writing, prefer unambiguous labels such as `signed_mean_risk` and `signed_p95_risk`.

---

## 4.2 Why mean and p95 must both be retained

\[
J_{\mathrm{signed\text{-}mean}}
\]

measures average normalized margin consumption over the set. It can remain small even if a small number of examples suffers severe harm.

\[
J_{\mathrm{signed\text{-}p95}}
\]

measures a high, but not worst-case, quantile of the per-sample risk distribution. It is intended to expose tail behavior without the instability of a maximum.

Neither one dominates the other:

- A plan can have lower mean and worse p95.
- A plan can have worse mean and lower p95.
- A plan can have good signed risk but still have nonzero flip rate if enough values cross \(1\).

**Current reporting rule.** Preserve and report both signed-mean and signed-p95. The current project discussion selected the mean statistic as the main planner score for the pilot, but p95 remains necessary as a diagnostic and should not be silently removed.

---

# 5. Oracle, held-out oracle, and planner proxy

This separation is mandatory. Do not use these terms interchangeably.

## 5.1 Oracle risk

**Status: [Fixed evaluation definition]**

For plan \(A\), run the actual compressed model \(f_{\tilde\theta(A)}\) end-to-end and compute \(s_A(x)\). Then define, for example,

\[
R^{\mathrm{oracle}}_A(x)
=
R^{\mathrm{sgn}}_A(x)
\tag{34}
\]

using (25), and aggregate it with (31) or (32).

The word **oracle** here means that the post-action outcome is known from a real forward computation. It does **not** mean that the ground-truth label is used.

The oracle flip rate is

\[
\operatorname{FlipRate}^{\mathrm{oracle}}_{\mathcal S}(A)
=
\frac1{|\mathcal S|}
\sum_{x\in\mathcal S}
\mathbb 1[
\hat y_A(x)\ne\hat y_0(x)
].
\tag{35}
\]

---

## 5.2 Held-out oracle risk

**Status: [Fixed evaluation protocol]**

Let

\[
\mathcal S_{\mathrm{cal}}
\cap
\mathcal S_{\mathrm{hold}}
=
\varnothing.
\tag{36}
\]

For the same already selected plan \(A\), compute

\[
J^{\mathrm{oracle}}_{\mathrm{p95}}
(A;\mathcal S_{\mathrm{hold}})
=
\operatorname{Quantile}_{0.95}
\left(
\left\{
R^{\mathrm{oracle}}_A(x)
:
x\in\mathcal S_{\mathrm{hold}}
\right\}
\right).
\tag{37}
\]

The formula is the same as the calibration-side oracle formula; the only difference is the input set. Its purpose is to test whether a decision or ranking made using calibration information transfers to unseen evaluation examples.

This is **not** an oracle for ground-truth accuracy. It is an oracle evaluation of actual compression-induced decision risk.

---

## 5.3 Planner-side proxy

**Status: [Fixed conceptual role; exact compositional formula not yet fixed]**

A planner cannot be evaluated as a useful planner if it simply performs an expensive full-forward oracle computation for every candidate plan. Define

\[
\widehat R_A(x)
\]

as a planner-side estimate of the corresponding oracle risk. Then

\[
\widehat J_{\mathrm{mean}}(A;\mathcal S_{\mathrm{cal}})
=
\frac1{|\mathcal S_{\mathrm{cal}}|}
\sum_{x\in\mathcal S_{\mathrm{cal}}
}
\widehat R_A(x),
\tag{38}
\]

and

\[
\widehat J_{\mathrm{p95}}(A;\mathcal S_{\mathrm{cal}})
=
\operatorname{Quantile}_{0.95}
\left(
\left\{
\widehat R_A(x)
:
x\in\mathcal S_{\mathrm{cal}}
\right\}
\right).
\tag{39}
\]

The planner’s core empirical question is not “is \(\widehat R\) numerically identical to \(R^{\mathrm{oracle}}\)?” It is at least:

1. Does \(\widehat R\) rank candidate layers or plans similarly to oracle risk?
2. Does choosing a plan with low \(\widehat J\) produce low **held-out oracle** risk?
3. Does it produce a better memory–risk trade-off than the baseline metrics?

### Critical notation rule

Use a hat only for a proxy/prediction. If an existing CSV or script uses `hat_R` for an actually measured full-forward quantity, document the legacy naming but do not perpetuate the ambiguity in new work.

---

# 6. Plan space and resource model

## 6.1 Layer-level action space

**Status: [Fixed pilot abstraction]**

For each layer \(\ell\),

\[
\mathcal A_\ell
=
\{
a_{\ell}^{(1)},\dots,a_{\ell}^{(K_\ell)}
\}.
\]

For a mixed-precision quantization pilot, each action may specify a bit-width \(b_{\ell,a}\), quantizer type, scale, or calibration configuration.

A full plan is

\[
A\in\mathcal A
=
\mathcal A_1\times\cdots\times\mathcal A_L.
\tag{40}
\]

The size of this space is

\[
|\mathcal A|
=
\prod_{\ell=1}^{L}
|\mathcal A_\ell|,
\tag{41}
\]

which becomes exponential in the number of layers when each layer has multiple candidate actions. This is why planner proxies and structured search are needed.

---

## 6.2 Memory

Let \(n_\ell\) be the number of stored parameters in layer \(\ell\). Under a weight-only fixed-bit approximation,

\[
M_\ell(a_\ell)
=
n_\ell b_{\ell,a_\ell}.
\tag{42}
\]

The plan memory is

\[
M(A)
=
\sum_{\ell=1}^{L}
M_\ell(a_\ell).
\tag{43}
\]

If all baseline weights are FP32,

\[
M_{\mathrm{FP32}}
=
\sum_{\ell=1}^{L}32n_\ell.
\tag{44}
\]

The actual memory saving ratio is

\[
\operatorname{Save}_M(A)
=
1-
\frac{M(A)}{M_{\mathrm{FP32}}}.
\tag{45}
\]

Equation (45) is exact only under the stated storage model. Real serialization may add scale, zero-point, metadata, alignment, or non-quantized parameter overhead. Therefore, experiment columns named `actual_saving` must state whether they use the idealized formula (45) or measured serialized size.

---

## 6.3 Latency

**Status: [Deferred / incomplete proxy]**

Let \(T(A)\) denote measured end-to-end latency on a specified device, runtime, batch size, input shape, and warm-up protocol.

A simple additive proxy would be

\[
\widehat T(A)
=
\sum_{\ell=1}^{L}
t_\ell(a_\ell),
\tag{46}
\]

where \(t_\ell(a_\ell)\) is a per-layer measurement or estimate. This is useful as a first approximation but is not guaranteed to be correct because kernels, memory movement, graph fusion, parallelism, and runtime overhead may make end-to-end latency non-additive.

Therefore, latency may be included as a future constraint only after the measurement protocol and proxy validity are established:

\[
T(A)\le T_{\max}
\quad\text{or}\quad
\widehat T(A)\le \widehat T_{\max}.
\tag{47}
\]

Do not state that latency is currently optimized unless the implementation and experiments explicitly demonstrate it.

---

# 7. Current optimization formulation

## 7.1 Resource-constrained risk minimization

A risk-first form is

\[
\begin{aligned}
\min_{A\in\mathcal A}
\quad &
\widehat J_{\mathrm{risk}}
(A;\mathcal S_{\mathrm{cal}})\\
\text{s.t.}\quad &
M(A)\le M_{\max},\\
&
\widehat T(A)\le \widehat T_{\max}
\quad\text{(optional / deferred).}
\end{aligned}
\tag{P3}
\]

This form asks: among plans that satisfy a resource budget, which one is predicted to preserve decisions best?

---

## 7.2 Risk-constrained compression maximization

The compression-first form is

\[
\begin{aligned}
\max_{A\in\mathcal A}
\quad &
\operatorname{Save}_M(A)\\
\text{s.t.}\quad&
\widehat J_{\mathrm{risk}}
(A;\mathcal S_{\mathrm{cal}})
\le \tau.
\end{aligned}
\tag{P4}
\]

This form asks: how much memory can be saved while keeping proxy decision risk below a tolerance?

For the current research framing, (P4) is usually the clearest expression because the end goal is compression under a decision-preservation requirement.

---

## 7.3 Lagrangian score for greedy or ranking baselines

A scalarized ranking score can be written as

\[
\mathcal L_\lambda(A)
=
\widehat J_{\mathrm{risk}}(A;\mathcal S_{\mathrm{cal}})
+
\lambda
\frac{M(A)}{M_{\mathrm{FP32}}}.
\tag{48}
\]

This may be useful for greedy search or plotting a trade-off. However:

- \(\lambda\) changes the implicit trade-off.
- Minimizing (48) is not the same as solving (P4) unless conditions are met.
- A scalarization can hide constraint violations.

Therefore, use it only as an explicit baseline or algorithmic device, not as proof of constrained optimality.

---

# 8. Single-layer evidence, multi-layer plans, and the interaction problem

## 8.1 Single-layer action measurement

For a candidate action \(a\) at layer \(\ell\), define the plan that changes only that layer:

\[
A^{(\ell,a)}
=
(a_1^{\mathrm{base}},\dots,a_{\ell-1}^{\mathrm{base}},
a,
a_{\ell+1}^{\mathrm{base}},\dots,a_L^{\mathrm{base}}).
\tag{49}
\]

Its oracle sample risk is

\[
R_{\ell,a}^{\mathrm{single}}(x)
=
R_{A^{(\ell,a)}}^{\mathrm{oracle}}(x).
\tag{50}
\]

This quantity can rank isolated layer actions. It does **not** determine the risk of a plan containing several non-baseline actions.

---

## 8.2 Why simple addition is not generally correct

A naive compositional surrogate would be

\[
\widehat R_A^{\mathrm{add}}(x)
=
\sum_{\ell=1}^{L}
R_{\ell,a_\ell}^{\mathrm{single}}(x).
\tag{51}
\]

Equation (51) is an understandable baseline, but it is not a theorem. It ignores:

- propagated activation changes;
- cross terms such as \(\Delta W_\ell\delta h_{\ell-1}\) from (5);
- nonlinearity and activation-mask changes;
- residual-branch recombination;
- possible cancellation of score perturbations;
- possible reinforcement of harmful perturbations.

The discrepancy is the interaction residual:

\[
I_A(x)
=
R_A^{\mathrm{oracle}}(x)
-
\sum_{\ell=1}^{L}
R_{\ell,a_\ell}^{\mathrm{single}}(x).
\tag{52}
\]

If \(I_A(x)\) is not negligible, independent layer ranking is insufficient for multi-layer planning.

---

## 8.3 Forward-only vector or sequential proxies

**Status: [Conceptual direction; exact final formula not fixed]**

A forward-only planner may attempt to retain more structure than a scalar single-layer metric. General notation:

\[
\widehat R_A(x)
=
\mathcal G
\left(
v_{1,a_1}(x),\dots,v_{L,a_L}(x)
\right),
\tag{53}
\]

where \(v_{\ell,a_\ell}(x)\) is a layer/action-specific forward-derived feature vector and \(\mathcal G\) is a composition rule.

Potential choices include:

- additive aggregation;
- conservative max-like aggregation;
- sequential recalibration after each selected action;
- a learned or fitted calibration-to-oracle map;
- vector propagation based on forward activations.

None of these should be called the confirmed project method until its exact definition, compute cost, and held-out validation are recorded.

---

## 8.4 Why repeated recalibration is a trade-off, not a free fix

Suppose a greedy planner chooses actions sequentially. After selecting \(a_1,\dots,a_k\), it may recompute features on the partially compressed model

\[
f_{\tilde\theta(a_1,\dots,a_k)}.
\]

This can reduce stale-feature error but increases computational cost. It also changes the planner’s computational budget and can approach oracle-like repeated evaluation if implemented too aggressively.

The project must report both:

\[
\text{planning quality}
\quad\text{and}\quad
\text{planning cost}.
\]

A method that achieves good risk only by repeated expensive full forwards may be useful as an upper-bound baseline but does not automatically satisfy the intended lightweight planner objective.

---

# 9. Baseline metrics and what they do — and do not — measure

## 9.1 Weight perturbation norm

For layer \(\ell\) and action \(a\),

\[
d_{\ell,a}^{W}
=
\left\|
\tilde W_\ell(a)-W_\ell
\right\|_F.
\tag{54}
\]

A normalized alternative is

\[
\bar d_{\ell,a}^{W}
=
\frac{
\left\|
\tilde W_\ell(a)-W_\ell
\right\|_F
}{
\|W_\ell\|_F+\varepsilon
}.
\tag{55}
\]

This measures the size of the parameter perturbation, not the importance of that perturbation for a particular input or decision margin.

---

## 9.2 Activation perturbation norm

For a set \(\mathcal S\),

\[
d_{\ell,a}^{h}
=
\frac1{|\mathcal S|}
\sum_{x\in\mathcal S}
\left\|
\tilde h_{\ell,a}(x)-h_\ell(x)
\right\|_2.
\tag{56}
\]

Here \(\tilde h_{\ell,a}\) must be defined precisely: it may be an activation under a single-layer action or under a full plan. The two are different objects and must not share a label without clarification.

Activation error incorporates data dependence but still does not directly account for later-layer amplification or the sample-specific binary decision margin.

---

## 9.3 Output-distribution divergence

Let

\[
p_0(x)
=
\operatorname{softmax}(z_0(x)/\tau_T),
\qquad
p_A(x)
=
\operatorname{softmax}(z_A(x)/\tau_T),
\]

where \(\tau_T>0\) is an optional temperature. The output KL divergence is

\[
D_{\mathrm{KL}}
\left(
p_0(x)\,\|\,p_A(x)
\right)
=
\sum_c
p_{0,c}(x)
\log
\frac{p_{0,c}(x)}{p_{A,c}(x)}.
\tag{57}
\]

KL measures output-distribution change. It is a useful baseline but not a direct decision-preservation target: a large change can leave the argmax unchanged, and a small change can flip a near-boundary prediction.

---

## 9.4 Empirical flip / empirical risk baseline

For a layer-action or plan that is actually forward-evaluated, empirical flip rate is (23). It is highly relevant to the target but may be expensive to compute for every candidate action or plan, which is why a planner proxy is needed.

Do not label an empirical full-forward result “proxy” merely because it is used for ranking.

---

## 9.5 Ranking evaluation

Suppose a proxy assigns a score \(q_i\) and an oracle assigns a score \(r_i\) to candidate actions or candidate plans \(i=1,\dots,N\).

### Kendall’s \(\tau_b\)

\[
\tau_b
=
\frac{C-D}
{
\sqrt{(C+D+T_q)(C+D+T_r)}
},
\tag{58}
\]

where \(C\) and \(D\) are the number of concordant and discordant pairs, and \(T_q,T_r\) account for ties.

Use \(\tau_b\) when ties may be common, which is plausible for discrete quantization actions and rounded metrics.

### Spearman rank correlation

Let \(\operatorname{rank}(q_i)\) and \(\operatorname{rank}(r_i)\) be ranks with an explicit tie rule. Spearman correlation is the Pearson correlation between those ranks:

\[
\rho_S
=
\operatorname{corr}
\left(
\operatorname{rank}(q_i),
\operatorname{rank}(r_i)
\right).
\tag{59}
\]

These correlations test ranking agreement, not calibration. A proxy can rank candidates well while its numerical values are poorly calibrated, and vice versa.

---

# 10. Calibration, held-out evaluation, and no-leakage rules

## 10.1 Split definitions

Use disjoint sets:

\[
\mathcal S_{\mathrm{cal}}
\cap
\mathcal S_{\mathrm{hold}}
=
\varnothing.
\tag{60}
\]

Optional test data can be denoted by

\[
\mathcal S_{\mathrm{test}},
\qquad
\mathcal S_{\mathrm{test}}
\cap
\left(
\mathcal S_{\mathrm{cal}}
\cup
\mathcal S_{\mathrm{hold}}
\right)
=
\varnothing.
\tag{61}
\]

### Calibration set

\[
\mathcal S_{\mathrm{cal}}
\]

may be used for quantizer calibration, proxy construction, action ranking, threshold selection, or plan selection. The exact allowed uses must be recorded per experiment.

### Held-out set

\[
\mathcal S_{\mathrm{hold}}
\]

is used to evaluate the selected plan or ranking rule without re-tuning. In the current project vocabulary, “held-out oracle p95 risk” means the p95 of actual full-forward risk on \(\mathcal S_{\mathrm{hold}}\), not a label-based accuracy statistic.

---

## 10.2 Minimum reporting tuple per plan

For every selected plan \(A\), report at least

\[
\left(
\operatorname{Save}_M(A),\;
\operatorname{Accuracy}_{\mathcal S_{\mathrm{hold}}}(A),\;
\operatorname{FlipRate}_{\mathcal S_{\mathrm{hold}}}(A),\;
J^{\mathrm{oracle}}_{\mathrm{signed\text{-}mean}}(A;\mathcal S_{\mathrm{hold}}),\;
J^{\mathrm{oracle}}_{\mathrm{signed\text{-}p95}}(A;\mathcal S_{\mathrm{hold}})
\right).
\tag{62}
\]

If planner quality is evaluated, additionally report

\[
\left(
\tau_b,\;
\rho_S,\;
\text{proxy-vs-oracle selected-plan gap}
\right).
\tag{63}
\]

The “selected-plan gap” should be explicitly defined, for example

\[
\Delta_{\mathrm{proxy\to oracle}}(A)
=
J^{\mathrm{oracle}}_{\mathrm{risk}}
(A;\mathcal S_{\mathrm{hold}})
-
\widehat J_{\mathrm{risk}}
(A;\mathcal S_{\mathrm{cal}}).
\tag{64}
\]

This gap mixes split shift and proxy error, so it should not be interpreted as proxy error alone.

---

# 11. Current extension path: structured pruning

**Status: [Deferred]**

The long-term action space may include structured pruning. For a structured mask or transformation \(P_{\ell,a}\),

\[
\tilde W_\ell(a)
=
P_{\ell,a}(W_\ell)
\tag{65}
\]

can represent channel, filter, head, block, or other structured removal.

Then the unified action space is

\[
\mathcal A_\ell
=
\mathcal A_\ell^{\mathrm{quant}}
\cup
\mathcal A_\ell^{\mathrm{prune}}
\cup
\mathcal A_\ell^{\mathrm{hybrid}}.
\tag{66}
\]

The generic planner equations (P1)–(P4) survive this extension. What changes is:

- the resource model \(M(A)\);
- the latency model \(T(A)\);
- the local perturbation \(e_\ell(a_\ell;x)\);
- the validity of any proxy trained only on quantization actions.

Therefore structured pruning should not be presented as already supported by a quantization-only pilot.

---

# 12. What is fixed vs. what is intentionally still open

## 12.1 Fixed

1. The active pilot is binary classification.
2. A plan is a layer-wise collection of compression actions.
3. The primary target is agreement with the FP32 model’s decision, not ground-truth accuracy alone.
4. The exact decision event is a prediction flip relative to FP32.
5. Margin-aligned signed risk is a meaningful working surrogate because the threshold \(R=1\) corresponds to crossing the FP32 decision boundary.
6. Oracle/full-forward evaluation, held-out oracle evaluation, and planner proxy are different objects.
7. Mean and p95 risk capture different aspects of the risk distribution and must both be reported.
8. The main planner formulation is memory saving subject to a risk constraint, or its equivalent constrained form.

## 12.2 Open / must be verified against code before being called “final”

1. The exact implementation of \(\widehat R_A(x)\).
2. Whether the current code uses raw margin, normalized margin, clipped margin, or another score transformation.
3. The exact quantile interpolation rule for p95.
4. The exact tie convention for \(s_A(x)=0\).
5. Whether `signed-*` experiment columns exactly implement (25) or a sign-equivalent variant.
6. The exact aggregation rule used by each planner variant.
7. Whether memory saving includes quantization metadata and non-quantized parameters.
8. Any latency proxy and device-specific measurement protocol.
9. The level of action granularity in every experiment: layer, block, channel, or parameter group.
10. Whether a claimed forward proxy uses no gradients and no repeated full-model recalibration.

---

# 13. Explicit non-claims and common failure modes

1. **Low weight L2 does not imply low decision risk.** It omits activations, downstream amplification, and decision margin.
2. **Low activation L2 does not imply no flip.** A small downstream score perturbation can flip a small-margin sample.
3. **Low KL does not imply preservation.** Argmax/sign boundaries can be crossed by small distribution shifts.
4. **High accuracy does not imply preservation.** Some FP32 errors may be corrected while some FP32 correct predictions may flip.
5. **A calibration-set oracle result is not held-out evidence.**
6. **A one-layer oracle ranking does not prove multi-layer planner validity.**
7. **An additive risk model is a baseline assumption, not a theorem.**
8. **The first-order propagation equation is not a guaranteed bound in a ReLU/ResNet network unless its assumptions are proved and enforced.**
9. **A planner that repeatedly invokes expensive post-action full forwards may be a strong search baseline, but it is not automatically a lightweight forward-only planner.**
10. **A decision-preservation formulation is not automatically novel.** Novelty requires a clear algorithmic contribution and empirical evidence beyond defining a metric.

---

# 14. Literature placement (context only)

These references provide context for mixed-precision quantization and calibration-based PTQ. They do **not** validate the project’s new decision-preservation proxy by themselves.

1. Z. Dong et al., *HAWQ: Hessian Aware Quantization of Neural Networks with Mixed-Precision*, 2019.  
   Mixed-precision bit allocation guided by second-order sensitivity.  
   arXiv:1905.03696.

2. Z. Dong et al., *HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks*, 2019.  
   Extends the sensitivity formulation to a trace-weighted mixed-precision setting.  
   arXiv:1911.03852.

3. Y. Li et al., *BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction*, ICLR 2021.  
   A calibration-data-based PTQ method emphasizing reconstruction at the block level.  
   arXiv:2102.05426.

4. Z. Yao et al., *HAWQ-V3: Dyadic Neural Network Quantization*, 2020.  
   Uses an optimization formulation with application constraints such as size, latency, and bit operations.  
   arXiv:2011.10680.

---

# 15. Implementation checklist before Codex changes any code

- [ ] Confirm the model’s binary score convention: scalar logit or \(z_1-z_0\).
- [ ] Confirm the sign/tie convention at score zero.
- [ ] Locate the exact code that computes `risk`, `mean_risk`, `p95_risk`, `signed_mean`, and `signed_p95`.
- [ ] Verify whether all risk calculations use the baseline FP32 score and the compressed full-plan score.
- [ ] Verify whether each quantity is oracle/full-forward or proxy/planner-side.
- [ ] Verify the calibration/held-out split and ensure no plan-selection leakage.
- [ ] Verify the quantile implementation and sample count.
- [ ] Verify actual versus requested memory saving computation.
- [ ] Verify the action granularity used by every planner.
- [ ] Record seed, model checkpoint, quantizer configuration, and input preprocessing before comparing results.
- [ ] Do not add structured pruning to claims or figures until the action semantics and resource accounting are implemented.
- [ ] Do not rename legacy metrics until their exact code-level formula is matched to this document.

---

# 16. Short glossary

| Term | Meaning in this project |
|---|---|
| FP32 baseline | Original uncompressed reference model \(f_\theta\) |
| Plan | One compression action per decision unit/layer |
| Oracle | Actual post-action full forward computation |
| Held-out oracle | Same oracle computation on data not used to choose/calibrate the plan |
| Proxy | Planner-side estimate used before or instead of exhaustive oracle evaluation |
| Decision preservation | Agreement of compressed prediction with the FP32 prediction |
| Flip | A change in the binary prediction relative to FP32 |
| Margin | Absolute FP32 score distance from the binary boundary |
| Signed risk | Normalized loss of margin in the FP32 decision direction |
| Mean risk | Average sample-wise risk |
| p95 risk | 95th percentile of sample-wise risk |
| Interaction | Difference between a multi-action outcome and the sum/combination of isolated-action effects |

---

## Revision note

This document intentionally favors mathematical correctness and explicit uncertainty over apparent completeness. When the code or experiment logs reveal an exact formula that differs from a working notation here, update the corresponding section with:

1. the code path,
2. the old and new formulas,
3. the affected result files,
4. whether previous conclusions change.
