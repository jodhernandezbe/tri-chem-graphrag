# Decision Framework: Hypotheses, Error Types & Metric Trade-offs

Companion to [`RESULTS.md`](RESULTS.md) §7 (End-to-End RAG Quality). It frames the evaluation
metrics — context precision, context recall, and faithfulness — in the language of hypothesis
testing and environmental decision-making, so a reader can map a score to a real-world
consequence and choose the embedding/threshold that matches their priorities.

The system recommends **source-reduction activities** for a query chemical, drawn from
structurally/operationally similar chemicals in the knowledge graph. Each recommendation is, in
effect, a hypothesis: *would this activity actually reduce this chemical's output?*

---

## 1 · The hypotheses

Let μ₁ be a facility's chemical output *with* a source-reduction activity and μ₀ the output
*without* it.

- **Null hypothesis ($H_0$):** the activity has no effect (or a negative effect) — chemical
  output stays the same or increases.
  $$H_0:\ \mu_1 \geq \mu_0$$
- **Alternative / research hypothesis ($H_a$):** the activity is effective — chemical output
  significantly decreases.
  $$H_a:\ \mu_1 < \mu_0$$

---

## 2 · The implications of errors

In environmental and industrial safety the stakes for the two errors differ significantly.

| Error type | What it means | Real-world implication |
|---|---|---|
| **False Positive (Type I)** | You conclude the activity is effective when it actually is not | You invest time, money, and resources into a "solution" that yields no environmental benefit |
| **False Negative (Type II)** | You conclude the activity is ineffective when it actually is effective | You discard a beneficial strategy, missing an opportunity to reduce pollution or operating costs |

---

## 3 · Balancing the trade-off

Which error to minimize depends on priorities:

- **Prioritize minimizing Type I errors (strictness):** if implementing a source-reduction
  activity requires expensive infrastructure changes, you want to be very sure it works before
  committing — set a strict threshold to avoid "false successes."
- **Prioritize minimizing Type II errors (sensitivity):** if the chemical is highly toxic or
  regulated, you may want to capture every possible reduction method — accepting a higher risk of
  false positives just to ensure you do not miss a legitimate way to reduce chemical output.

---

## 4 · Precision and recall

- **Precision (Positive Predictive Value):** accuracy among instances predicted positive —
  *"of all the cases labeled positive, how many were actually positive?"*
- **Recall (Sensitivity / True Positive Rate):** ability to find all positives —
  *"of all the actual positive cases, how many did the model correctly identify?"*

| Metric | Formula | Relationship to error |
|---|---|---|
| Precision | $\frac{TP}{TP + FP}$ | Penalized by **False Positives** (Type I errors) |
| Recall | $\frac{TP}{TP + FN}$ | Penalized by **False Negatives** (Type II errors) |

---

## 5 · The relationship with hypothesis testing

- **Type I error (False Positive):** rejecting $H_0$ when it is actually true → **lowers precision**.
- **Type II error (False Negative):** failing to reject $H_0$ when $H_a$ is true → **lowers recall**.

The tension between precision and recall is a **decision-threshold** trade-off: a strict
threshold to avoid false positives (high precision) tends to miss true cases (low recall), and
vice-versa.

---

## 6 · Mapping to this system, and the hallucination guardrail

Here the "positives" are recommended source-reduction activities: **context precision** ↔
precision (Type I / false positives), **context recall / coverage** ↔ recall (Type II / false
negatives). Because a missed reduction (Type II) means continued, avoidable, often irreversible
emissions, the screening question *"is there a useful source reduction for this chemical?"*
prioritizes **recall** → **Hybrid** (best recall/coverage in `RESULTS.md` §3/§6/§7) is the right
embedding. Reserve **Morgan's** higher precision for fully-automated, high-cost-of-action
settings where false positives are the dominant cost.

In the final RAGAS run through the shipped pipeline (manuscript Table 8; eight QA pairs per condition, so
differences of a few hundredths are noise), Hybrid has the highest context recall (0.43, against 0.40 for
Morgan and 0.15 for MolDeBERTa), Morgan the higher context precision (0.31, against 0.25 for Hybrid), and
MolDeBERTa the highest faithfulness (0.61, against 0.60 for Morgan and 0.46 for Hybrid).

**Hallucination guardrail (faithfulness).** Precision and recall both *assume the recommended
activity is real*. The third RAGAS metric, **faithfulness**, guards a different failure mode: the
generator *fabricating* a reduction not supported by the retrieved graph evidence — a hallucinated
positive that no confusion-matrix cell captures because it never existed in the data. High
faithfulness means every recommendation is traceable to an actual reported reduction in the
knowledge graph, so the precision/recall trade-off operates over *grounded* candidates rather than
invented ones. Deployment enforces this two ways: the empty-records guard returns a deterministic
"no records" message instead of inventing an answer, and faithfulness scoring flags any condition
whose answers drift from the evidence.

The shipped fused retrieval adds a traceability layer on top of this. Every record carries an evidence tier
(from same chemical with the stated use, tier 1, down to other practice, tier 6), and a fixed notice says
when no record matches the stated use.
A reader can therefore see which recommendation rests on the same chemical and use, and which rests on
analogues or general practice. The notices are fixed text, not generated by a language model.

**In summary —** faithfulness keeps recommendations *grounded*, recall keeps them *complete*
(few Type II misses), and precision keeps them *relevant* (few Type I false alarms).
