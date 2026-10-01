# Results

Development cohort: 2,508 ICU stays, 963 in-hospital deaths (38.4%).
All predictions are out-of-fold from 5-fold stratified cross-validation,
averaged over three random seeds.

## Discrimination

| Model | AUC | 95% CI |
|---|---|---|
| MELD 3.0 | 0.6123 | 0.590–0.634 |
| SOFA | 0.6517 | 0.629–0.672 |
| FT-Transformer | 0.7899 | 0.773–0.807 |
| Hybrid Transformer | **0.7985** | 0.781–0.815 |

### All-pairs DeLong tests

| Comparison | ΔAUC | Z | P |
|---|---|---|---|
| Hybrid vs FT | +0.009 | −2.47 | 0.014 |
| FT vs MELD 3.0 | +0.178 | 14.57 | <0.001 |
| FT vs SOFA | +0.138 | 12.69 | <0.001 |
| Hybrid vs MELD 3.0 | +0.186 | 15.38 | <0.001 |
| Hybrid vs SOFA | +0.147 | 13.44 | <0.001 |
| SOFA vs MELD 3.0 | +0.039 | −3.52 | <0.001 |

### Per-seed stability

| Model | seed 42 | seed 7 | seed 2024 | 3-seed average |
|---|---|---|---|---|
| FT-Transformer | 0.7783 | 0.7776 | 0.7757 | 0.7899 |
| Hybrid Transformer | 0.7834 | 0.7873 | 0.7791 | 0.7985 |

Individual seeds land at 0.776–0.787. Averaging three seeds lifts the AUC to
0.790–0.799, which is ensemble averaging rather than a property of any single
model.

## Rank association

| Model | Spearman ρ | P |
|---|---|---|
| FT-Transformer | 0.488 | <0.001 |
| Hybrid Transformer | 0.503 | <0.001 |
| SOFA | 0.256 | <0.001 |
| MELD 3.0 | 0.189 | <0.001 |

## Non-linearity (restricted cubic splines)

| Model | LR χ² | df | P for non-linearity |
|---|---|---|---|
| FT-Transformer | 4.03 | 1 | 0.045 |
| Hybrid Transformer | 6.64 | 1 | 0.010 |
| SOFA | 7.49 | 1 | 0.006 |
| MELD 3.0 | 2.90 | 1 | 0.089 |

## Calibration

| Model | Hosmer-Lemeshow χ² | df | P |
|---|---|---|---|
| MELD 3.0 | 390.3 | 8 | <0.001 |
| SOFA | 278.2 | 8 | <0.001 |
| FT-Transformer | 38.4 | 8 | <0.001 |
| Hybrid Transformer | 23.9 | 8 | 0.002 |

Conventional scores are grossly miscalibrated in this population and
systematically overestimate risk. The Transformers are far better but still fail
the test; neither should be used as a probability without recalibration.

## Decision curve analysis

Net benefit at threshold probabilities of 0.2, 0.3 and 0.4:

| Model | NB@0.2 | NB@0.3 | NB@0.4 |
|---|---|---|---|
| MELD 3.0 | 0.2237 | 0.1278 | 0.0360 |
| SOFA | 0.2292 | 0.1345 | 0.0514 |
| FT-Transformer | 0.2612 | 0.2069 | 0.1643 |
| Hybrid Transformer | **0.2643** | **0.2137** | **0.1664** |

## Reproducibility

| Model | ICC(2,1) | ICC(1,1) | ANOVA F | P | mean SD across repeats |
|---|---|---|---|---|---|
| FT-Transformer (3 seeds) | 0.8343 | 0.8336 | 208.2 | <0.001 | 0.0982 |
| Hybrid Transformer (3 seeds) | 0.8290 | 0.8281 | 245.8 | <0.001 | 0.0987 |

For context, measured on the same cohort with the same framework:

| Source of variation | ICC |
|---|---|
| MELD 3.0 / MELD / SOFA across 20 imputed datasets | 0.994–1.000 |
| Forward-Wald logistic model across 30 half-sample refits | 0.933 |
| **Transformers across 3 random seeds** | **0.829–0.834** |

Deep models are the least reproducible component of the pipeline. Note also that
the ANOVA *P* values are highly significant despite ICC ≈ 0.83 — with n = 2,508
even small between-run differences reach significance, so the effect size
(mean SD across repeats ≈ 0.10) should be reported alongside the *P* value.

## Where the Transformers sit relative to classical models

| Model class | AUC |
|---|---|
| Logistic regression (35 features) | 0.7945 |
| Random forest | 0.7979 |
| SVM (RBF) | 0.7993 |
| Histogram gradient boosting | 0.7934 |
| FT-Transformer (3-seed average) | 0.7899 |
| Hybrid Transformer (3-seed average) | 0.7985 |
| Hybrid Transformer (single seed) | 0.776–0.787 |

Six classical model classes spanned an AUC range of only 0.0155, and every one of
them reached a training AUC above 0.99 — the models are capacity-saturated. A
learning curve over 400–2,000 training cases plateaued at ≈0.79 by n = 800.

The honest conclusion is that **the Transformer adds nothing over classical
machine learning on this dataset**. The value of the architecture is that it
consumes the longitudinal laboratory trajectory natively through the day-token
mechanism, without the manual feature engineering that a tabular model would
require — not that it predicts better.
