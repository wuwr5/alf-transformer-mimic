# Methods

## Cohort and outcome

Retrospective cohort of patients admitted to the intensive care unit with a
diagnosis code for liver failure in MIMIC-IV v2.2. The outcome is in-hospital
mortality (`admissions.hospital_expire_flag`). Follow-up starts at ICU
admission.

## Missing data

Multiple imputation by chained equations (MICE) with predictive mean matching,
20 imputed datasets, 20 iterations (`mice` in R 4.4.x). Log-scale imputation for
skewed variables. Two issues required specific handling.

### Derived scores must not predict their own components

The MIMIC-IV MELD implementation derives the score from bilirubin, INR and
creatinine, substituting 1 for any missing component. Using MELD as a predictor
while imputing bilirubin, INR and creatinine therefore propagates that
substitution. Empirically, including MELD biased the imputed bilirubin downward:
median 1.59 mg/dL against an observed 6.49 mg/dL. Excluding MELD from the
imputation model moved the imputed median to 4.28 mg/dL and preserved the
variance of INR (imputed SD 1.04 against an observed 1.82).

MELD is consequently excluded from the imputation model. A monitoring-intensity
auxiliary variable (the number of laboratory tests observed) is included
instead, which partially absorbs the informative missingness.

### Missingness is informative

Laboratory tests were ordered at the discretion of the treating clinician.
Patients with unmeasured bilirubin had a median MELD of 16.3 versus 26.8 in
those measured, and a mean SOFA of 5.9 versus 9.6. Imputation was therefore
performed under a missing-at-random assumption conditional on the full variable
set including the outcome, and the assumption is reported as a limitation.

Calibration of the imputation was assessed by comparing observed and imputed
medians within MELD quintiles. Key liver-specific variables calibrated well
(INR 1.40 vs 1.40, prothrombin time 14.9 vs 15.1 s, albumin within 0.1 g/dL).
Residual underestimation remained for transaminases and lactate.

## Models

### FT-Transformer

Each numerical feature is projected with its own weight vector, each binary
feature with its own bias vector, and a learnable `[CLS]` token is prepended.
The sequence is passed through `n_layers` Transformer encoder layers with
pre-norm residual connections, and the head reads the encoded `[CLS]` state.

### Hybrid Transformer

The FT-Transformer extended with one token per follow-up day. Each day token
concatenates the standardised 4-dimensional laboratory vector with a
4-dimensional missingness indicator, projected to `d_model`, plus a learnable
day-position embedding. Missing values are median-filled *only to provide a
numeric placeholder*; the model always sees the mask, so no information is
fabricated.

### Training

5-fold stratified cross-validation. Within each training fold a further 15% is
held out as an **inner** validation split used for early stopping. The outer
validation fold is never used for any modelling decision.

**This is not a cosmetic detail.** Selecting the best epoch on the outer fold —
a common shortcut — inflated the reported AUC from 0.779 to 0.851 for the
FT-Transformer and from 0.783 to 0.848 for the Hybrid Transformer, i.e. by
0.06–0.07 AUC.

Optimiser: AdamW, learning rate 1e-3, weight decay 1e-4, cosine annealing,
batch size 128, gradient norm clipped at 5.0, early stopping with patience 12 up
to 60 epochs. Predictions are averaged over three random seeds, which acts as a
cheap ensemble; the per-seed predictions are retained for the reproducibility
analysis.

## Statistical evaluation

All analyses are applied identically to every model.

| Aspect | Method |
|---|---|
| Rank association | Spearman correlation between prediction and binary outcome |
| Non-linearity | Logistic regression with a restricted cubic spline (Harrell basis, knots at the 10th, 50th and 90th percentiles), likelihood-ratio test against a linear term |
| Discrimination | ROC AUC with 1000-replicate bootstrap CI; all-pairs DeLong tests (Sun & Xu fast algorithm) |
| Calibration | Hosmer-Lemeshow test on deciles, plus a decile calibration plot |
| Clinical utility | Decision curve analysis, net benefit over thresholds 0.01–0.60 |
| Reproducibility | ICC(2,1) and ICC(1,1) over repeated fits, with a one-way repeated-measures ANOVA |

The repeated-measures ANOVA is computed in closed form rather than with
`statsmodels.AnovaRM`, which builds a dense `(n·k) × (n·k)` design matrix and
exhausts memory for cohorts of a few thousand subjects (≈19 GB at n = 2,508 and
k = 20).

Significance was set at two-sided *P* < 0.05. No adjustment for multiplicity was
applied to the calibration and discrimination comparisons, which are reported as
a pre-specified panel rather than as independent hypothesis tests.

## Reproducibility

Random seeds are fixed throughout (seeds 42, 7 and 2024 for the three model
runs; `random_state=42` for cross-validation splits, bootstrap resampling and
the imputation). Training is deterministic on CPU. Total runtime for both
architectures, five folds and three seeds is approximately 25 minutes on 12 CPU
cores.
