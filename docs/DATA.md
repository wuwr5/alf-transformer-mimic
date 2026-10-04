# Data

This repository contains **no patient data**. Everything under `data/` is
excluded by `.gitignore` on purpose.

## 1. Obtain MIMIC-IV

MIMIC-IV is distributed by the MIT Laboratory for Computational Physiology.

1. Create a PhysioNet account and complete the required CITI training course.
2. Request access to [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/).
3. Sign the data use agreement and download the `hosp`, `icu` and `note` modules.

Access for the analysis reported in this repository was granted under PhysioNet
credentialed access, **record ID 12801113**.

The data use agreement forbids redistribution, so derived tables must not be
committed either.

## 2. Expected layout

```
<MIMIC_ROOT>/
├── hosp/
│   ├── admissions.csv.gz
│   ├── patients.csv.gz
│   ├── diagnoses_icd.csv.gz
│   ├── procedures_icd.csv.gz
│   ├── labevents.csv.gz
│   ├── prescriptions.csv.gz
│   └── d_labitems.csv.gz
├── icu/
│   ├── icustays.csv.gz
│   ├── chartevents.csv.gz
│   ├── inputevents.csv.gz
│   ├── outputevents.csv.gz
│   ├── procedureevents.csv.gz
│   └── d_items.csv.gz
└── note/
    └── discharge.csv.gz
```

## 3. Derived concept tables (`mimic-code`)

`scripts/01_build_cohort.py` reads the raw CSVs directly with DuckDB, but it
does **not** recompute the derived concepts. It expects the `mimiciv_derived`
schema to already exist in the DuckDB file passed to `--db`.

Required concepts:

`icustay_detail`, `first_day_lab`, `first_day_vitalsign`, `first_day_sofa`,
`first_day_rrt`, `meld`, `enzyme`, `coagulation`, `chemistry`,
`complete_blood_count`.

```bash
git clone https://github.com/MIT-LCP/mimic-code.git
# then run the concept SQL in mimic-code/mimic-iv/concepts_duckdb against the
# database you will pass to --db; 01_build_cohort.py names any concept it cannot find
```

`duckdb` is not installed by `environment.yml` alone — install it with
`pip install -e ".[pipeline]"`.

## 4. Cohort definition

Patients are selected by ICD-9 and ICD-10 codes for liver failure:

| Version | Codes |
|---|---|
| ICD-9 | `570`, `5722`, `5724` |
| ICD-10 | `K704`, `K7040`, `K7041`, `K72`, `K720`, `K7200`, `K7201`, `K721`, `K7210`, `K7211`, `K729`, `K7290`, `K7291`, `K762`, `K767`, `K9182`, `K9183` |

This is a deliberately broad definition: it mixes acute liver failure,
acute-on-chronic liver failure and unspecified hepatic failure. Applying the
strict definition of acute liver failure requires a West Haven encephalopathy
grade, which MIMIC-IV does not record, so the broad definition is used and the
limitation is stated explicitly.

Exclusions, applied in order:

1. age below 18 years at first admission (none in practice — MIMIC-IV is an
   adult database);
2. for patients with repeated admissions for liver failure, only the first
   admission is retained;
3. ICU stays shorter than 24 hours.

## 5. Derived tables produced by `scripts/01_build_cohort.py`

### `cohort.csv`

One row per ICU stay, containing demographics, aetiology flags, complication
flags, first-day laboratory values and vital signs, severity scores, and the
outcome.

### `trajectory_panel.csv`

Wide daily panel keyed by `stay_id` with columns `<var>_d0` … `<var>_d6` for
`bilirubin`, `inr`, `creatinine` and `platelet`. Each cell is the most abnormal
value of that laboratory on that day (maximum for bilirubin, INR and creatinine;
minimum for platelets). Days without a measurement are left empty and are
represented to the model through missingness-indicator tokens.

Coverage in the development cohort: 2,330 of 2,508 ICU stays (92.9%) had at
least one trajectory value.

## 6. Missing data

`scripts/02_impute.R` performs multiple imputation by chained equations with
predictive mean matching (`mice`, `m = 20`, `maxit = 20`, `method = "pmm"`).
Highly right-skewed variables (creatinine, ALT, AST, total bilirubin, lactate)
are imputed on the natural-log scale and back-transformed.

Two variables are missing **by design** and are not imputed: days from ICU
admission to death and days from hospital admission to death are undefined for
patients who survive.

`MELD` is excluded from the imputation model — see [`METHODS.md`](METHODS.md).
