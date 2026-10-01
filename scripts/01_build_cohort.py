#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 1 - build the analysis cohort and the daily trajectory panel from MIMIC-IV.

Reads the gzipped MIMIC-IV CSVs directly with DuckDB, so no database import is
required. Concept tables (SOFA, MELD, first-day laboratory summaries) are taken
from the official `mimic-code` DuckDB concepts where available.

Example
-------
    python scripts/01_build_cohort.py \
        --mimic-root /path/to/mimic-iv-2.2 \
        --concepts /path/to/mimic-code/mimic-iv/concepts_duckdb \
        --out data/
"""
from __future__ import annotations

import argparse
import os
import sys

import duckdb

# Broad liver-failure code set. See docs/DATA.md for the rationale.
COHORT_CODES = [
    "570", "5722", "5724",
    "K704", "K7040", "K7041",
    "K72", "K720", "K7200", "K7201",
    "K721", "K7210", "K7211",
    "K729", "K7290", "K7291",
    "K762", "K767", "K9182", "K9183",
]

TRAJ_VARS = {
    "bilirubin": ("mimiciv_derived.enzyme", "bilirubin_total", "exp"),
    "inr": ("mimiciv_derived.coagulation", "inr", "exp"),
    "creatinine": ("mimiciv_derived.chemistry", "creatinine", "exp"),
    "platelet": ("mimiciv_derived.complete_blood_count", "platelet", "raw"),
}

TABLE_FILES = {
    "mimiciv_hosp.admissions": "hosp/admissions.csv.gz",
    "mimiciv_hosp.patients": "hosp/patients.csv.gz",
    "mimiciv_hosp.diagnoses_icd": "hosp/diagnoses_icd.csv.gz",
    "mimiciv_hosp.procedures_icd": "hosp/procedures_icd.csv.gz",
    "mimiciv_hosp.labevents": "hosp/labevents.csv.gz",
    "mimiciv_icu.icustays": "icu/icustays.csv.gz",
    "mimiciv_icu.chartevents": "icu/chartevents.csv.gz",
    "mimiciv_icu.inputevents": "icu/inputevents.csv.gz",
    "mimiciv_icu.outputevents": "icu/outputevents.csv.gz",
    "mimiciv_icu.procedureevents": "icu/procedureevents.csv.gz",
    "mimiciv_icu.d_items": "icu/d_items.csv.gz",
}

# Concepts that must exist in the DuckDB database before the cohort is built.
REQUIRED_CONCEPTS = ["icustay_detail", "first_day_lab", "first_day_vitalsign",
                     "first_day_sofa", "first_day_rrt", "meld", "enzyme",
                     "coagulation", "chemistry", "complete_blood_count"]


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def import_tables(con: duckdb.DuckDBPyConnection, root: str, db: str) -> None:
    """Materialise the required MIMIC-IV tables into a DuckDB database.

    Uses ``read_csv`` rather than ``COPY ... FROM``: the latter raises an
    unterminated-quote error on the free-text ``comments`` column of
    ``labevents``.
    """
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    for table, rel in TABLE_FILES.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            log(f"missing {path} - skipping {table}")
            continue
        cnt = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        if cnt:
            log(f"{table}: already loaded ({cnt:,} rows)")
            continue
        log(f"loading {table} <- {rel}")
        con.execute(f"INSERT INTO {table} SELECT * FROM read_csv('{path}', header=true)")
        cnt = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        log(f"  {cnt:,} rows")


def build_cohort(con: duckdb.DuckDBPyConnection) -> None:
    codes = ",".join(f"'{c}'" for c in COHORT_CODES)
    con.execute(f"""
    CREATE OR REPLACE TABLE cohort AS
    WITH dx AS (
        SELECT DISTINCT subject_id, hadm_id
        FROM mimiciv_hosp.diagnoses_icd
        WHERE upper(trim(icd_code)) IN ({codes})
    ),
    icu AS (
        SELECT d.*,
               ROW_NUMBER() OVER (PARTITION BY d.subject_id
                                  ORDER BY d.admittime, d.hadm_id) AS hadm_seq
        FROM (SELECT DISTINCT subject_id, hadm_id, admittime, admission_age FROM (
                SELECT * FROM mimiciv_derived.icustay_detail
                WHERE hadm_id IN (SELECT hadm_id FROM dx))) d
    )
    SELECT i.*
    FROM mimiciv_derived.icustay_detail i
    JOIN dx ON i.hadm_id = dx.hadm_id
    """)
    # exclusions: age < 18, only the first liver-failure admission, ICU stay >= 24 h
    con.execute("""
    CREATE OR REPLACE TABLE cohort AS
    WITH ranked AS (
        SELECT subject_id, hadm_id,
               ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) AS seq
        FROM (SELECT DISTINCT subject_id, hadm_id, admittime FROM mimiciv_derived.icustay_detail
              WHERE hadm_id IN (SELECT hadm_id FROM cohort))
    )
    SELECT c.*, r.seq AS hadm_seq
    FROM cohort c JOIN ranked r USING (subject_id, hadm_id)
    WHERE r.seq = 1
      AND c.admission_age >= 18
      AND c.los_icu >= 1.0
    """)
    n = con.execute("SELECT count(*) FROM cohort").fetchone()[0]
    log(f"cohort: {n:,} ICU stays")


def build_trajectory(con: duckdb.DuckDBPyConnection, out: str, n_days: int = 7) -> None:
    """Daily worst-value panel for the trajectory variables."""
    selects = []
    for var, (tbl, col, kind) in TRAJ_VARS.items():
        agg = "MIN" if var == "platelet" else "MAX"
        selects.append(f"""
        {var}_src AS (
            SELECT icu.stay_id,
                   date_diff('day', CAST(icu.icu_intime AS DATE),
                             CAST(x.charttime AS DATE)) AS d,
                   {agg}(x.{col}) AS v
            FROM cohort icu
            JOIN {tbl} x
              ON x.hadm_id = icu.hadm_id
             AND x.charttime >= icu.icu_intime
             AND x.charttime <  icu.icu_intime + INTERVAL {n_days} DAY
            WHERE x.{col} IS NOT NULL AND x.{col} > 0
            GROUP BY 1, 2
            HAVING d >= 0 AND d < {n_days}
        )""")
    with_src = "WITH " + ",\n".join(selects)
    con.execute(f"""
    CREATE OR REPLACE TABLE trajectory_long AS
    {with_src}
    SELECT * FROM {list(TRAJ_VARS)[0]}_src
    """)
    # pivot to wide
    cols = []
    for var in TRAJ_VARS:
        for d in range(n_days):
            cols.append(f"MAX(CASE WHEN var='{var}' AND d={d} THEN v END) AS {var}_d{d}")
    union_parts = " UNION ALL ".join(
        f"SELECT stay_id, '{var}' AS var, d, v FROM {var}_src" for var in TRAJ_VARS)
    con.execute(f"""
    CREATE OR REPLACE TABLE trajectory_panel AS
    SELECT stay_id, {', '.join(cols)}
    FROM ({union_parts})
    GROUP BY stay_id
    """)
    n = con.execute("SELECT count(*) FROM trajectory_panel").fetchone()[0]
    log(f"trajectory panel: {n:,} stays x {len(TRAJ_VARS) * n_days} columns")
    con.execute(f"COPY trajectory_panel TO '{os.path.join(out, 'trajectory_panel.csv')}'"
                " (HEADER, DELIMITER ',')")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-root", required=True,
                    help="directory containing the hosp/ and icu/ folders")
    ap.add_argument("--db", default="mimic_alf.duckdb",
                    help="DuckDB file; must already contain the mimic-code concepts")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-days", type=int, default=7)
    ap.add_argument("--skip-import", action="store_true",
                    help="assume the raw tables are already loaded into --db")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    con = duckdb.connect(args.db)

    if not args.skip_import:
        import_tables(con, args.mimic_root, args.db)

    existing = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='mimiciv_derived'").fetchall()}
    missing = [c for c in REQUIRED_CONCEPTS if c not in existing]
    if missing:
        log(f"!! missing derived concepts: {missing}")
        log("   run mimic-code concepts_duckdb/duckdb.sql against this database first")
        return 1

    build_cohort(con)
    build_trajectory(con, args.out, args.n_days)
    con.execute(f"COPY cohort TO '{os.path.join(args.out, 'cohort_raw.csv')}'"
                " (HEADER, DELIMITER ',')")
    log("wrote cohort_raw.csv and trajectory_panel.csv")
    log("next: run scripts/02_impute.R, then scripts/03_train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
