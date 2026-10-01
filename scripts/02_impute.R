# =============================================================================
# Step 2 - multiple imputation by chained equations (MICE-PMM)
#
#   Rscript scripts/02_impute.R
#
# Input : data/cohort_raw.csv      (from scripts/01_build_cohort.py)
# Output: data/cohort.csv          (mean of the m imputations, for descriptives)
#         data/imputed_datasets/   (m completed datasets, for analysis)
#         results/imputation_quality_check.csv
#
# Key modelling decisions (see docs/METHODS.md):
#   * highly skewed variables are imputed on the natural-log scale;
#   * MELD is EXCLUDED as a predictor because MIMIC-IV derives it from
#     bilirubin / INR / creatinine with missing components substituted by 1,
#     which would bias the imputed values downward;
#   * a monitoring-intensity auxiliary variable is included instead;
#   * days-to-death variables are missing by design and are not imputed.
# =============================================================================
suppressPackageStartupMessages(library(mice))

IN       <- "data/cohort_raw.csv"
OUT_DIR  <- "data"
OUT_MAIN <- file.path(OUT_DIR, "cohort.csv")
M        <- 20
MAXIT    <- 20
SEED     <- 42

dir.create(file.path(OUT_DIR, "imputed_datasets"), showWarnings = FALSE, recursive = TRUE)
dir.create("results", showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "derived"), showWarnings = FALSE, recursive = TRUE)

df <- read.csv(IN, check.names = FALSE)
cat("rows:", nrow(df), " cols:", ncol(df), "\n")

targets <- c("Hemoglobin", "Platelet", "WBC", "Albumin", "BUN", "Chloride",
             "Creatinine", "Sodium", "ALT", "AST", "Total_bilirubin", "PT",
             "APTT", "INR", "Lactate", "SpO2", "MAP")
targets <- intersect(targets, names(df))
LOGVARS <- intersect(c("Creatinine", "ALT", "AST", "Total_bilirubin", "Lactate"), targets)

cat("\n=== missing before imputation ===\n")
for (v in targets) {
  n <- sum(is.na(df[[v]]))
  cat(sprintf("  %-18s %6d  %5.2f%%\n", v, n, 100 * n / nrow(df)))
}

df$n_labs_observed <- rowSums(!is.na(df[, targets, drop = FALSE]))

preds <- c(targets,
           "Age", "SOFA", "los_icu", "outcome",
           "Ascites", "Sepsis", "HE", "HRS", "EVB", "SBP", "Shock", "Pneumonia",
           "rrt", "Liver_transplantation", "Vasopressin",
           "Alcoholic_only", "Alcoholic_plus_viral", "Viral_only", "gender",
           "n_labs_observed")
preds <- intersect(preds, names(df))

dat <- df[, preds, drop = FALSE]
if ("gender" %in% names(dat)) dat$gender <- as.factor(dat$gender)
for (v in LOGVARS) dat[[v]] <- log(dat[[v]])

meth <- make.method(dat); meth[] <- ""; meth[targets] <- "pmm"
pred <- make.predictorMatrix(dat)

cat("\n=== MICE: m =", M, ", maxit =", MAXIT, ", method = pmm ===\n")
t0 <- Sys.time()
imp <- mice(dat, m = M, maxit = MAXIT, method = meth,
            predictorMatrix = pred, seed = SEED, printFlag = FALSE)
cat("elapsed:", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), "min\n")
saveRDS(imp, file.path(OUT_DIR, "derived", "mice_object.rds"))

restore <- function(d) { for (v in LOGVARS) d[[v]] <- exp(d[[v]]); d }

# ---- export the m completed datasets ------------------------------------
for (i in seq_len(M)) {
  di <- df
  ci <- restore(complete(imp, i))
  for (v in targets) di[[v]] <- ci[[v]]
  write.csv(di, file.path(OUT_DIR, "imputed_datasets", sprintf("imputed_%02d.csv", i)),
            row.names = FALSE)
}
cat("wrote", M, "imputed datasets\n")

# ---- mean-of-imputations dataset (descriptive use only) ------------------
mean_df <- df
for (v in targets) {
  mat <- sapply(seq_len(M), function(i) restore(complete(imp, i))[[v]])
  mean_df[[v]] <- rowMeans(mat)
}
write.csv(mean_df, OUT_MAIN, row.names = FALSE)
cat("wrote", OUT_MAIN, "\n")

# ---- quality check: observed vs imputed at the imputed positions ---------
qc <- data.frame()
for (v in targets) {
  mi <- which(is.na(df[[v]]))
  if (length(mi) == 0) next
  impv <- unlist(lapply(seq_len(M), function(i) restore(complete(imp, i))[[v]][mi]))
  qc <- rbind(qc, data.frame(
    variable = v, n_missing = length(mi),
    obs_median = round(median(df[[v]][!is.na(df[[v]])]), 2),
    imp_median = round(median(impv), 2),
    obs_sd = round(sd(df[[v]][!is.na(df[[v]])]), 3),
    imp_sd = round(sd(impv), 3)))
}
write.csv(qc, "results/imputation_quality_check.csv", row.names = FALSE)
print(qc)

note <- paste(
  "IMPORTANT: cohort.csv is the mean of", M, "imputations and is intended for",
  "descriptive summaries only. For any statistical analysis use the datasets in",
  "data/imputed_datasets/ and pool estimates with Rubin's rules, e.g.:\n",
  "  imp <- readRDS('data/derived/mice_object.rds')\n",
  "  fit <- with(imp, glm(outcome ~ MELD + SOFA, family = binomial))\n",
  "  summary(pool(fit))")
cat("\n", note, "\n")
