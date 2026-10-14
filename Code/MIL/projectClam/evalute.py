import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

from metrics.cindex import concordance_index


# ==========================================================
# Read existing metrics from prediction CSV
# ==========================================================

def read_existing_metrics(df):

    # ------------------------------------------------------
    # Cox loss
    # ------------------------------------------------------

    cox_value = str(
        df["Cox loss"].iloc[0]
    )

    # Handles values such as:
    # tensor(3.4335)

    cox_value = (
        cox_value
        .replace("tensor(", "")
        .replace(")", "")
    )

    cox_loss = float(
        cox_value
    )

    # ------------------------------------------------------
    # C-index
    # ------------------------------------------------------

    # IMPORTANT:
    # This is the C-index already calculated
    # during model testing.

    cindex = float(
        df["C-index"].iloc[0]
    )

    return cox_loss, cindex


# ==========================================================
# Bootstrap 95% CI for the EXISTING C-index
# ==========================================================

def bootstrap_cindex_ci(
    df,
    n_bootstrap=2000,
    seed=42
):

    rng = np.random.default_rng(
        seed
    )

    n = len(df)

    time = df[
        "survival_months"
    ].to_numpy(
        dtype=float
    )

    event = df[
        "event"
    ].to_numpy(
        dtype=float
    )

    risk = df[
        "risk_score"
    ].to_numpy(
        dtype=float
    )

    bootstrap_scores = []

    for i in range(
        n_bootstrap
    ):

        # --------------------------------------------------
        # Sample patients with replacement
        # --------------------------------------------------

        indices = rng.integers(
            0,
            n,
            size=n
        )

        t = time[indices]
        e = event[indices]
        r = risk[indices]

        # --------------------------------------------------
        # Need events in bootstrap sample
        # --------------------------------------------------

        if e.sum() == 0:
            continue

        # --------------------------------------------------
        # Calculate C-index on bootstrap sample
        #
        # This is ONLY for estimating the CI.
        #
        # It does NOT replace the official C-index
        # stored in the CSV.
        # --------------------------------------------------

        try:

            c = concordance_index(
                t.tolist(),
                e.tolist(),
                r.tolist()
            )

            if np.isfinite(c):

                bootstrap_scores.append(
                    c
                )

        except Exception:

            continue

    bootstrap_scores = np.asarray(
        bootstrap_scores
    )

    if len(bootstrap_scores) == 0:

        raise RuntimeError(
            "Could not generate valid "
            "bootstrap C-index estimates."
        )

    # ------------------------------------------------------
    # Percentile bootstrap 95% CI
    # ------------------------------------------------------

    lower = np.percentile(
        bootstrap_scores,
        2.5
    )

    upper = np.percentile(
        bootstrap_scores,
        97.5
    )

    return (
        lower,
        upper,
        bootstrap_scores
    )


# ==========================================================
# Kaplan-Meier analysis
# ==========================================================

def kaplan_meier_analysis(
    df,
    model_name,
    output_dir
):

    df = df.copy()

    # ------------------------------------------------------
    # Median risk cutoff
    # ------------------------------------------------------

    risk_cutoff = df[
        "risk_score"
    ].median()

    df["risk_group"] = np.where(
        df["risk_score"] >= risk_cutoff,
        "High risk",
        "Low risk"
    )

    high = (
        df["risk_group"]
        == "High risk"
    )

    low = (
        df["risk_group"]
        == "Low risk"
    )

    # ------------------------------------------------------
    # Kaplan-Meier
    # ------------------------------------------------------

    km_low = KaplanMeierFitter()

    km_high = KaplanMeierFitter()

    plt.figure(
        figsize=(8, 6)
    )

    # Low risk

    km_low.fit(
        durations=df.loc[
            low,
            "survival_months"
        ],

        event_observed=df.loc[
            low,
            "event"
        ],

        label="Low risk"
    )

    km_low.plot_survival_function(
        ci_show=True
    )

    # High risk

    km_high.fit(
        durations=df.loc[
            high,
            "survival_months"
        ],

        event_observed=df.loc[
            high,
            "event"
        ],

        label="High risk"
    )

    km_high.plot_survival_function(
        ci_show=True
    )

    plt.xlabel(
        "Survival time (months)"
    )

    plt.ylabel(
        "Overall survival probability"
    )

    plt.title(
        f"{model_name}\n"
        "Kaplan-Meier Survival Curves"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    # ------------------------------------------------------
    # Save figure
    # ------------------------------------------------------

    km_file = os.path.join(
        output_dir,
        "KM_curve.png"
    )

    plt.savefig(
        km_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return (
        risk_cutoff,
        high,
        low,
        km_file
    )


# ==========================================================
# Log-rank test
# ==========================================================

def calculate_logrank(
    df,
    high,
    low
):

    result = logrank_test(

        df.loc[
            high,
            "survival_months"
        ],

        df.loc[
            low,
            "survival_months"
        ],

        event_observed_A=df.loc[
            high,
            "event"
        ],

        event_observed_B=df.loc[
            low,
            "event"
        ]
    )

    return result.p_value


# ==========================================================
# Hazard ratio
# ==========================================================

def calculate_hazard_ratio(
    df,
    high
):

    cox_df = df[
        [
            "survival_months",
            "event"
        ]
    ].copy()

    # High-risk = 1
    # Low-risk  = 0

    cox_df["high_risk"] = (
        high.astype(int)
    )

    cph = CoxPHFitter()

    cph.fit(
        cox_df,
        duration_col="survival_months",
        event_col="event"
    )

    # ------------------------------------------------------
    # Hazard ratio
    # ------------------------------------------------------

    hr = cph.hazard_ratios_[
        "high_risk"
    ]

    # ------------------------------------------------------
    # 95% CI
    #
    # lifelines stores CI on log(HR) scale
    # ------------------------------------------------------

    ci = cph.confidence_intervals_.loc[
        "high_risk"
    ]

    hr_lower = np.exp(
        ci.iloc[0]
    )

    hr_upper = np.exp(
        ci.iloc[1]
    )

    return (
        hr,
        hr_lower,
        hr_upper
    )


# ==========================================================
# Main evaluation
# ==========================================================

def evaluate(
    csv_file,
    model_name,
    output_dir,
    n_bootstrap=2000,
    seed=42
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ======================================================
    # Load CSV
    # ======================================================

    df = pd.read_csv(
        csv_file
    )

    required_columns = [
        "survival_months",
        "event",
        "risk_score",
        "Cox loss",
        "C-index"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            f"{missing}"
        )

    # Remove rows with missing
    # survival information

    df = df.dropna(
        subset=[
            "survival_months",
            "event",
            "risk_score"
        ]
    ).copy()

    # ======================================================
    # Header
    # ======================================================

    print()
    print("=" * 70)

    print(
        model_name
    )

    print("=" * 70)

    print(
        "Prediction file:",
        csv_file
    )

    print(
        "Patients:",
        len(df)
    )

    print(
        "Events:",
        int(
            df["event"].sum()
        )
    )

    print(
        "Censored:",
        int(
            (df["event"] == 0).sum()
        )
    )

    # ======================================================
    # Existing Cox loss + C-index
    # ======================================================

    cox_loss, cindex = (
        read_existing_metrics(df)
    )

    print()
    print(
        "Existing test metrics"
    )

    print(
        f"Cox loss : {cox_loss:.4f}"
    )

    print(
        f"C-index  : {cindex:.4f}"
    )

    # ======================================================
    # Bootstrap 95% CI
    # ======================================================

    print()
    print(
        f"Bootstrap C-index CI "
        f"({n_bootstrap} iterations)..."
    )

    (
        ci_lower,
        ci_upper,
        bootstrap_scores
    ) = bootstrap_cindex_ci(
        df,
        n_bootstrap=n_bootstrap,
        seed=seed
    )

    print(
        f"C-index 95% CI: "
        f"{ci_lower:.4f} - "
        f"{ci_upper:.4f}"
    )

    # ======================================================
    # Kaplan-Meier
    # ======================================================

    (
        risk_cutoff,
        high,
        low,
        km_file
    ) = kaplan_meier_analysis(
        df,
        model_name,
        output_dir
    )

    print()
    print(
        f"Risk cutoff (median): "
        f"{risk_cutoff:.6f}"
    )

    print(
        "Low-risk patients:",
        int(low.sum())
    )

    print(
        "High-risk patients:",
        int(high.sum())
    )

    # ======================================================
    # Log-rank
    # ======================================================

    logrank_p = calculate_logrank(
        df,
        high,
        low
    )

    print()
    print(
        f"Log-rank p-value: "
        f"{logrank_p:.6g}"
    )

    # ======================================================
    # Hazard ratio
    # ======================================================

    (
        hr,
        hr_lower,
        hr_upper
    ) = calculate_hazard_ratio(
        df,
        high
    )

    print()
    print(
        f"Hazard ratio: "
        f"{hr:.4f}"
    )

    print(
        f"HR 95% CI: "
        f"{hr_lower:.4f} - "
        f"{hr_upper:.4f}"
    )

    # ======================================================
    # Save bootstrap distribution
    # ======================================================

    bootstrap_file = os.path.join(
        output_dir,
        "bootstrap_cindex.csv"
    )

    pd.DataFrame({
        "bootstrap_cindex":
        bootstrap_scores
    }).to_csv(
        bootstrap_file,
        index=False
    )

    # ======================================================
    # Save summary
    # ======================================================

    summary = pd.DataFrame([
        {
            "model": model_name,

            "n_patients": len(df),

            "n_events": int(
                df["event"].sum()
            ),

            "n_censored": int(
                (df["event"] == 0).sum()
            ),

            "cox_loss": cox_loss,

            # IMPORTANT:
            # This is the original C-index
            # already stored in the CSV.

            "c_index": cindex,

            "c_index_95_lower":
                ci_lower,

            "c_index_95_upper":
                ci_upper,

            "risk_cutoff":
                risk_cutoff,

            "n_low_risk":
                int(low.sum()),

            "n_high_risk":
                int(high.sum()),

            "logrank_p":
                logrank_p,

            "hazard_ratio":
                hr,

            "hr_95_lower":
                hr_lower,

            "hr_95_upper":
                hr_upper
        }
    ])

    summary_file = os.path.join(
        output_dir,
        "survival_evaluation.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # ======================================================
    # Final output
    # ======================================================

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Cox loss: "
        f"{cox_loss:.4f}"
    )

    print(
        f"C-index: "
        f"{cindex:.4f}"
    )

    print(
        f"C-index 95% CI: "
        f"{ci_lower:.4f} - "
        f"{ci_upper:.4f}"
    )

    print(
        f"Log-rank p: "
        f"{logrank_p:.6g}"
    )

    print(
        f"HR: "
        f"{hr:.4f} "
        f"({hr_lower:.4f} - "
        f"{hr_upper:.4f})"
    )

    print()
    print(
        "KM curve:"
    )

    print(
        km_file
    )

    print()
    print(
        "Summary:"
    )

    print(
        summary_file
    )

    print(
        "Bootstrap samples:"
    )

    print(
        bootstrap_file
    )

    print("=" * 70)

    return summary

# ==========================================================
# Configuration
# ==========================================================

CSV_FILE = (
    "D:/datasets/MIL/clam/test_predictions.csv"
)

MODEL_NAME = (
    "GigaPath + CLAM"
)

OUTPUT_DIR = (
    "D:/datasets/MIL/results/GigaPath_CLAM"
)

N_BOOTSTRAP = 2000

SEED = 42


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    evaluate(
        csv_file=CSV_FILE,
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED
    )