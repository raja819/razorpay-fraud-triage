
"""
Part A — Data Foundation for the Razorpay Fraud Triage capstone.

Responsibilities:
1. Audit the generated transaction data.
2. Remove exact duplicates and invalid transaction amounts.
3. Normalize merchant-category and city text.
4. Add three transaction-risk features.
5. Print reproducible audit checkpoints for review.

Important:
- Missing distance values are intentionally NOT imputed here.
  Part B must learn the imputation value from the training split only.
- chargeback_filed is retained in the cleaned dataset for auditing,
  but must be excluded from model features because it can leak
  information recorded after the transaction outcome.
"""

from pathlib import Path
import pandas as pd 
import numpy as np

# Keep the input path relative to the project root. Run this script
# from the root folder so the same command works after cloning the repo.

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT/"transactions_raw.csv"


def clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned copy of the raw transaction DataFrame.

    Cleaning rules:
    - Do not modify the caller's DataFrame.
    - Remove exact duplicate rows.
    - Standardize merchant_category and city text.
    - Remove rows with a missing amount.
    - Keep only amounts in the inclusive range [1, 1_000_000].
    - Preserve missing distance_from_home_km values for Part B.
    """
    required_columns= {
        "merchant_category",
        "city",
        "amount_inr"
    }
    missing_columns = required_columns.difference(raw_df.columns)

    if missing_columns:
        raise ValueError("Input data is missing required columns:"+",".join(sorted(missing_columns)))

    # Work on a copy so callers can safely reuse their original data.
    cleaned_df = raw_df.copy(deep=True)

    # Exact duplicate rows do not represent independent transactions.
    cleaned_df = cleaned_df.drop_duplicates()

    # Normalize text after removing duplicates. strip() removes extra
    # whitespace; title() standardizes capitalization.
    for column in ("merchant_category", "city"):
        cleaned_df[column] = (cleaned_df[column]).astype("string").str.strip().str.title()

    # A transaction without an amount cannot pass the amount checks.
    cleaned_df = cleaned_df.dropna(subset=["amount_inr"])
    
    # Use inclusive business limits: 1 and 1,000,000 are valid amounts.
    valid_amount = cleaned_df["amount_inr"].between(1, 1_000_000, inclusive="both")
    cleaned_df = cleaned_df.loc[valid_amount]

    # Do not fill distance_from_home_km here. Its median must be learned
    # from the training split in Part B to avoid data leakage.
    return cleaned_df.reset_index(drop=True)



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with three engineered transaction-risk features.

    amount_vs_avg_ratio: current amount divided by the 30-day average.
    is_night: 1 when hour_of_day is 0–5; otherwise 0.
    high_value: 1 when amount_inr is strictly greater than 50,000.
    """
    required_columns = {
        "amount_inr",
        "avg_amount_30d_inr",
        "hour_of_day",
    }
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError("Cannot enginner features; missing columns:"+",".join(sorted(missing_columns)))

    # Avoid modifying the DataFrame passed by the caller.
    featured_df = df.copy(deep=True)

    # A zero denominator cannot produce a meaningful ratio. Represent
    # that exceptional case as missing rather than creating infinity.
    average_amount = featured_df["avg_amount_30d_inr"].where(
        featured_df["avg_amount_30d_inr"] != 0
    )

    featured_df["amount_vs_avg_ratio"] = (
        featured_df["amount_inr"] / average_amount
    )

    featured_df["is_night"] = (
        featured_df["hour_of_day"].le(5).astype("int64")
    )

    featured_df["high_value"] = (
        featured_df["amount_inr"].gt(50_000).astype("int64")
    )

    return featured_df


def verify_checkpoint(
    checkpoint_name: str,
    actual_value,
    expected_value,
) -> None:
    """
    Print a readable PASS/REVIEW result for an expected audit checkpoint.

    REVIEW means the observed value differs from the assignment reference
    and should be investigated; it does not silently alter the data.
    """
    passed = actual_value == expected_value
    status = "PASS" if passed else "REVIEW"

    print(
        f"[{status} {checkpoint_name}]:"
        f"actual={actual_value!r}, expected={expected_value!r}"
    )


def audit_and_report(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Print Part A data-quality checkpoints and return the featured data.

    This function is for the assignment's audit/reporting workflow.
    The reusable cleaning and feature functions remain independently
    callable for later tests and for the Part B training pipeline.
    """

    print("\n" + "=" * 72)
    print("PART A - RAW DATA AUDIT")
    print("=" * 72)

    print(f"Raw shape: {raw_df.shape}")
    verify_checkpoint("Raw row count", len(raw_df), 6090)
    verify_checkpoint("Raw column count", raw_df.shape[1], 14)

    duplicate_count = int(raw_df.duplicated().sum())
    print(f"Exact duplicate rows: {duplicate_count}")
    verify_checkpoint("Exact duplicate rows", duplicate_count, 90)


    # Count raw spellings before normalization to expose inconsistent text.
    if "merchant_category" in raw_df.columns:
        category_spellings = int(raw_df["merchant_category"].nunique(dropna=True))
        print(f"Distinct raw merchant-category spelling: {category_spellings}")
        verify_checkpoint("Raw merchant-category spellings", category_spellings, 24)

    if "city" in raw_df.columns:
        city_spellings = int(raw_df["city"].nunique(dropna=True))
        print(f"Distinct raw city spellings: {city_spellings}")
        verify_checkpoint("Raw city spellings", city_spellings, 24)

    # Examine the de-duplicated data separately so the counts match the
    # order of the assignment's cleaning rules.
    deduplicated_df = raw_df.drop_duplicates()

    missing_amount_count = int(deduplicated_df["amount_inr"].isna().sum())
    print(f"Missing amount rows after de-duplication: {missing_amount_count}")
    verify_checkpoint("Missing amounts after de-duplication", missing_amount_count, 76)

    amount_values = deduplicated_df["amount_inr"].dropna()
    out_of_range_count = int(
        (~amount_values.between(1, 1_000_000, inclusive="both")).sum()
    )
    print(f"Out-of-range amount rows after de-dupliction:{out_of_range_count}")
    verify_checkpoint("Out-of-range amounts", out_of_range_count, 5)

    missing_distance_deduplicated = int(deduplicated_df["distance_from_home_km"].isna().sum())
    print(
        "Missing distance rows after de-duplication:"
        f"{missing_distance_deduplicated}"
    )
    verify_checkpoint("Missing distances after de-duplication", missing_distance_deduplicated, 193)

    # Apply the same public cleaning function that later pipeline steps use.
    cleaned_df = clean_transactions(raw_df)

    print("\n" + "=" * 72)
    print("CLEANING RESULTS")
    print("=" * 72)
    
    print(f"Cleaned shape: {cleaned_df.shape}")
    verify_checkpoint("Cleaned row count", len(cleaned_df), 5919)
    
    print("\nCleaned city counts:")
    print(cleaned_df["city"].value_counts().to_string())

    missing_distance_cleaned = int(
        cleaned_df["distance_from_home_km"].isna().sum()
    )
    print(f"Missing distance rows retained: {missing_distance_cleaned}")
    verify_checkpoint("Missing distances retained",missing_distance_cleaned,191)

    print("\nMerchant-category counts:")
    category_counts = cleaned_df["merchant_category"].value_counts()
    print(category_counts.to_string())

    expected_categories = {
        "Grocery": 1406,
        "Food Delivery": 1326,
        "Fashion": 976,
        "Electronics": 845,
        "Utilities": 685,
        "Travel": 681,
    }
    actual_categories = category_counts.to_dict()
    verify_checkpoint(
        "Merchant-category distribution",
        actual_categories,
        expected_categories,
    )

    print("\nDistinct cleaned cities:")
    cleaned_city_count = int(cleaned_df["city"].nunique(dropna=True))
    print(cleaned_city_count)
    verify_checkpoint("Distinct cleaned cities", cleaned_city_count, 6)

    print("\nPayment-mode counts:")
    payment_counts = cleaned_df["payment_mode"].value_counts()
    print(payment_counts.to_string())

    expected_payment_modes = {
        "UPI": 3221,
        "Card": 1450,
        "NetBanking": 731,
        "Wallet": 517,
    }

    # Compare using the actual capitalization found in the dataset, while
    # keeping the expected audit values explicit and easy to inspect.
    actual_payment_modes = {
        str(key): int(value)
        for key, value in payment_counts.items()
    }

    expected_payment_modes = {
        "UPI": 3221,
        "Card": 1450,
        "NetBanking": 731,
        "Wallet": 517,
    }

    verify_checkpoint(
        "Payment-mode distribution",
        actual_payment_modes,
        expected_payment_modes,
    )

    amount_min = round(float(cleaned_df["amount_inr"].min()), 2)
    amount_max = round(float(cleaned_df["amount_inr"].max()), 2)
    print(f"\nCleaned amount range: {amount_min} to {amount_max} INR")

    verify_checkpoint("Minimum cleaned amount", amount_min, 141.37)
    verify_checkpoint("Maximum cleaned amount", amount_max, 105883.95)


    # Check the target after cleaning; this is the label distribution,
    # not a feature to include in X.
    if "is_fraud" in cleaned_df.columns:
        fraud_counts = cleaned_df["is_fraud"].value_counts().sort_index()
        fraud_rate = float(cleaned_df["is_fraud"].mean())

        print("\nFraud-label counts:")
        print(fraud_counts.to_string())
        print(f"Fraud prevalence: {fraud_rate:.4%}")
        verify_checkpoint("Fraud rows", int(cleaned_df["is_fraud"].sum()), 366)

        majority_class_accuracy = float(cleaned_df["is_fraud"].value_counts(normalize=True).max())
        print("Never-fraud baseline accuaracy:" f"{majority_class_accuracy:.2%}")

    # Chargeback is retained for the audit, but MUST NOT be used as a model
    # feature: it can be recorded after a transaction has already occurred.
    if {"chargeback_filed", "is_fraud"}.issubset(cleaned_df.columns):
        print("\nChargeback-vs-fraud audit (investigate target leakage):")
        chargeback_table = pd.crosstab(
            cleaned_df["chargeback_filed"],
            cleaned_df["is_fraud"],
            margins=True,
        )
        print(chargeback_table.to_string())

    featured_df = add_features(cleaned_df)
    print("\n" + "=" * 72)
    print("FEATURE-ENGINEERING CHECKPOINTS")
    print("=" * 72)

    expected_features = {
        "amount_vs_avg_ratio",
        "is_night",
        "high_value"
    }
    missing_engineered_features = expected_features.difference(featured_df.columns)
    verify_checkpoint(
        "Engineered feature columns present",
        sorted(expected_features.intersection(featured_df.columns)),
        sorted(expected_features)
    )

    night_count = int(featured_df["is_night"].sum())
    high_value_count = int(featured_df["high_value"].sum())
    ratio_max_raw = float(
        featured_df["amount_vs_avg_ratio"].max()
    )

    max_ratio = round(ratio_max_raw, 2)

    print(f"Maximum amount-vs-average ratio (raw): {ratio_max_raw:.6f}")
    print(f"Maximum amount-vs-average ratio (rounded): {max_ratio:.2f}")

    verify_checkpoint(
        "Maximum amount_vs_avg_ratio",
        max_ratio,
        13.11
    )


    # Feature engineering should not change the row count or target labels.
    verify_checkpoint(
        "Feature engineering preserves row count",
        len(featured_df),
        len(cleaned_df)
    )

    ratio_values = featured_df["amount_vs_avg_ratio"].dropna()

    infinite_ratio_count = int(
        (~np.isfinite(ratio_values.to_numpy(dtype=float))).sum()
    )

    print(f"Infinite non-null ratios: {infinite_ratio_count}")
    verify_checkpoint(
        "Infinite non-null ratios",
        infinite_ratio_count,
        0
    )
    print("\n" + "=" * 72)
    print("PART A AUDIT COMPLETE")
    print("=" * 72)
    print(
        "Note: chargeback_filed, txn_id and customer_id may remain in the "
        "audit DataFrame, but exclude them from model inputs in Part B."
    )
    print(
        "Note: distance_from_home_km is intentionally not imputed in Part A; "
        "learn its median from the training split only."
    )

    return featured_df


def main() -> None:
    """Run the audit when this file is executed as a script."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_DATA_PATH.name} in {PROJECT_ROOT}."
            "Run this script from the project root or check the CSV location."
        )

    raw_df = pd.read_csv(RAW_DATA_PATH)

    # Keep a copy for a simple non-mutation check.
    original_snapshot = raw_df.copy(deep=True)
    featured_df = audit_and_report(raw_df)

    # The reusable functions must not mutate the original DataFrame.
    if not raw_df.equals(original_snapshot):
        raise AssertionError(
            "The raw DataFrame was modified during the Part a workflow."
        )

    print(f"\nFinal featured shape: {featured_df.shape}")
    print("Raw-input non-mutation check: PASS")


if __name__ == "__main__":
    main()