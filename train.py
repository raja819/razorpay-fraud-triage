from pathlib import Path
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from prepare import clean_transactions, add_features


# Keep file paths relative to this project, not to the terminal's current folder.
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "transactions_raw.csv"

RANDOM_STATE = 42
TARGET_COLUMN = "is_fraud"

# These identifiers and post-transaction outcomes must not be model inputs.
EXCLUDED_FEATURES = [
    "txn_id",
    "customer_id",
    "chargeback_filed",
    "is_fraud"
]

# Fix the model's input columns so training and API predictions use the same order.
# Seven original numeric/binary features + three engineered features + 16 dummy columns.
MODEL_COLUMNS = [
    "amount_inr",
    "hour_of_day",
    "txn_count_last_24h",
    "avg_amount_30d_inr",
    "distance_from_home_km",
    "is_new_device",
    "account_age_days",
    "amount_vs_avg_ratio",
    "is_night",
    "high_value",
    "merchant_category_Electronics",
    "merchant_category_Fashion",
    "merchant_category_Food Delivery",
    "merchant_category_Grocery",
    "merchant_category_Travel",
    "merchant_category_Utilities",
    "city_Bengaluru",
    "city_Chennai",
    "city_Delhi",
    "city_Kolkata",
    "city_Mumbai",
    "city_Pune",
    "payment_mode_Card",
    "payment_mode_NetBanking",
    "payment_mode_UPI",
    "payment_mode_Wallet",
]


def load_and_prepare_data():
    """Load raw transactions and prepare leakage-safe model inputs."""

    # Load the original CSV without modifying the source file.
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}\n"
            "Make sure transactions_raw.csv is in the project root."
        )

    raw_df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw dataset shape: {raw_df.shape}")

    # Reuse the cleaning and feature-engineering logic from Part A.
    cleaned_df = clean_transactions(raw_df)
    featured_df = add_features(cleaned_df)

    # Ensure each transaction row has a unique index for safe alignment.
    if not featured_df.index.is_unique:
        raise ValueError("Featured dataset contains duplicate row indices.") 

    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"featured dataset shape: {featured_df.shape}")

    # The target is what the model must predict -
    if TARGET_COLUMN not in featured_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    # Validate the original target before converting its dtype.
    raw_target = featured_df[TARGET_COLUMN]

    if raw_target.isna().any():
        raise ValueError("Fraud target contains missing values.")

    if not raw_target.isin([0, 1]).all():
        raise ValueError("The fraud target must contain only 0 and 1.")

    y = raw_target.astype(int)

    # Remove identifiers, the target, and the post-outcome chargeback field.
    X = featured_df.drop(columns=EXCLUDED_FEATURES, errors="raise")

    # Convert categorical columns to ont-hot encoded numeric columns.
    X = pd.get_dummies(
        X,
        columns=["merchant_category", "city", "payment_mode"],
        dtype=int
    )

    # Enforce the exact feature names and order expected by the model.
    missing_columns = sorted(set(MODEL_COLUMNS) - set(X.columns))
    unexpected_columns = sorted(set(X.columns) - set(MODEL_COLUMNS))

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Model feature columns do not match the expected schema.\n"
            f"Missing columns: {missing_columns}\n"
            f"Unexcepted columns: {unexpected_columns}"
        )

    X = X.reindex(columns=MODEL_COLUMNS)

    # Basic integrity checks before splitting the data.
    if len(X) != len(y):
        raise ValueError("Feature and target row counts do not match.")

    # Missing values are allowed only in distance_from_home_km.
    # That column will be imputed later using the training-set median.
    missing_by_column = X.isna().sum()
    distance_column = "distance_from_home_km"

    other_missing = missing_by_column.drop(
        labels=[distance_column],
        errors="ignore",
    )

    if (other_missing > 0).any():
        raise ValueError(
            "Unexpected missing values in model features:\n"
            f"{other_missing[other_missing > 0]}"
        )

    missing_distance = int(X[distance_column].isna().sum())

    print(f"Missing feature values: {int(missing_by_column.sum())}")
    print(f"Missing distance values: {missing_distance}")

    # Missing distance values are expected at this stage.
    # Infinite values are not allowed in any numeric feature.
    numeric_values = X.select_dtypes(
        include=np.number
    ).to_numpy(dtype=float)

    if np.isinf(numeric_values).any():
        raise ValueError(
            "Feature data contains infinite numeric values."
        )

    # Reject infinite values and unexpected missing values.
    distance_column = "distance_from_home_km"

    if np.isinf(numeric_values).any():
        raise ValueError("Feature data contains infinite numeric values.")

    if X.drop(columns=[distance_column]).isna().any().any():
        raise ValueError("Unexpected missing values found outside distance_from_home_km.")
    
    print(f"Model feature count: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    print(f"Target fraud rate: {y.mean():.4%}")


    
    return X, y, featured_df

    
def split_and_impute(X, y):
    """Create stratified train/validation/test splits and impute safely."""

    # First split: 60% training and 40% temporary data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y
        
    )

    # Second split: divide the temporary data equally into validation and test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp

    )

    # Learn the imputation value from training data only.
    distance_column = "distance_from_home_km"
    training_median = X_train[distance_column].median()

    if pd.isna(training_median):
        raise ValueError(f"Cannot calculate a training medain for {distance_column}")


    # Apply the same training -derived median to all three splits.
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[distance_column] = X_train[distance_column].fillna(training_median)
    X_val[distance_column] = X_val[distance_column].fillna(training_median)
    X_test[distance_column] = X_test[distance_column].fillna(training_median)

    # Confirm that imputation removed missing values from all splits.
    for split_name, split_X in [
        ("Training", X_train),
        ("Validation", X_val),
        ("Test", X_test)
    ]:
        missing_count = int(split_X.isna().sum().sum())
        print(f"{split_name} missing feature values: {missing_count}")

        if missing_count:
            raise ValueError(f"{split_name} data still contains missing feature values.")


    # Display split size and target distributions for verification.
    print(f"\nTraining shape: {X_train.shape}")
    print(f"Validtion shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Training median for {distance_column}: {training_median}")

    print("\nFraud counts by split:")
    print(f"Training:\n{y_train.value_counts().sort_index()}")
    print(f"Validation:\n{y_val.value_counts().sort_index()}")
    print(f"Test:\n{y_test.value_counts().sort_index()}")

    return(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        training_median
    )


def evaluate_binary_classifier(model, X_eval, y_eval, threshold=0.5):
    """Calculate classification metrics at a specified decision threshold."""

    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_eval, predictions),
        "precision": precision_score(y_eval, predictions, zero_division=0),
        "recall": recall_score(y_eval, predictions, zero_division=0),
        "f1": f1_score(y_eval, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_eval, probabilities)
    }


def oversample_training_data(X_train, y_train):
    """Randomly oversample the minority class using training rows only."""

    if not X_train.index.equals(y_train.index):
        raise ValueError(
            "X_train and y_train indices do not match before oversampling."
        )
    
    train_df = X_train.copy()
    train_df["_target"] = y_train.to_numpy()

    class_counts = train_df["_target"].value_counts()
    if len(class_counts) != 2:
        raise ValueError("Oversampling requires both target classes.")

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    target_count = int(class_counts.max())

    majority_rows = train_df[train_df["_target"] == majority_class]
    minority_rows = train_df[train_df["_target"] == minority_class]

    # Sample minority rows with replacement untill both classes are balanced.
    extra_minority = minority_rows.sample(
        n=target_count - len(minority_rows),
        replace=True,
        random_state=RANDOM_STATE
    )

    balanced_df = pd.concat(
        [majority_rows, minority_rows, extra_minority],
        ignore_index=True
    )

    # Shuffle the rows reproducibly after combining both classes.
    balanced_df = balanced_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    X_balanced = balanced_df.drop(columns="_target")
    y_balanced = balanced_df["_target"].astype(int)

    return X_balanced, y_balanced


def run_logistic_regression_experiments(X_train, y_train, X_val, y_val):
    """Compare plain, class-weighted, and oversampled Logistic Regression."""

    experiments = {}

    # Experiment 1 : Ordinary Logistic Regression.
    experiments["Logistic Regression - Plain"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])


    # Experiment 2 : give the minority fraud class greater training weight.
    experiments["Logistic Regression - Balanced"] = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ) 
        )
    ])


    # Experiment 3 : train on a balanced resampled version of training data,
    X_over, y_over = oversample_training_data(X_train, y_train)

    experiments["Logistic Regression - Oversampled"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    results = []

    for name, model in experiments.items():
        if name == "Logistic Regression - Oversampled":
            model.fit(X_over, y_over)
            print(
                "\nOversampled training class counts:\n"
                f"{y_over.value_counts().sort_index()}"
            )
        else:
            model.fit(X_train, y_train)

        metrics = evaluate_binary_classifier(
            model, X_val, y_val, threshold=0.5
        )

        results.append({"experiment": name, **metrics})

    results_df = pd.DataFrame(results).set_index("experiment")

    print("\nValidation comparison at threshold = 0.5:")
    print(results_df.round(3).to_string())

    return experiments, results_df 


def run_cross_validation_and_tuning(X_train, y_train):
    """Compare classifiers using training-only five-fold CV and grid search."""

    # Keep every CV fold within the training split.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Baseline : class-weighted Logistic Regression.
    balanced_lr = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ])

    lr_scores = cross_val_score(
        balanced_lr,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=1
    )

    print("\nBalanced Logistic Regression - 5-fold CV")
    print(f"Fold F1 scores: {np.round(lr_scores, 4).tolist()}")
    print(f"Mean F1: {lr_scores.mean():.4f}")
    print(f"Standard deviation: {lr_scores.std():.4f}")

    # Untuned Random Forest baseline.
    baseline_rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    rf_scores = cross_val_score(
        baseline_rf,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=1
    )

    print("\nBaseline Random Forest — 5-fold CV")
    print(f"Fold F1 scores: {np.round(rf_scores, 4).tolist()}")
    print(f"Mean F1: {rf_scores.mean():.4f}")
    print(f"Standard deviation: {rf_scores.std():.4f}")


    # Grid search: 4 depth choices x 3 leaf-size choice = 12 combinations.
    param_grid = {
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [5, 10, 20]
    }

    tuned_rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    grid_search = GridSearchCV(
        estimator=tuned_rf,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
        error_score="raise"
    )

    grid_search.fit(X_train, y_train)

    print("\nRandom Forest Grid Search")
    print(f"Combinations tested: {len(grid_search.cv_results_['params'])}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best mean CV F1: {grid_search.best_score_:.4f}")

    
    return {
        "logistic_cv_scores": lr_scores,
        "baseline_rf_cv_scores": rf_scores,
        "grid_search": grid_search,
        "best_model": grid_search.best_estimator_
    }


def tune_decision_threshold(model, X_val, y_val, minimum_recall=0.65):
    """Choose the highest threshold that achieves the required validation recall."""

    # Use validation probabilities to evaluate candidate thresholds.
    probabilities = model.predict_proba(X_val)[:, 1]
    threshold_rows = []

    for threshold in np.round(np.arange(0.1, 1.0, 0.1), 1):
        predictions = (probabilities >= threshold).astype(int)

        threshold_rows.append({
            "threshold": float(threshold),
            "precision": precision_score(y_val, predictions, zero_division=0),
            "recall": recall_score(y_val, predictions, zero_division=0),
            "f1": f1_score(y_val, predictions, zero_division=0),
            "flagged_rows": int(predictions.sum())
        })

    threshold_df = pd.DataFrame(threshold_rows)

    print("\nValidtion threshold comparison:")
    print(threshold_df.round(3).to_string(index=False))

    # Keep only thresholds that meet the required recall.
    eligible = threshold_df[
        threshold_df["recall"] >= minimum_recall
    ]

    if eligible.empty:
        raise ValueError(
            f"No threshold achieved the required recall of {minimum_recall:.0%}."
        )


    # Among eligible thresholds, choose the highest one to reduce unnecessary alerts.
    selected_row = eligible.sort_values(
        "threshold", ascending=False
    ).iloc[0]

    selected_threshold = float(selected_row["threshold"])

    print(f"\nMinimum required validation recall: {minimum_recall:.0%}")
    print(f"Selected threshold: {selected_threshold:.1f}")
    print(f"Recall at selected threshold: {selected_row['recall']:.3f}")
    print(f"Precision at selected threshold: {selected_row['precision']:.3f}")
    print(f"F1 at selected threshold: {selected_row['f1']:.3f}")
    print(f"Validation rows flagged: {int(selected_row['flagged_rows'])}")

    return selected_threshold, threshold_df


def evaluate_final_test(model, X_test, y_test, threshold):
    """Evaluate the selected model once on the held-out test split."""

    # Predict fraud probabilities; apply the threshold selected on validation.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    # Explicitly set label order so the matrix is always [[TN, FP], [FN, TP]].
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    flagged_rows = int(predictions.sum())
    false_positives_per_true_positive = (
        fp / tp if tp > 0 else float("inf")
    )

    print("\nFINAL TEST-SET RESULTS")
    print("-" * 45)

    for metrics_name, metric_value in metrics.items():
        print(f"{metrics_name}: {metric_value:.4f}")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array([[tn, fp], [fn, tp]]))

    print(f"\nTrue negatives (TN): {tn}")
    print(f"False positives (FP): {fp}")
    print(f"False negatives (FN): {fn}")
    print(f"True positives (TP): {tp}")
    print(f"Total flagged transactions: {flagged_rows}")
    print(
        "False positives per true positive:"
        f"{false_positives_per_true_positive:.2f}"
    )

    return {
        "metrics": metrics,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
        "flagged_rows": flagged_rows,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positives_per_true_positive": (
            false_positives_per_true_positive
        )
    }


def demonstrate_data_leakage(
    featured_df,
    X_train,
    X_val,
    y_train,
    y_val,
    best_params,
    training_median
):
    """Compare clean and intentionally leaked models on identical splits."""

    # The clean model uses only the approved 26 features.
    clean_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
        **best_params
    )

    clean_model.fit(X_train, y_train)
    clean_probabilities = clean_model.predict_proba(X_val)[:, 1]
    clean_predictions = (clean_probabilities >= 0.5).astype(int)

    clean_f1 = f1_score(y_val, clean_predictions, zero_division=0)

    # Confirm the leakage field exists in the source data.
    if "chargeback_filed" not in featured_df.columns:
        raise ValueError(
            "Leakage demonstration requires chargeback_filed."
        )

    # Confirm row indices are unique and consistent.
    if not featured_df.index.is_unique:
        raise ValueError("featured_df contains duplicate row indices.")

    if not X_train.index.is_unique or not X_val.index.is_unique:
        raise ValueError("Train/validation indices must be unique.")

    if not X_train.index.isin(featured_df.index).all():
        raise ValueError("Some training rows are missing from featured_df.")

    if not X_val.index.isin(featured_df.index).all():
        raise ValueError("Some validation rows are missing from featured_df.")

    if not X_train.index.intersection(X_val.index).empty:
        raise ValueError("Training and validation rows overlap.")

    if not y_train.index.equals(X_train.index):
        raise ValueError("y_train indices do not match X_train.")

    if not y_val.index.equals(X_val.index):
        raise ValueError("y_val indices do not match X_val.")

    # Verify that source targets match the train/validation labels.
    source_target = featured_df["is_fraud"]

    if not source_target.loc[X_train.index].astype(int).equals(
        y_train.astype(int)
    ):
        raise ValueError("Training labels do not match featured_df rows.")

    if not source_target.loc[X_val.index].astype(int).equals(
        y_val.astype(int)
    ):
        raise ValueError("Validation labels do not match featured_df rows.")

    # Demonstration only: add the post-outcome chargeback field.
    # NEVER use this leaked model in the API or final production model.
    leaked_X = featured_df.drop(
        columns=["txn_id", "customer_id", "is_fraud"]
    ).copy()

    leaked_X = pd.get_dummies(
        leaked_X,
        columns=["merchant_category", "city", "payment_mode"],
        dtype=int
    )

    # Keep the same clean features in the same order, with leakage last.
    leaked_X = leaked_X.reindex(
        columns=MODEL_COLUMNS + ["chargeback_filed"],
        fill_value=0
    )

    # Use the training-set median for a fair preprocessing comparison.
    leaked_X["distance_from_home_km"] = (
        leaked_X["distance_from_home_km"].fillna(training_median)
    )

    # Reuse the original row indices from the train/validation split.
    # No index reset is performed; alignment is checked explicitly above.
    train_indices = X_train.index
    val_indices = X_val.index

    X_train_leaked = leaked_X.loc[train_indices]
    X_val_leaked = leaked_X.loc[val_indices]

    leaked_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
        **best_params
    )

    leaked_model.fit(X_train_leaked, y_train)
    leaked_probabilities = leaked_model.predict_proba(X_val_leaked)[:, 1]
    leaked_predictions = (leaked_probabilities >= 0.5).astype(int)

    leaked_f1 = f1_score(y_val, leaked_predictions, zero_division=0)

    print("\nDATA LEAKAGE DEMONSTRATION (threshold = 0.50)")
    print(f"Clean validation F1:  {clean_f1:.4f}")
    print(f"Leaked validation F1: {leaked_f1:.4f}")
    print(f"F1 difference:        {leaked_f1 - clean_f1:+.4f}")
    print(
        "Leakage warning: chargeback_filed is a post-outcome field "
        "and must never be used as a prediction feature."
    )

    return {
        "clean_validation_f1": clean_f1,
        "leaked_validation_f1": leaked_f1,
        "f1_difference": leaked_f1 - clean_f1
    }


def report_feature_importance(model, feature_columns):
    """Display the five most influential features in the fitted forest."""

    importances = model.feature_importances_

    if len(importances) != len(feature_columns):
        raise ValueError(
            "Feature importance count does not match feature column count."
        )

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances
    }).sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    print("\nTOP 5 FEATURE IMPORTANCES")
    print(importance_df.head(5).to_string(index=False))

    return importance_df


def build_final_model():
    """
    Build the model and return the metadata required by the API.

    The validation set selects the threshold. After that decision, the final
    estimator is refitted on train + validation data. The held-out test set
    remains excluded from fitting.
    """

    X, y, featured_df = load_and_prepare_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        training_median
    ) = split_and_impute(X, y)

    # Compare the three required Logistic Regression experiments.
    run_logistic_regression_experiments(
        X_train,
        y_train,
        X_val,
        y_val
    )

    # Tune hyperparameters using training data only.
    tuning_results = run_cross_validation_and_tuning(X_train, y_train)
    best_params = tuning_results["grid_search"].best_params_

    # Fit the selected model on the training split for validation decisions.
    validation_model = tuning_results["best_model"]

    # Choose a threshold from validation predictions only.
    threshold, _ = tune_decision_threshold(
        validation_model,
        X_val,
        y_val,
        minimum_recall=0.65
    )

    # Evaluate the held-out test set once, after model/threshold selection.
    evaluate_final_test(
        validation_model,
        X_test,
        y_test,
        threshold
    )

    # Required controlled leakage comparison; never used as the final model.
    demonstrate_data_leakage(
        featured_df,
        X_train,
        X_val,
        y_train,
        y_val,
        best_params,
        training_median
    )

    # Refit the clean selected estimator using train + validation data.
    # The median is intentionally retained from the training split.
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    final_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
        **best_params
    )
    final_model.fit(X_train_val, y_train_val)

    # The final model must use only approved prediction features.
    if list(final_model.feature_names_in_) != MODEL_COLUMNS:
        raise ValueError(
            "Final model feature columns do not match MODEL_COLUMNS."
        )

    if "chargeback_filed" in final_model.feature_names_in_:
        raise ValueError(
            "Leakage feature found in the final model."
        )

    report_feature_importance(final_model, MODEL_COLUMNS)

    return {
        "model": final_model,
        "columns": MODEL_COLUMNS.copy(),
        "median": float(training_median),
        "threshold": float(threshold)
    }


def save_model_artifact(artifacts):
    """Save the trained model and inference metadata to disk."""

    artifact_path = PROJECT_ROOT / "model_artifact.pkl"

    with artifact_path.open("wb") as file:
        pickle.dump(artifacts, file)

    print(f"\nModel artifact saved to: {artifact_path}")



def main():
    """Run the complete Part B training and evaluation workflow."""

    print("=" * 65)
    print("RAZORPAY MERCHANT RISK DESK — MODEL TRAINING")
    print("=" * 65)

    artifacts = build_final_model()
    save_model_artifact(artifacts)

    print("\nFINAL MODEL ARTIFACT CHECK")
    print(f"Model type: {type(artifacts['model']).__name__}")
    print(f"Number of model columns: {len(artifacts['columns'])}")
    print(f"Training-derived distance median: {artifacts['median']}")
    print(f"Selected decision threshold: {artifacts['threshold']:.1f}")

    # Guard against accidental inclusion of identifiers or the target.
    forbidden_columns = {
        "txn_id",
        "customer_id",
        "chargeback_filed",
        "is_fraud",
    }

    if forbidden_columns.intersection(artifacts["columns"]):
        raise ValueError("Forbidden columns found in the final model schema.")

    if len(artifacts["columns"]) != 26:
        raise ValueError(
            f"Expected 26 model features, got {len(artifacts['columns'])}."
        )

    print("\nPart B training workflow completed.")


if __name__ == "__main__":
    main()