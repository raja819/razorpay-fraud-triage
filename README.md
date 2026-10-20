# Razorpay Merchant Risk Desk — Fraud Triage Service

An offline-first fraud triage service that combines transaction data preparation, machine learning risk scoring, policy retrieval, AI-assisted case notes, and a FastAPI interface. The system is designed to support human review rather than automatically approve suspicious transactions.

## Overview

This project implements a fraud triage workflow using Python, pandas, NumPy, scikit-learn, FastAPI, and a provided deterministic mock LLM.

**Key capabilities**
- Audit and clean raw transaction data.
- Engineer transaction risk features.
- Train a fraud classification model and select a decision threshold.
- Retrieve relevant fraud policies using TF-IDF and cosine similarity.
- Generate and validate case notes with retry and safe-default behavior.
- Expose transaction scoring, case handling, health, and metrics endpoints.
- Log POST requests and validate service behavior through automated smoke tests.

**Important:** This is a portfolio project and demonstration service, not an official Razorpay product or a production-ready fraud prevention system.

## Project Structure

```text
razorpay-fraud-triage/
├── app.py
├── prepare.py
├── train.py
├── retrieval.py
├── casenote.py
├── smoke_test.py
├── generate_data.py
├── mock_llm.py
├── fraud_policies.json
├── retrieval_queries.json
├── case_scenarios.json
├── transactions_raw.csv
├── model_artifact.pkl
├── service.log
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.11 or later
- Git
- Windows, macOS, or Linux
- No paid APIs or cloud services are required for the provided offline workflow

Install dependencies in a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The assignment specifies the required pandas and scikit-learn version ranges in `requirements.txt`. Use a compatible Python environment if your installed versions cannot satisfy those pins.

## Reproduce the Project

Run commands from the repository root.

### 1. Generate the raw dataset

The supplied `generate_data.py` is part of the assignment and must remain unmodified.

```powershell
python generate_data.py
```

This generates `transactions_raw.csv`. The committed CSV is included so the generated dataset can also be inspected directly.

### 2. Prepare and audit the data

```powershell
python prepare.py
```

This step audits duplicates, category spellings, missing values, amount ranges, fraud prevalence, and chargeback behavior. It also creates engineered features and checks that the input data is not mutated.

### 3. Train and evaluate the model

```powershell
python train.py
```

This step performs a stratified train/validation/test split, trains and compares candidate models, selects a threshold, reports test metrics, and saves the fitted model and preprocessing metadata in `model_artifact.pkl`.

### 4. Test policy retrieval

```powershell
python retrieval.py
```

The retrieval component uses TF-IDF vectors and cosine similarity to find the closest policy in `fraud_policies.json`. Queries below the configured `MIN_SCORE` threshold return `no_policy_match`.

### 5. Test case-note generation

```powershell
python casenote.py
```

The case-note component uses the supplied `mock_llm.py`, validates generated output, retries invalid responses, and uses a safe default when validation fails. The mock service also includes a simulated connection failure for testing.

### 6. Start the API

```powershell
python -m uvicorn app:app --reload
```

Wait until the server reports that application startup is complete. Keep this terminal running.

### 7. Run the smoke tests

Open a second terminal, activate the virtual environment, navigate to the repository root, and run:

```powershell
python smoke_test.py
```

The smoke test checks transaction scoring, validation errors, policy handling, safe defaults, simulated service outages, health and metrics endpoints, and request logging.

The expected final message when all checks pass is:

```text
Checks passed: 11/11
Checks failed: 0/11
ALL SMOKE TEST CHECKS PASSED.
```

## Part A — Data Audit and Feature Engineering

The raw dataset contained **6,090 rows and 14 columns**.

| Audit item | Result |
|---|---:|
| Exact duplicate rows | 90 |
| Rows after cleaning | 5,919 |
| Missing amount values after deduplication | 76 |
| Out-of-range amount values | 5 |
| Missing distance values after cleaning | 191 |
| Fraudulent transactions | 366 |
| Non-fraudulent transactions | 5,553 |
| Fraud prevalence | 6.18% |

The audit identified inconsistent category and city spellings. After cleaning, merchant categories included Grocery, Food Delivery, Fashion, Electronics, Utilities, and Travel.

Engineered features included:

- `amount_vs_avg_ratio`
- `is_night`
- `high_value`

The cleaned dataset retained 5,919 rows, and the engineered feature table contained 17 columns. The maximum observed `amount_vs_avg_ratio` was approximately 13.11.

The `chargeback_filed` field was identified as a post-outcome field and excluded from model features to avoid target leakage.

## Part B — Model Training and Threshold Selection

The dataset was split into stratified training, validation, and test sets.

| Split | Rows | Fraud rows |
|---|---:|---:|
| Training | 3,551 | 220 |
| Validation | 1,184 | 73 |
| Test | 1,184 | 73 |

Missing `distance_from_home_km` values were imputed using the training-set median of 12.6. Categorical features were one-hot encoded, producing 26 model input columns.

The final model was a Random Forest classifier. A decision threshold of **0.40** was selected to prioritize fraud recall.

### Test-set results

| Metric | Result |
|---|---:|
| Accuracy | 0.8632 |
| Precision | 0.2808 |
| Recall | 0.7808 |
| F1 score | 0.4130 |
| ROC-AUC | 0.9211 |
| Flagged transactions | 203 |
| False positives | 146 |
| True positives | 57 |

Confusion matrix, with rows representing actual classes and columns representing predicted classes:

```text
                 Predicted 0   Predicted 1
Actual 0              965           146
Actual 1               16            57
```

The threshold captures a high proportion of fraudulent transactions, but the relatively low precision means many flagged transactions are legitimate. This trade-off makes human review important.

A leakage demonstration showed validation F1 increasing from 0.4410 to 0.7975 when the post-outcome chargeback field was improperly included. This illustrates why leakage checks are essential.

## Part C — Policy Retrieval and Case Notes

### Policy retrieval

The retrieval component uses TF-IDF vectorization and cosine similarity to rank policy matches.

- Configured minimum similarity score: `0.45`
- In-scope retrieval test cases passed: 8/8
- Out-of-scope test cases correctly returned no match: 2/2
- Total retrieval test cases passed: 10/10

A query that scores below the minimum threshold is treated as `no_policy_match`. This prevents the service from inventing a policy match when the retrieved evidence is insufficient.

### Case-note validation

The case-note workflow validates generated output and retries invalid responses up to three attempts. It also handles malformed JSON, missing fields, invalid values, incorrect policy IDs, and invalid team assignments.

Recorded scenario results:

- Scenarios tested: 10
- Validated case notes: 9
- Safe-default outcomes: 1
- No-policy-match scenario: handled separately
- Simulated connection failure: tested with transaction ID `T900099`

The case-note service is designed to avoid automatically approving a flagged transaction. When the generated output cannot be validated, the workflow falls back to a safe hold decision.

## Part D — FastAPI Service and Smoke Tests

The FastAPI application exposes the following endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /score` | Score a transaction and return a triage decision |
| `POST /case` | Retrieve policy guidance and produce a safe case decision |
| `GET /health` | Report service health |
| `GET /metrics` | Return in-memory request and review metrics |

The API validates request types and business values. Invalid business values return HTTP 400, malformed or missing required fields return HTTP 422, and the simulated case-note outage returns HTTP 503.

Additional safeguards include:

- Holding certain unflagged high-value transactions with a new device for human review.
- Safely holding transactions when no policy matches.
- Logging each POST request with its level, path, HTTP status, and human-review indicator.
- Tracking total requests, successes, errors, and human-review decisions.

### Smoke-test results

**11/11 checks passed.**

The smoke test verified:

1. Valid transaction scoring
2. High-risk transaction review
3. Invalid amount handling
4. Wrong field type handling
5. Missing required field handling
6. Unknown merchant category handling
7. Matching policy case decision
8. Safe hold when no policy matches
9. Safe default for invalid case-note output
10. Simulated case-note outage
11. Health, metrics, POST log count, and log levels

The verified `service.log` contained exactly 10 POST entries: 5 INFO, 4 WARNING, and 1 ERROR.

## Limitations

1. **Synthetic data:** Results are based on the supplied generated dataset and may not reflect real transaction behavior or real-world fraud prevalence.
2. **Precision and review workload:** The selected threshold produces false positives, so legitimate transactions may be sent for human review.
3. **Model generalization:** The evaluation uses a limited dataset and does not establish performance on future transactions or unseen production populations.
4. **Policy retrieval limitations:** TF-IDF is lexical rather than deeply semantic. Relevant policies may be missed when query wording differs, and low-confidence queries correctly produce no match.
5. **Mock LLM:** Case-note generation uses a deterministic mock service and does not establish the quality, reliability, or safety of a real language model.
6. **In-memory metrics:** Metrics reset when the application restarts and are not designed for durable monitoring across multiple workers.
7. **No production integrations:** The project does not integrate with live payment processing, customer accounts, external policy systems, or a production alerting platform.
8. **Human oversight required:** This is a demonstration triage workflow. Decisions should be reviewed and validated before use in a real financial system.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- TF-IDF and cosine similarity
- FastAPI
- Pydantic
- Uvicorn
- Git and GitHub

## Disclaimer

This repository is an independent educational portfolio project. It is not affiliated with, endorsed by, or an official product of Razorpay. The service and results are for demonstration purposes only and should not be used as the sole basis for real financial or fraud decisions.
