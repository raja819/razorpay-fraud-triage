
"""FastAPI service for the Razorpay Merchant Risk Desk."""

import logging
import pickle
from pathlib import Path
from threading import Lock
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, StrictStr
from casenote import run_case


ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "model_artifact.pkl"
LOG_PATH = ROOT / "service.log"

ALLOWED_CATEGORIES = {
    "Grocery",
    "Electronics",
    "Travel",
    "Food Delivery",
    "Fashion",
    "Utilities",
}

CATEGORICAL_COLUMNS = [
    "merchant_category",
    "city",
    "payment_mode",
]

# Load the trained model once when the service starts.
with ARTIFACT_PATH.open("rb") as file:
    artifact = pickle.load(file)

MODEL = artifact["model"]
MODEL_COLUMNS = list(artifact["columns"])
DISTANCE_MEDIAN = float(artifact["median"])
FRAUD_THRESHOLD = float(artifact["threshold"])

# One log entry per POST request, including unsuccessful requests.
logger = logging.getLogger("risk_service")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

file_handler = logging.FileHandler(
    LOG_PATH, mode="w", encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(file_handler)

app = FastAPI(
    title="Razorpay Merchant Risk Desk",
    version="1.0.0",
)

@app.get("/health")
async def health():
    return {"status": "ok"}

metrics = {
    "requests_total": 0,
    "success_total": 0,
    "error_total": 0,
    "human_review_total": 0,
}
metrics_lock = Lock()


class Transaction(BaseModel):
    """Strict types ensure malformed inputs receive HTTP 422."""

    model_config = ConfigDict(strict=True)

    txn_id: StrictStr
    merchant_category: StrictStr
    city: StrictStr
    payment_mode: StrictStr
    amount_inr: StrictFloat
    hour_of_day: StrictInt
    txn_count_last_24h: StrictInt
    avg_amount_30d_inr: StrictFloat
    is_new_device: StrictInt
    account_age_days: StrictInt
    distance_from_home_km: Optional[StrictFloat | StrictInt] = None


class CaseTransaction(Transaction):
    question: StrictStr


class BusinessInputError(Exception):
    def __init__(self, message: str, field: str):
        self.message = message
        self.field = field


@app.exception_handler(BusinessInputError)
async def business_input_error_handler(
    request: Request, exc: BusinessInputError
):
    return JSONResponse(
        status_code=400,
        content={"error": exc.message, "field": exc.field},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
):
    # Preserve FastAPI's 422 behavior for missing or wrongly typed fields.
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(ConnectionError)
async def connection_error_handler(
    request: Request, exc: ConnectionError
):
    return JSONResponse(
        status_code=503,
        content={"error": "case-note service temporarily unavailable, try again later"},
    )


@app.middleware("http")
async def request_metrics_and_logging(request: Request, call_next):
    response = await call_next(request)

    if request.method == "POST":
        status = response.status_code
        needs_review = bool(
            getattr(request.state, "human_review", False)
        )

        with metrics_lock:
            metrics["requests_total"] += 1

            if status < 400:
                metrics["success_total"] += 1
            else:
                metrics["error_total"] += 1

            if needs_review:
                metrics["human_review_total"] += 1

        if status >= 500:
            level = "ERROR"
        elif status >= 400:
            level = "WARNING"
        else:
            level = "INFO"

        log_message = (
            f"level={level} method=POST "
            f"path={request.url.path} "
            f"status={status} human_review={needs_review}"
        )

        if level == "ERROR":
            logger.error(log_message)
        elif level == "WARNING":
            logger.warning(log_message)
        else:
            logger.info(log_message)

    return response


def validate_business_values(txn: Transaction) -> None:
    """Return HTTP 400 for invalid, correctly typed business values."""

    if not 1 <= txn.amount_inr <= 1_000_000:
        raise BusinessInputError(
            "amount_inr must be between 1 and 1000000",
            "amount_inr",
        )

    if not 0 <= txn.hour_of_day <= 23:
        raise BusinessInputError(
            "hour_of_day must be between 0 and 23",
            "hour_of_day",
        )

    if txn.txn_count_last_24h < 0:
        raise BusinessInputError(
            "txn_count_last_24h must be non-negative",
            "txn_count_last_24h",
        )

    if txn.avg_amount_30d_inr <= 0:
        raise BusinessInputError(
            "avg_amount_30d_inr must be greater than 0",
            "avg_amount_30d_inr",
        )

    normalized_category = txn.merchant_category.strip().title()
    if normalized_category not in ALLOWED_CATEGORIES:
        raise BusinessInputError(
            "Unknown merchant_category",
            "merchant_category",
        )

    if txn.is_new_device not in (0, 1):
        raise BusinessInputError(
            "is_new_device must be 0 or 1",
            "is_new_device",
        )

    if txn.account_age_days < 0:
        raise BusinessInputError(
            "account_age_days must be non-negative",
            "account_age_days",
        )

    if (
        txn.distance_from_home_km is not None
        and txn.distance_from_home_km < 0
    ):
        raise BusinessInputError(
            "distance_from_home_km must be non-negative",
            "distance_from_home_km",
        )


def transaction_features(txn: Transaction) -> pd.DataFrame:
    """Build exactly the features and column order used during training."""

    row = txn.model_dump(exclude={"txn_id"})
    row["merchant_category"] = txn.merchant_category.strip().title()
    row["city"] = txn.city.strip().title()
    row["payment_mode"] = txn.payment_mode.strip().title()

    # Missing distance is imputed using the training-set median.
    if row["distance_from_home_km"] is None:
        row["distance_from_home_km"] = DISTANCE_MEDIAN

    df = pd.DataFrame([row])

    # Match prepare.py's feature engineering.
    df["amount_vs_avg_ratio"] = (
        df["amount_inr"] / df["avg_amount_30d_inr"]
    )
    df["is_night"] = df["hour_of_day"].le(5).astype("int64")
    df["high_value"] = df["amount_inr"].gt(50_000).astype("int64")

    # These columns are outcomes/identifiers, not model inputs.
    df = df.drop(
        columns=["customer_id", "chargeback_filed", "is_fraud"],
        errors="ignore",
    )

    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=False,
        dtype=int
    )

    # Guarantee training-time column names and order.
    df = df.reindex(columns=MODEL_COLUMNS, fill_value=0)
    return df


def score_transaction(txn: Transaction) -> dict:
    """Score a transaction and apply the required safety override."""

    validate_business_values(txn)
    features = transaction_features(txn)

    fraud_probability = float(MODEL.predict_proba(features)[0][1])
    flagged = fraud_probability >= FRAUD_THRESHOLD

    # An unflagged high-value transaction on a new device must be held.
    safety_override = (
        not flagged
        and txn.amount_inr >= 100_000
        and txn.is_new_device == 1
    )

    human_review = flagged or safety_override
    decision = "hold_for_review" if human_review else "approve"

    return {
        "txn_id": txn.txn_id,
        "fraud_probability": round(fraud_probability, 4),
        "flagged": bool(flagged),
        "decision": decision,
        "human_review": bool(human_review),
    }


@app.post("/score")
async def score_endpoint(txn: Transaction, request: Request):
    result = score_transaction(txn)
    request.state.human_review = result["human_review"]
    return result


@app.post("/case")
async def case_endpoint(txn: CaseTransaction, request: Request):
    # Validate and score first, using the same saved model as /score.
    result = score_transaction(txn)
    request.state.human_review = result["human_review"]

    case_result = run_case(
        txn_id=txn.txn_id,
        fraud_probability=result["fraud_probability"],
        amount_inr=txn.amount_inr,
        question=txn.question,
    )

    result["attempts"] = case_result.get("attempts", 0)
    result["safe_default"] = (
        case_result.get("status") == "safe_default"
    )
    result["case_note"] = case_result.get("case_note")
    result["case_status"] = case_result.get("status")
    result["policy_id"] = case_result.get("policy_id")

    # No policy match must fail safely.
    if case_result.get("status") == "no_policy_match":
        result["decision"] = "hold_for_review"
        result["human_review"] = True
        request.state.human_review = True

    # Keep the API decision consistent with case-note safety rules.
    note = result["case_note"]

    if result["flagged"]:
        result["decision"] = "hold_for_review"
        result["human_review"] = True
        request.state.human_review = True

        if note and note.get("recommended_action") == "approve":
            note["recommended_action"] = "hold_for_review"

    elif note and note.get("recommended_action") == "hold_for_review":
        result["decision"] = "hold_for_review"
        result["human_review"] = True
        request.state.human_review = True

    # Safe defaults always require a human to review the case.
    if result["safe_default"]:
        result["decision"] = "hold_for_review"
        result["human_review"] = True
        request.state.human_review = True

    return result


@app.get("/metrics")
async def get_metrics():
    with metrics_lock:
        snapshot = dict(metrics)

    total = snapshot["requests_total"]

    snapshot["success_rate"] = (
        round(snapshot["success_total"] / total, 4)
        if total else 0.0
    )
    snapshot["error_rate"] = (
        round(snapshot["error_total"] / total, 4)
        if total else 0.0
    )
    snapshot["human_review_rate"] = (
        round(snapshot["human_review_total"] / total, 4)
        if total else 0.0
    )

    return snapshot