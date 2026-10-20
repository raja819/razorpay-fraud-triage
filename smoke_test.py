
import json
import re
import sys
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8000"
LOG_FILE = Path(__file__).resolve().parent / "service.log"

passed = 0
failed = 0


def check(number, description, test_function):
    """Run one smoke-test check and report its result."""
    global passed, failed

    try:
        test_function()
        passed += 1
        print(f"PASS Q{number}: {description}")
    except Exception as exc:
        failed += 1
        print(f"FAIL Q{number}: {description}")
        print(f"    Reason: {exc}")


def assert_status(response, expected_status):
    assert response.status_code == expected_status, (
        f"Expected HTTP {expected_status}, "
        f"received {response.status_code}: {response.text}"
    )


def score_payload(**overrides):
    """Return a valid transaction payload with optional overrides."""
    payload = {
        "txn_id": "T900101",
        "merchant_category": "Grocery",
        "city": "Bengaluru",
        "payment_mode": "UPI",
        "amount_inr": 1500.0,
        "hour_of_day": 14,
        "txn_count_last_24h": 2,
        "avg_amount_30d_inr": 1200.0,
        "distance_from_home_km": 5.0,
        "is_new_device": 0,
        "account_age_days": 365,
    }
    payload.update(overrides)
    return payload


def post_json(path, payload):
    return requests.post(
        f"{BASE_URL}{path}",
        json=payload,
        timeout=15,
    )


def test_valid_score():
    response = post_json("/score", score_payload())
    assert_status(response, 200)

    body = response.json()
    assert "fraud_probability" in body, body
    assert isinstance(body["flagged"], bool), body
    assert body["decision"] in {
        "approve",
        "hold_for_review",
        "decline",
    }, body


def test_high_risk_score():
    response = post_json(
        "/score",
        score_payload(
            txn_id="T900102",
            amount_inr=100000.0,
            is_new_device=1,
        ),
    )
    assert_status(response, 200)

    body = response.json()
    assert body["decision"] == "hold_for_review", body
    assert body["human_review"] is True, body


def test_invalid_amount():
    response = post_json(
        "/score",
        score_payload(amount_inr=0.0),
    )
    assert_status(response, 400)

    body = response.json()
    assert body.get("field") == "amount_inr", body


def test_wrong_field_type():
    response = post_json(
        "/score",
        score_payload(hour_of_day="fourteen"),
    )
    assert_status(response, 422)


def test_missing_required_field():
    payload = score_payload()
    del payload["amount_inr"]

    response = post_json("/score", payload)
    assert_status(response, 422)


def test_unknown_merchant_category():
    response = post_json(
        "/score",
        score_payload(merchant_category="UnknownCategory"),
    )
    assert_status(response, 400)

    body = response.json()
    assert body.get("field") == "merchant_category", body


def test_matching_policy():
    payload = score_payload(
        txn_id="T900101",
        question="temporary hold on repeated UPI payments within a day",
    )
    response = post_json("/case", payload)
    assert_status(response, 200)

    body = response.json()
    assert body.get("case_status") == "success", body
    assert body.get("policy_id") == "P2", body
    assert body.get("decision") == "hold_for_review", body
    assert body.get("human_review") is True, body


def test_no_policy_match():
    payload = score_payload(
        txn_id="T900100",
        question="what is the cafeteria menu today",
    )
    response = post_json("/case", payload)
    assert_status(response, 200)

    body = response.json()
    assert body.get("case_status") == "no_policy_match", body
    assert body.get("policy_id") is None, body
    assert body.get("decision") == "hold_for_review", body
    assert body.get("human_review") is True, body


def test_safe_default():
    payload = score_payload(
        txn_id="T900005",
        merchant_category="Electronics",
        account_age_days=10,
        question=(
            "electronics merchant payment on an account "
            "younger than 30 days"
        ),
    )
    response = post_json("/case", payload)
    assert_status(response, 200)

    body = response.json()
    assert body.get("safe_default") is True, body
    assert body.get("decision") == "hold_for_review", body
    assert body.get("human_review") is True, body


def test_simulated_service_outage():
    payload = score_payload(
        txn_id="T900099",
        question="temporary hold on repeated UPI payments within a day",
    )
    response = post_json("/case", payload)
    assert_status(response, 503)

    body = response.json()
    assert "error" in body, body


def test_health_metrics_and_logs():
    # Health endpoint
    health_response = requests.get(
        f"{BASE_URL}/health",
        timeout=10,
    )
    assert_status(health_response, 200)
    assert health_response.json().get("status") == "ok"

    # Metrics endpoint
    metrics_response = requests.get(
        f"{BASE_URL}/metrics",
        timeout=10,
    )
    assert_status(metrics_response, 200)
    metrics = metrics_response.json()

    expected_metrics = {
        "requests_total": 10,
        "success_total": 5,
        "error_total": 5,
        "human_review_total": 4,
    }

    for key, expected in expected_metrics.items():
        assert metrics.get(key) == expected, (
            f"{key}: expected {expected}, got {metrics.get(key)}"
        )

    assert abs(metrics.get("success_rate", -1) - 0.5) < 1e-9, metrics
    assert abs(metrics.get("error_rate", -1) - 0.5) < 1e-9, metrics
    assert abs(metrics.get("human_review_rate", -1) - 0.4) < 1e-9, metrics

    # Exactly 10 POST request log lines
    assert LOG_FILE.exists(), f"Log file not found: {LOG_FILE}"

    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    post_lines = [line for line in lines if "method=POST" in line]

    assert len(post_lines) == 10, (
        f"Expected exactly 10 POST log lines, got {len(post_lines)}"
    )

    assert all("path=" in line and "status=" in line for line in post_lines), (
        "A POST log line is missing path or status"
    )

    levels = {
        "INFO": sum("level=INFO" in line for line in post_lines),
        "WARNING": sum("level=WARNING" in line for line in post_lines),
        "ERROR": sum("level=ERROR" in line for line in post_lines),
    }

    expected_levels = {"INFO": 5, "WARNING": 4, "ERROR": 1}
    assert levels == expected_levels, (
        f"Expected log levels {expected_levels}, got {levels}"
    )


def main():
    global passed, failed

    print("=" * 60)
    print("RAZORPAY MERCHANT RISK DESK - SMOKE TEST")
    print("=" * 60)

    # Q1-Q10 each send exactly one POST request.
    tests = [
        (1, "Valid transaction scoring", test_valid_score),
        (2, "High-risk transaction held for review", test_high_risk_score),
        (3, "Invalid amount returns HTTP 400", test_invalid_amount),
        (4, "Wrong field type returns HTTP 422", test_wrong_field_type),
        (5, "Missing required field returns HTTP 422", test_missing_required_field),
        (6, "Unknown merchant category returns HTTP 400", test_unknown_merchant_category),
        (7, "Matching policy produces a safe case decision", test_matching_policy),
        (8, "No policy match safely holds the transaction", test_no_policy_match),
        (9, "Invalid case-note output uses safe default", test_safe_default),
        (10, "Simulated case-note outage returns HTTP 503", test_simulated_service_outage),
    ]

    for number, description, test_function in tests:
        check(number, description, test_function)

    # Q11 combines health, metrics, and log verification.
    check(
        11,
        "Health, metrics, exactly 10 POST logs, and log levels",
        test_health_metrics_and_logs,
    )

    print("=" * 60)
    print(f"Checks passed: {passed}/11")
    print(f"Checks failed: {failed}/11")
    print("=" * 60)

    if failed:
        sys.exit(1)

    print("ALL SMOKE TEST CHECKS PASSED.")


if __name__ == "__main__":
    main()
