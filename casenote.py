
"""Generate and validate fraud-triage case notes using the offline mock."""

import json
from pathlib import Path

from mock_llm import generate, reset
from retrieval import retrieve


PROJECT_ROOT = Path(__file__).resolve().parent
SCENARIOS_PATH = PROJECT_ROOT / "case_scenarios.json"

MAX_ATTEMPTS = 3

REQUIRED_KEYS = {
    "summary",
    "priority",
    "recommended_team",
    "recommended_action",
    "policy_id",
}

ALLOWED_PRIORITIES = {"low", "medium", "high"}
ALLOWED_TEAMS = {
    "card_risk",
    "upi_risk",
    "customer_care",
    "merchant_ops",
}
ALLOWED_ACTIONS = {
    "approve",
    "hold_for_review",
    "block_and_notify",
}


def build_prompt(txn_id, fraud_probability, amount_inr, policy_id):
    """Build a prompt containing the four required labelled lines."""
    return (
        "ROLE: You are a merchant fraud-triage assistant.\n"
        "TASK: Draft a concise case note using the applicable policy.\n"
        "CONSTRAINTS: Return only a JSON object. Do not invent facts.\n"
        "OUTPUT FORMAT: summary, priority, recommended_team, "
        "recommended_action, policy_id.\n"
        f"TXN_ID: {txn_id}\n"
        f"FRAUD_PROBABILITY: {fraud_probability}\n"
        f"AMOUNT_INR: {amount_inr}\n"
        f"POLICY_ID: {policy_id}\n"
    )


def validate_case_note(raw_text, expected_policy_id):
    """
    Validate generated output using three gates.

    Gate 1: valid JSON object.
    Gate 2: all required keys exist and every value is a string.
    Gate 3: allowed enum values and matching policy ID.
    Gate 0: valid case note.
    """
    # Gate 1: JSON parsing and object check.
    try:
        note = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return 1, None, "invalid_json"

    if not isinstance(note, dict):
        return 1, None, "json_not_object"

    # Gate 2: required keys and string values.
    if not REQUIRED_KEYS.issubset(note.keys()):
        return 2, note, "missing_required_key"

    if not all(isinstance(note[key], str) for key in REQUIRED_KEYS):
        return 2, note, "non_string_value"

    # Gate 3: enums and policy consistency.
    if note["priority"] not in ALLOWED_PRIORITIES:
        return 3, note, "invalid_priority"

    if note["recommended_team"] not in ALLOWED_TEAMS:
        return 3, note, "invalid_recommended_team"

    if note["recommended_action"] not in ALLOWED_ACTIONS:
        return 3, note, "invalid_recommended_action"

    if note["policy_id"] not in {
        "P1", "P2", "P3", "P4", "P5", "P6"
    }:
        return 3, note, "invalid_policy_id"

    if note["policy_id"] != expected_policy_id:
        return 3, note, "policy_id_mismatch"

    return 0, note, "valid"


def safe_default(txn_id, policy_id, reason):
    """Return a conservative result if generation cannot be validated."""
    return {
        "txn_id": txn_id,
        "status": "safe_default",
        "policy_id": policy_id,
        "attempts": MAX_ATTEMPTS,
        "validation_gate": None,
        "reason": reason,
        "case_note": {
            "summary": (
                f"Transaction {txn_id} requires manual review because "
                "a validated case note could not be generated."
            ),
            "priority": "high",
            "recommended_team": "merchant_ops",
            "recommended_action": "hold_for_review",
            "policy_id": policy_id,
        },
    }


def run_case(txn_id, fraud_probability, amount_inr, question):
    """Retrieve the policy, generate a case note, and validate it."""
    policy_id, policy_text, score = retrieve(question)

    # Do not ask the mock to invent a note if no policy matched.
    if policy_id is None:
        return {
            "txn_id": txn_id,
            "status": "no_policy_match",
            "policy_id": None,
            "similarity_score": score,
            "attempts": 0,
            "validation_gate": None,
            "case_note": None,
        }

    prompt = build_prompt(
        txn_id=txn_id,
        fraud_probability=fraud_probability,
        amount_inr=amount_inr,
        policy_id=policy_id,
    )

    last_reason = "unknown_validation_error"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Service outages must propagate to the caller.
        raw_text = generate(prompt)

        gate, note, reason = validate_case_note(
            raw_text,
            expected_policy_id=policy_id,
        )

        if gate == 0:
            # Safety override: high-risk transactions must not be approved.
            if (
                fraud_probability >= 0.5
                and note["recommended_action"] == "approve"
            ):
                note["recommended_action"] = "hold_for_review"

            return {
                "txn_id": txn_id,
                "status": "success",
                "policy_id": policy_id,
                "similarity_score": score,
                "attempts": attempt,
                "validation_gate": 0,
                "case_note": note,
            }

        last_reason = reason

    # All three attempts were invalid. Fail safely.
    result = safe_default(txn_id, policy_id, last_reason)
    result["similarity_score"] = score
    result["attempts"] = MAX_ATTEMPTS
    result["validation_gate"] = gate
    return result


def run_scenarios():
    """Run the provided ten case scenarios and print their outcomes."""
    with SCENARIOS_PATH.open("r", encoding="utf-8") as file:
        scenarios = json.load(file)

    if not isinstance(scenarios, list):
        raise ValueError("case_scenarios.json must contain a list.")

    # Reset mock call counters once before the full scenario run.
    # Do not reset inside run_case: retries must advance defect counters.
    reset()

    results = []

    for scenario in scenarios:
        result = run_case(
            txn_id=scenario["txn_id"],
            fraud_probability=scenario["fraud_probability"],
            amount_inr=scenario["amount_inr"],
            question=scenario["question"],
        )
        results.append(result)

        print("\n" + "=" * 70)
        print(f"Transaction: {result['txn_id']}")
        print(f"Status: {result['status']}")
        print(f"Policy: {result['policy_id']}")
        print(f"Attempts: {result['attempts']}")

        if result.get("similarity_score") is not None:
            print(f"Similarity: {result['similarity_score']:.4f}")

        if result.get("validation_gate") is not None:
            print(f"Validation gate: {result['validation_gate']}")

        if result.get("reason"):
            print(f"Reason: {result['reason']}")

        print("Case note:")
        print(json.dumps(result.get("case_note"), indent=2))

    print("\n" + "=" * 70)
    print("CASE-NOTE SCENARIO SUMMARY")
    print(f"Scenarios processed: {len(results)}")
    print(
        "Successful notes:",
        sum(result["status"] == "success" for result in results),
    )
    print(
        "Safe defaults:",
        sum(result["status"] == "safe_default" for result in results),
    )
    print(
        "No policy match:",
        sum(result["status"] == "no_policy_match" for result in results),
    )

    return results


if __name__ == "__main__":
    run_scenarios()
