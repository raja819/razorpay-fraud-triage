
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Project root: folder containing this Python file.
PROJECT_ROOT = Path(__file__).resolve().parent

POLICIES_PATH = PROJECT_ROOT / "fraud_policies.json"
QUERIES_PATH = PROJECT_ROOT / "retrieval_queries.json"

# A query must meet this score to receive a policy match.
# The threshold rejects unrelated questions with low similarity.
MIN_SCORE = 0.45


def load_policies():
    """Load and validate the policy records from JSON."""
    with POLICIES_PATH.open("r", encoding="utf-8") as file:
        policies = json.load(file)

    if not isinstance(policies, list) or not policies:
        raise ValueError("fraud_policies.json must contain a non-empty list.")

    for policy in policies:
        if not isinstance(policy, dict):
            raise ValueError("Every policy must be a JSON object.")

        if not policy.get("policy_id") or not policy.get("text"):
            raise ValueError("Every policy must contain policy_id and text.")

    return policies


# Load the policies once when this module is imported.
POLICIES = load_policies()

# Build the search index from policy text only.
# We do not include policy IDs or titles in the vectorizer.
VECTORIZER = TfidfVectorizer()
POLICY_TEXTS = [policy["text"] for policy in POLICIES]
POLICY_MATRIX = VECTORIZER.fit_transform(POLICY_TEXTS)


def retrieve(query):
    """
    Return (policy_id, policy_text, score) for the best policy.

    If the best similarity score is below MIN_SCORE, return
    (None, None, score) to indicate that no policy matched.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    query_vector = VECTORIZER.transform([query])
    scores = cosine_similarity(query_vector, POLICY_MATRIX)[0]

    # Index of the policy with the highest cosine similarity.
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])
    best_policy = POLICIES[best_index]

    # Reject low-scoring, potentially unrelated questions.
    if best_score < MIN_SCORE:
        return None, None, best_score

    return (
        best_policy["policy_id"],
        best_policy["text"],
        best_score,
    )


def run_retrieval_tests():
    """Run all provided queries and report matching results."""
    with QUERIES_PATH.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if not isinstance(queries, list):
        raise ValueError("retrieval_queries.json must contain a list.")

    correct_in_scope = 0
    total_in_scope = 0
    correct_out_of_scope = 0
    total_out_of_scope = 0

    print(f"MIN_SCORE: {MIN_SCORE:.2f}")
    print(f"Policies indexed: {len(POLICIES)}")
    print(f"Queries to test: {len(queries)}")
    print("-" * 90)

    for index, item in enumerate(queries, start=1):
        expected = item["expected_policy"]
        policy_id, policy_text, score = retrieve(item["query"])

        # Check whether the expected result matches the actual result.
        passed = policy_id == expected

        if expected is None:
            total_out_of_scope += 1
            if passed:
                correct_out_of_scope += 1
        else:
            total_in_scope += 1
            if passed:
                correct_in_scope += 1

        result = policy_id if policy_id is not None else "no_policy_match"
        status = "PASS" if passed else "FAIL"

        print(f"\nQuery {index}: {item['query']}")
        print(f"Expected: {expected}")
        print(f"Retrieved: {result}")
        print(f"Similarity: {score:.4f}")
        print(f"Result: {status}")

    print("\n" + "=" * 50)
    print("RETRIEVAL TEST SUMMARY")
    print("=" * 50)
    print(
        f"In-scope accuracy: "
        f"{correct_in_scope}/{total_in_scope}"
    )
    print(
        f"Out-of-scope rejection: "
        f"{correct_out_of_scope}/{total_out_of_scope}"
    )
    print(
        f"Overall expected results: "
        f"{correct_in_scope + correct_out_of_scope}/{len(queries)}"
    )

    if (
        correct_in_scope != total_in_scope
        or correct_out_of_scope != total_out_of_scope
    ):
        raise AssertionError(
            "Retrieval tests failed. Check the similarity scores "
            "and MIN_SCORE before proceeding."
        )

    print("\nAll retrieval tests passed.")


if __name__ == "__main__":
    run_retrieval_tests()
