"""
Evaluation harness for the RAG pipeline. Unlike pytest tests, this makes
real LLM calls and checks end-to-end behavior — meant to be run manually,
not on every CI push.

Usage: uv run python eval.py
"""

import app as app_module
from app import rag_agentic, load_data, search_trials

app_module.load_data()


def check_retrieval_relevance(query, expected_keyword_in_conditions):
    results = search_trials(query, num_results=5)
    found = any(
        expected_keyword_in_conditions.lower() in r["conditions"].lower()
        for r in results
    )
    return found, f"Expected '{expected_keyword_in_conditions}' among top-5 conditions for '{query}'"


def check_eligibility_answer(nct_id, patient_age, patient_sex, expect_eligible, fake_trial):
    original_trials = app_module.all_trials
    app_module.all_trials = original_trials + [fake_trial]

    query = f"Is a {patient_age}-year-old {patient_sex.lower()} eligible for trial {nct_id}?"
    answer = rag_agentic(query)

    app_module.all_trials = original_trials

    answer_lower = answer.lower()
    negative_signals = ["not eligible", "ineligible", "does not meet", "not meet"]
    said_eligible = not any(signal in answer_lower for signal in negative_signals) and "eligible" in answer_lower
    passed = said_eligible == expect_eligible
    return passed, f"Query: '{query}' -> got answer indicating eligible={said_eligible}, expected {expect_eligible}"


eval_cases = []

# Category 1: retrieval relevance
eval_cases.append((
    "retrieval: memory loss -> Alzheimer's",
    lambda: check_retrieval_relevance("What trials exist for memory loss?", "Alzheimer"),
))
eval_cases.append((
    "retrieval: high blood sugar -> diabetes",
    lambda: check_retrieval_relevance("Trials for patients with high blood sugar", "diabetes"),
))

# Category 2: tool-use correctness with synthetic trials
FAKE_TRIAL_AGE = {
    "nct_id": "NCT99999901",
    "title": "Fake Eval Trial - Age Only",
    "status": "RECRUITING",
    "conditions": "eval testing",
    "eligibility_criteria": "Ages 18-65.",
    "sex": "ALL",
    "min_age": "18 Years",
    "max_age": "65 Years",
}
FAKE_TRIAL_SEX = {
    "nct_id": "NCT99999902",
    "title": "Fake Eval Trial - Female Only",
    "status": "RECRUITING",
    "conditions": "eval testing",
    "eligibility_criteria": "Ages 18-65, female only.",
    "sex": "FEMALE",
    "min_age": "18 Years",
    "max_age": "65 Years",
}

eval_cases.append((
    "eligibility: age within range -> eligible",
    lambda: check_eligibility_answer("NCT99999901", 30, "woman", True, FAKE_TRIAL_AGE),
))
eval_cases.append((
    "eligibility: age above range -> not eligible",
    lambda: check_eligibility_answer("NCT99999901", 80, "man", False, FAKE_TRIAL_AGE),
))
eval_cases.append((
    "eligibility: sex mismatch -> not eligible",
    lambda: check_eligibility_answer("NCT99999902", 30, "man", False, FAKE_TRIAL_SEX),
))
eval_cases.append((
    "eligibility: sex match -> eligible",
    lambda: check_eligibility_answer("NCT99999902", 30, "woman", True, FAKE_TRIAL_SEX),
))


def run_eval():
    passed_count = 0
    print(f"Running {len(eval_cases)} eval cases...\n")
    for name, case_fn in eval_cases:
        try:
            passed, detail = case_fn()
        except Exception as e:
            passed, detail = False, f"Exception: {e}"
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        print(f"       {detail}\n")
        if passed:
            passed_count += 1

    print(f"Result: {passed_count}/{len(eval_cases)} passed")


if __name__ == "__main__":
    run_eval()