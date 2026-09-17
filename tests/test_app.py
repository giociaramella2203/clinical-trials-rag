import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import parse_age, check_eligibility, parse_study, build_agentic_prompt
import app as app_module


def test_parse_age_normal():
    assert parse_age("18 Years") == 18

def test_parse_age_none():
    assert parse_age(None) is None

def test_parse_age_no_digits():
    assert parse_age("N/A") is None


def test_check_eligibility_within_range():
    app_module.all_trials = [
        {"nct_id": "NCT00000001", "title": "Test Trial", "conditions": "test condition", "min_age": "18 Years", "max_age": "65 Years"}
    ]
    result = check_eligibility("NCT00000001", 30)
    assert result["eligible_by_age"] == True

def test_check_eligibility_too_old():
    app_module.all_trials = [
        {"nct_id": "NCT00000001", "title": "Test Trial", "conditions": "test condition", "min_age": "18 Years", "max_age": "65 Years"}
    ]
    result = check_eligibility("NCT00000001", 70)
    assert result["eligible_by_age"] == False

def test_check_eligibility_no_max_age():
    app_module.all_trials = [
        {"nct_id": "NCT00000001", "title": "Test Trial", "conditions": "test condition", "min_age": "18 Years", "max_age": None}
    ]
    result = check_eligibility("NCT00000001", 90)
    assert result["eligible_by_age"] == True

def test_check_eligibility_sex_mismatch():
    app_module.all_trials = [
        {"nct_id": "NCT00000001", "title": "Test Trial", "conditions": "test condition", "min_age": "18 Years", "max_age": "65 Years", "sex": "FEMALE"}
    ]
    result = check_eligibility("NCT00000001", 30, patient_sex="MALE")
    assert result["eligible_by_sex"] == False

def test_check_eligibility_sex_all_accepts_anyone():
    app_module.all_trials = [
        {"nct_id": "NCT00000001", "title": "Test Trial", "conditions": "test condition", "min_age": "18 Years", "max_age": "65 Years", "sex": "ALL"}
    ]
    result = check_eligibility("NCT00000001", 30, patient_sex="MALE")
    assert result["eligible_by_sex"] == True


def test_parse_study_extracts_fields():
    fake_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000003", "briefTitle": "Test Trial"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "conditionsModule": {"conditions": ["Test Condition"]},
            "eligibilityModule": {
                "eligibilityCriteria": "Must be an adult.",
                "sex": "ALL",
                "minimumAge": "18 Years",
                "maximumAge": "65 Years",
            },
        }
    }
    result = parse_study(fake_study)
    assert result["nct_id"] == "NCT00000003"
    assert result["title"] == "Test Trial"
    assert result["min_age"] == "18 Years"

def test_parse_study_handles_missing_fields():
    fake_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000004", "briefTitle": "Sparse Trial"},
            "statusModule": {"overallStatus": "RECRUITING"},
        }
    }
    result = parse_study(fake_study)
    assert result["nct_id"] == "NCT00000004"
    assert result["conditions"] == []
    assert result["max_age"] is None

def test_build_agentic_prompt_includes_query_and_context():
    fake_results = [
        {"nct_id": "NCT00000005", "title": "Example Trial", "eligibility_criteria": "Ages 18-50."}
    ]
    prompt = build_agentic_prompt("What trials exist for X?", fake_results)
    assert "What trials exist for X?" in prompt
    assert "NCT00000005" in prompt
    assert "Example Trial" in prompt