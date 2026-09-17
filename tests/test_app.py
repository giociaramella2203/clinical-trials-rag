import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import parse_age, check_eligibility, parse_study, build_agentic_prompt


def test_parse_age_normal():
    assert parse_age("18 Years") == 18

def test_parse_age_none():
    assert parse_age(None) is None

def test_parse_age_no_digits():
    assert parse_age("N/A") is None

import app as app_module

def test_check_eligibility_within_range():
    app_module.all_trials = [
        {"nct_id": "NCT00000001", "min_age": "18 Years", "max_age": "65 Years"}
    ]
    
    result = check_eligibility("NCT00000001", 30)
    
    assert result["eligible_by_age"] == True

def test_check_eligibility_too_old():
    app_module.all_trials = [
        {"nct_id": "NCT00000001", "min_age": "18 Years", "max_age": "65 Years"}
    ]
    
    result = check_eligibility("NCT00000001", 70)
    
    assert result["eligible_by_age"] == False

def test_check_eligibility_no_max_age():
    app_module.all_trials = [
        {"nct_id": "NCT00000001", "min_age": "18 Years", "max_age": None}
    ]

    result = check_eligibility("NCT00000001", 90)

    assert result["eligible_by_age"] == True

def test_check_eligibility_sex_mismatch():
    app_module.all_trials = [
        {"nct_id": "NCT00000001", "min_age": "18 Years", "max_age": "65 Years", "sex": "FEMALE"}
    ]

    result = check_eligibility("NCT00000001", 30, patient_sex="MALE")

    assert result["eligible_by_sex"] == False

def test_check_eligibility_sex_all_accepts_anyone():
    app_module.all_trials = [
        {"nct_id": "NCT00000001", "min_age": "18 Years", "max_age": "65 Years", "sex": "ALL"}
    ]

    result = check_eligibility("NCT00000001", 30, patient_sex="MALE")

    assert result["eligible_by_sex"] == True
