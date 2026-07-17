import os
import json
import pytest
from datasets.validator import DatasetValidator

def test_dataset_validator(tmp_path):
    # Valid json content
    valid_file = tmp_path / "valid.json"
    valid_data = [
        {"problem_description": "desc1", "gurobi_code": "code1"},
        {"problem_description": "desc2", "gurobi_code": "code2"}
    ]
    with open(valid_file, "w", encoding="utf-8") as f:
        json.dump(valid_data, f)

    # Invalid json content (missing description field)
    invalid_file = tmp_path / "invalid.json"
    invalid_data = [
        {"not_desc": "desc1"},
        {"problem_description": "desc2", "gurobi_code": "code2"}
    ]
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump(invalid_data, f)

    validator = DatasetValidator()
    
    valid, dups, errors = validator.validate_schema_and_duplicates(str(valid_file))
    assert valid is True
    assert dups == 0
    
    invalid, dups_inv, errors_inv = validator.validate_schema_and_duplicates(str(invalid_file))
    assert invalid is False
    assert len(errors_inv) > 0
