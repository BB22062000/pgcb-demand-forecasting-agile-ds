# Q3(c)-2: Automated Test Script

import pandas as pd
from validation import validate_pgcb_data, validation_passed


def test_clean_pgcb_data_passes_validation():
    clean_data = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2025-01-01 00:00:00",
            "2025-01-01 01:00:00",
            "2025-01-01 02:00:00"
        ]),
        "demand_mw": [8000, 8200, 8100]
    })

    report = validate_pgcb_data(clean_data)

    assert validation_passed(report) is True


def test_invalid_demand_fails_validation():
    invalid_data = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2025-01-01 00:00:00",
            "2025-01-01 01:00:00"
        ]),
        "demand_mw": [8000, 0]
    })

    report = validate_pgcb_data(invalid_data)

    assert validation_passed(report) is False
