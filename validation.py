# Q3(c)-1: Validation Script for CI/CD Testing

import pandas as pd


def validate_pgcb_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return validation status for the PGCB modelling dataset."""

    required_columns = ["datetime", "demand_mw"]
    report = []

    missing_required = [col for col in required_columns if col not in data.columns]
    report.append({
        "Validation Check": "Required columns",
        "Result": len(missing_required),
        "Status": "PASS" if len(missing_required) == 0 else "FAIL"
    })

    missing_cells = int(data.isna().sum().sum())
    report.append({
        "Validation Check": "Missing values",
        "Result": missing_cells,
        "Status": "PASS" if missing_cells == 0 else "FAIL"
    })

    duplicate_rows = int(data.duplicated().sum())
    report.append({
        "Validation Check": "Duplicate rows",
        "Result": duplicate_rows,
        "Status": "PASS" if duplicate_rows == 0 else "FAIL"
    })

    duplicate_times = int(data.duplicated(subset=["datetime"]).sum())
    report.append({
        "Validation Check": "Duplicate timestamps",
        "Result": duplicate_times,
        "Status": "PASS" if duplicate_times == 0 else "FAIL"
    })

    invalid_demand = int((data["demand_mw"] <= 0).sum())
    report.append({
        "Validation Check": "Invalid demand values",
        "Result": invalid_demand,
        "Status": "PASS" if invalid_demand == 0 else "FAIL"
    })

    return pd.DataFrame(report)


def validation_passed(report: pd.DataFrame) -> bool:
    """Return True only when all validation checks pass."""
    return bool((report["Status"] == "PASS").all())
