from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from app_helpers import risk_summary


def test_risk_summary_counts_review_rows_and_amounts():
    dataframe = pd.DataFrame(
        {
            "Risk_Level": ["LOW", "MEDIUM", "HIGH", "HIGH"],
            "amount": [100.0, 250.0, 400.0, 600.0],
        }
    )

    assert risk_summary(dataframe) == {
        "total": 4,
        "low": 1,
        "medium": 1,
        "high": 2,
        "review_count": 3,
        "total_amount": 1350.0,
        "review_amount": 1250.0,
    }


def test_risk_summary_handles_missing_risk_levels():
    dataframe = pd.DataFrame(
        {
            "Risk_Level": ["LOW", "LOW"],
            "amount": [10.0, 20.0],
        }
    )

    summary = risk_summary(dataframe)

    assert summary["low"] == 2
    assert summary["medium"] == 0
    assert summary["high"] == 0
    assert summary["review_count"] == 0
