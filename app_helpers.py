def risk_summary(dataframe):
    """Return counts and amounts needed for the user-facing review summary."""
    risk_counts = dataframe["Risk_Level"].value_counts()
    review_rows = dataframe["Risk_Level"].isin(["HIGH", "MEDIUM"])
    return {
        "total": len(dataframe),
        "low": int(risk_counts.get("LOW", 0)),
        "medium": int(risk_counts.get("MEDIUM", 0)),
        "high": int(risk_counts.get("HIGH", 0)),
        "review_count": int(review_rows.sum()),
        "total_amount": float(dataframe["amount"].sum()),
        "review_amount": float(dataframe.loc[review_rows, "amount"].sum()),
    }
