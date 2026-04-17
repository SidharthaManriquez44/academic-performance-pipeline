def validate_data(df):
    print("Validating data...")

    if df.isnull().sum().sum() > 0:
        raise ValueError("There are null values in the data")

    if "Total_Score" not in df.columns:
        raise ValueError("A column named 'Total_Score' is missing")

    print("Validation passed")
    return True
