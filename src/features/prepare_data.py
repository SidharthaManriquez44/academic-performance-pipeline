from sklearn.preprocessing import LabelEncoder

def prepare_data(df, kind="multiclass"):

    df = df.copy()

    # TARGET
    if kind == "multiclass":
        le = LabelEncoder()
        y = le.fit_transform(df["Grade"])

    elif kind == "binary":
        y = df["Grade"].isin(["D", "F"]).astype(int)
        le = None

    else:
        raise ValueError("kind must be 'binary' or 'multiclass'")

    # Drop columns
    X = df.drop(columns=[
        "Grade",
        "Student_ID",
        "First_Name",
        "Last_Name",
        "Email",
        "Final_Score",
        "Total_Score",
    ], errors="ignore")

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, numeric_features, categorical_features, le
