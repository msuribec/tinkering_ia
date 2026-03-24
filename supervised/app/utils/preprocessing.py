import pandas as pd


INSURANCE_NUMERIC_COLUMNS = ["age", "bmi", "children"]
INSURANCE_REQUIRED_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
]
INSURANCE_REGION_LEVELS = ["northeast", "northwest", "southeast", "southwest"]
INSURANCE_REGION_DUMMY_COLUMNS = [
    "region_northwest",
    "region_southeast",
    "region_southwest",
]
INSURANCE_FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    *INSURANCE_REGION_DUMMY_COLUMNS,
]


def _normalize_categories(series):
    return series.astype(str).str.strip().str.lower()


def _validate_required_columns(df, required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def transform_insurance_features(df):
    """Create a fixed regression feature matrix for insurance inputs."""
    _validate_required_columns(df, INSURANCE_REQUIRED_COLUMNS)

    clean_df = df.copy()
    features = pd.DataFrame(index=clean_df.index)

    for column in INSURANCE_NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(clean_df[column], errors="raise")

    sex = _normalize_categories(clean_df["sex"])
    smoker = _normalize_categories(clean_df["smoker"])
    region = _normalize_categories(clean_df["region"])

    sex_map = {"female": 0, "male": 1}
    smoker_map = {"no": 0, "yes": 1}

    features["sex"] = sex.map(sex_map)
    features["smoker"] = smoker.map(smoker_map)

    invalid_sex = sorted(sex[features["sex"].isna()].unique())
    if invalid_sex:
        raise ValueError(
            f"Invalid values in 'sex': {', '.join(invalid_sex)}. "
            "Expected only 'male' or 'female'."
        )

    invalid_smoker = sorted(smoker[features["smoker"].isna()].unique())
    if invalid_smoker:
        raise ValueError(
            f"Invalid values in 'smoker': {', '.join(invalid_smoker)}. "
            "Expected only 'yes' or 'no'."
        )

    region_categorical = pd.Categorical(region, categories=INSURANCE_REGION_LEVELS)
    invalid_region_mask = pd.isna(region_categorical)
    invalid_regions = sorted(region[invalid_region_mask].unique())
    if invalid_regions:
        raise ValueError(
            f"Invalid values in 'region': {', '.join(invalid_regions)}. "
            f"Expected one of: {', '.join(INSURANCE_REGION_LEVELS)}."
        )

    region_dummies = pd.get_dummies(region_categorical, prefix="region", dtype=int)
    region_dummies = region_dummies.reindex(
        columns=[f"region_{name}" for name in INSURANCE_REGION_LEVELS],
        fill_value=0,
    )
    region_dummies = region_dummies.drop(columns=["region_northeast"])

    features["sex"] = features["sex"].astype(int)
    features["smoker"] = features["smoker"].astype(int)
    features = pd.concat(
        [features[["age", "sex", "bmi", "children", "smoker"]], region_dummies],
        axis=1,
    )
    return features[INSURANCE_FEATURE_COLUMNS]


def preprocess_insurance(df, require_target=True):
    """Preprocess insurance data using the same fixed schema for train and inference."""
    X = transform_insurance_features(df)
    y = None
    if require_target:
        if "charges" not in df.columns:
            raise ValueError("Missing required target column: charges")
        y = pd.to_numeric(df["charges"], errors="raise")
    return X, y, INSURANCE_FEATURE_COLUMNS.copy()


def preprocess_digits(df, feature_names):
    """Preprocess Digits data loaded from the committed local CSV."""
    target_column = "target" if "target" in df.columns else "digit"
    X = df[feature_names].copy()
    y = df[target_column].copy()
    return X, y, list(feature_names)
