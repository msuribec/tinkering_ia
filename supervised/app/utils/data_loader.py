from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
INSURANCE_DATA_PATH = DATA_RAW_DIR / "insurance.csv"
DIGITS_DATA_PATH = DATA_RAW_DIR / "digits.csv"


def _require_data_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}. "
            "Make sure the repository includes the /data/raw datasets."
        )
    return path


def load_insurance():
    """Load the committed Medical Cost Insurance dataset from the local repo."""
    return pd.read_csv(_require_data_file(INSURANCE_DATA_PATH))


def load_digits_data():
    """Load the committed Digits dataset from the local repo."""
    df = pd.read_csv(_require_data_file(DIGITS_DATA_PATH))
    if "digit" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"digit": "target"})

    feature_names = sorted(
        [col for col in df.columns if col.startswith("pixel_")],
        key=lambda name: int(name.split("_")[1]),
    )

    if not feature_names:
        raise ValueError("Digits CSV must contain pixel_0 ... pixel_63 columns.")
    if "target" not in df.columns:
        raise ValueError("Digits CSV must contain a 'target' column.")

    return df, feature_names
