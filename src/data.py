"""Loading, cleaning and splitting the Telco data."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, RANDOM_STATE, TEST_SIZE


def load_clean(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges is text with 11 blanks. All 11 have tenure == 0 (brand-new customers,
    # nothing billed yet), so 0 is the logical value. Decision is documented in MODEL_CARD.md.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    blank = df["TotalCharges"].isna()
    assert (df.loc[blank, "tenure"] == 0).all(), "Blank TotalCharges found for tenure > 0"
    df.loc[blank, "TotalCharges"] = 0.0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def split(df: pd.DataFrame):
    """Stratified 80/20 hold-out. The hold-out is evaluated ONCE, at the very end."""
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["Churn"],
                                   random_state=RANDOM_STATE)
    return train.reset_index(drop=True), test.reset_index(drop=True)
