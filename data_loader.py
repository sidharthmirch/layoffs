import kagglehub
import os
import pandas as pd

DATASET_SLUG = "nudratabbas/global-company-layoffs-prediction-dataset"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (or use cached) the layoffs dataset and return (train_df, test_df)."""
    path = kagglehub.dataset_download(DATASET_SLUG)
    train_df = pd.read_csv(os.path.join(path, "layoffs/train.csv"))
    test_df = pd.read_csv(os.path.join(path, "layoffs/test.csv"))
    return train_df, test_df