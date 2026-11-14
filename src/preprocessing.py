import joblib
import pandas as pd

from src.config_loader import load_config, resolve_path


class Preprocessor:
    """Loads fitted objects once. Only transforms, never fits."""

    def __init__(self):
        cfg = load_config()
        paths = cfg["transformers"]
        self.numeric_cols = cfg["features"]["numeric_cols"]
        self.categorical_col = cfg["features"]["categorical_col"]

        self.imputer = joblib.load(resolve_path(paths["imputer_path"]))
        self.encoder = joblib.load(resolve_path(paths["encoder_path"]))
        self.scaler = joblib.load(resolve_path(paths["scaler_path"]))
        self.feature_list = (
            resolve_path(paths["feature_list_path"]).read_text().splitlines()
        )

    def transform(self, feats: pd.DataFrame) -> pd.DataFrame:
        out = feats.copy()

        # 1. impute numeric columns
        out[self.numeric_cols] = self.imputer.transform(out[self.numeric_cols])

        # 2. one-hot encode state (unknown states become all zeros)
        encoded = self.encoder.transform(out[[self.categorical_col]])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out([self.categorical_col]),
            index=out.index,
        )
        out = pd.concat([out.drop(columns=[self.categorical_col]), encoded_df], axis=1)

        # 3. scale numeric columns
        out[self.numeric_cols] = self.scaler.transform(out[self.numeric_cols])

        # 4. exact column order the model was trained on
        return out[self.feature_list]