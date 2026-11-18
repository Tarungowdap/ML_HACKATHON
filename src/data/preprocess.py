import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler

FEATURE_COLS = [
    "perp_dist_m", "proj_ratio_t", "arc_length_m", "gaussian_emission",
    "heading_diff_rad", "cos_heading_diff", "sin_heading_diff", "dt_sec",
    "step_dist_m", "speed_mps", "accel_mps2", "is_stationary",
    "from_out_deg", "to_out_deg", "from_in_deg", "dist_rank",
    "dist_diff_from_best_m", "dist_ratio_to_best", "num_candidates",
]

def fit_and_transform_splits(train_df: pd.DataFrame, student_df: pd.DataFrame, artifact_path: str):
    scaler = RobustScaler()
    train_scaled = train_df.copy()
    student_scaled = student_df.copy()
    scaled_cols = [f"{c}_scaled" for c in FEATURE_COLS]
    train_scaled[scaled_cols] = scaler.fit_transform(train_df[FEATURE_COLS])
    student_scaled[scaled_cols] = scaler.transform(student_df[FEATURE_COLS])
    joblib.dump({"scaler": scaler, "feature_cols": FEATURE_COLS, "scaled_cols": scaled_cols}, artifact_path)
    return train_scaled, student_scaled
