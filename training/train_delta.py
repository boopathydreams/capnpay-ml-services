"""Delta model training using parity features and warm start from Champion.

CLI usage:
  cd ml-services
  python -m training.train_delta --recent_csv data/recent.csv --replay_csv data/full.csv
"""
from pathlib import Path
from typing import Dict, Any
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.features import Encoders, prepare_training_data, validate_features
from models.calibration import MulticlassPlatt


def load_champion(champion_dir: Path):
    model = xgb.Booster()
    model.load_model(str(champion_dir / "xgb_model.json"))
    enc = Encoders.load(champion_dir / "encoders.pkl")
    categories = json.loads((champion_dir / "categories.json").read_text())
    return model, enc, categories


def train_delta(recent_df: pd.DataFrame, replay_df: pd.DataFrame, champion_dir: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    champ_model, enc, categories = load_champion(champion_dir)

    # Combine
    recent_df = recent_df.copy(); recent_df["weight"] = 2.0
    replay_df = replay_df.copy(); replay_df["weight"] = 1.0
    df = pd.concat([recent_df, replay_df], ignore_index=True)

    X_df, y = prepare_training_data(df, enc)
    validate_features(X_df, enc.feature_spec)
    w = df["weight"].values if "weight" in df.columns else None
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_df.values, y.values, w, test_size=0.2, random_state=42, stratify=y.values
    )

    params = {
        "objective": "multi:softprob",
        "num_class": len(categories),
        "eval_metric": "mlogloss",
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "tree_method": "hist",
        "random_state": 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
    delta_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dval, "val")],
        xgb_model=champ_model,
        verbose_eval=False,
    )

    # Calibrate
    val_probs = delta_model.predict(dval)
    calibrator = MulticlassPlatt(); calibrator.fit(val_probs, y_val)

    # Save
    (out_dir / "xgb_model.json").write_text("")  # ensure file created if save fails
    delta_model.save_model(str(out_dir / "xgb_model.json"))
    enc.save(out_dir / "encoders.pkl")
    (out_dir / "calibrator.pkl").write_bytes((lambda c: (c.save(out_dir / "calibrator.pkl") or b""))(calibrator) or b"")
    (out_dir / "categories.json").write_text(json.dumps(categories))

    acc = float(accuracy_score(y_val, np.argmax(val_probs, axis=1)))
    return {"accuracy": acc, "samples": len(df)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Delta model with warm start from Champion")
    parser.add_argument("--recent_csv", required=True, help="Path to recent labeled CSV")
    parser.add_argument("--replay_csv", required=True, help="Path to full labeled CSV for replay sampling")
    parser.add_argument("--champion_dir", default="model_artifacts/champion", help="Champion artifacts directory")
    parser.add_argument("--out_dir", default="model_artifacts/delta", help="Output directory for Delta artifacts")
    args = parser.parse_args()

    r = pd.read_csv(args.recent_csv)
    f = pd.read_csv(args.replay_csv)
    res = train_delta(r, f, Path(args.champion_dir), Path(args.out_dir))
    print("Delta metrics:", res)
