"""
Champion-Delta ensemble for ml-services.

Loads champion and delta artifact bundles (model, encoders, calibrator, categories)
and provides blended predictions with calibrated probabilities.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb

from .features import Encoders, prepare_training_data, validate_features
from .calibration import MulticlassPlatt


@dataclass
class EnsembleConfig:
    alpha: float = 0.7  # champion weight
    confidence_threshold: float = 0.6


class ModelArtifacts:
    def __init__(self, artifacts_dir: Path):
        self.dir = artifacts_dir
        self.model: Optional[xgb.Booster] = None
        self.encoders: Optional[Encoders] = None
        self.calibrator: Optional[MulticlassPlatt] = None
        self.categories: List[str] = []
        self.load()

    def load(self):
        model_path = self.dir / "xgb_model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        self.encoders = Encoders.load(self.dir / "encoders.pkl")
        cal_path = self.dir / "calibrator.pkl"
        if cal_path.exists():
            self.calibrator = MulticlassPlatt.load(cal_path)
        cat_path = self.dir / "categories.json"
        if cat_path.exists():
            try:
                self.categories = json.loads(cat_path.read_text())
            except Exception:
                self.categories = []

    def predict_proba(self, raw_df: pd.DataFrame) -> np.ndarray:
        X_df, _ = prepare_training_data(raw_df, self.encoders)
        validate_features(X_df, self.encoders.feature_spec)
        dmat = xgb.DMatrix(X_df.values)
        probs = self.model.predict(dmat)
        if self.calibrator is not None:
            probs = self.calibrator.transform(probs)
        return probs


class ChampionDeltaEnsemble:
    def __init__(self, champion_dir: Path, delta_dir: Optional[Path] = None, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.champion = ModelArtifacts(champion_dir)
        self.delta = ModelArtifacts(delta_dir) if delta_dir and (delta_dir / "xgb_model.json").exists() else None

    def predict(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        if raw_df.shape[0] != 1:
            raise ValueError("predict expects single-row DataFrame")
        t0 = time.time()
        champ = self.champion.predict_proba(raw_df)[0]
        if self.delta is not None:
            try:
                delt = self.delta.predict_proba(raw_df)[0]
                probs = self.config.alpha * champ + (1 - self.config.alpha) * delt
                source = "ensemble"
            except Exception:
                probs = champ
                source = "champion"
        else:
            probs = champ
            source = "champion"

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        cats = self.champion.categories or []
        category_name = cats[top_idx] if cats and top_idx < len(cats) else str(top_idx)
        topk_idx = np.argsort(probs)[::-1][:3]
        topk = [
            {
                "category": (cats[i] if cats and i < len(cats) else str(i)),
                "confidence": float(probs[i]),
            }
            for i in topk_idx
        ]
        return {
            "category": category_name,
            "confidence": confidence,
            "topk": topk,
            "prediction_source": source,
            "meets_threshold": confidence >= self.config.confidence_threshold,
        }

