"""
Feature engineering module with train/serve parity for ml-services.

Provides:
- FeatureSpec: schema definition (columns, types)
- Encoders: label/scale encoders with save/load
- build_features: raw -> engineered features (merchant + catalog priors)
- prepare_training_data: apply scalers/encoders and coerce dtypes
- validate_features: schema/dtype/null checks
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import hashlib
import json
import pickle
import re
from pathlib import Path
import logging

try:
    from core.community_labels import get_label
except Exception:

    def get_label(*args, **kwargs):
        return None


logger = logging.getLogger(__name__)


@dataclass
class FeatureSpec:
    feature_columns: List[str]
    categorical_features: List[str]
    numerical_features: List[str]
    derived_features: List[str]
    feature_dtypes: Dict[str, str]
    version: str = "1.1"


class Encoders:
    def __init__(self):
        self.vpa_handle_encoder = LabelEncoder()
        self.merchant_name_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.amount_scaler = StandardScaler()
        self.feature_spec: Optional[FeatureSpec] = None
        self.category_mapping: Dict[int, str] = {}

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "Encoders":
        with open(path, "rb") as f:
            return pickle.load(f)


def _normalize_name(text: str) -> str:
    if not isinstance(text, str) or not text:
        return "unknown"
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", s).strip()


def _extract_vpa_handle(vpa: str) -> str:
    if not isinstance(vpa, str) or "@" not in vpa:
        return "unknown"
    return vpa.split("@")[-1].lower()


def _safe_label_transform(
    series: pd.Series, le: LabelEncoder, unk: str = "<UNK>"
) -> pd.Series:
    values = series.astype(str).copy()
    classes = set(le.classes_.tolist()) if hasattr(le, "classes_") else set()
    if unk not in classes:
        # Extend encoder classes (train-time should include unk already; this is a guard)
        if hasattr(le, "classes_"):
            le.classes_ = np.concatenate([le.classes_, np.array([unk])])
    classes = set(le.classes_.tolist()) if hasattr(le, "classes_") else {unk}
    mask = ~values.isin(classes)
    if mask.any():
        values.loc[mask] = unk
    return pd.Series(le.transform(values), index=series.index)


def _load_merchant_catalog() -> Optional[pd.DataFrame]:
    try:
        path = Path(__file__).resolve().parent.parent / "data" / "merchant_category.csv"
        if not path.exists():
            logger.info("Merchant catalog not found; skipping priors")
            return None
        df = pd.read_csv(path)
        df["normalized_name"] = df["Merchant Name"].apply(_normalize_name)
        # Map vendor categories to canonical buckets used in ml-services (align with business taxonomy)
        canon_map = {
            "food & dining": "Food & Dining",
            "shopping": "Shopping",
            "transport": "Transport",
            "healthcare": "Healthcare",
            "healthcare & pharmacy": "Healthcare",
            "bills": "Bills",
            "bills & utilities": "Bills",
            "personal": "Personal",
        }
        df["canonical_category"] = df["Category"].astype(str).str.lower().map(canon_map)
        return df
    except Exception as e:
        logger.warning(f"Failed loading merchant catalog: {e}")
        return None


def build_features(
    df: pd.DataFrame, encoders: Optional[Encoders] = None
) -> pd.DataFrame:
    features = df.copy()

    # Amount
    features["log_amount"] = np.log1p(features["amount"].clip(lower=0))

    # Time (expect timestamp or hour/day)
    if "timestamp" in features.columns:
        ts = pd.to_datetime(features["timestamp"], errors="coerce")
        features["hour"] = ts.dt.hour
        features["day_of_week"] = ts.dt.dayofweek
    if "hour" in features.columns:
        hour = features["hour"].fillna(0).astype(int)
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    if "day_of_week" in features.columns:
        dow = features["day_of_week"].fillna(0).astype(int)
        features["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
        features["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)

    # VPA handle id
    if "vpa" in features.columns:
        features["vpa_handle"] = features["vpa"].apply(_extract_vpa_handle)
    else:
        features["vpa_handle"] = "unknown"
    if encoders and hasattr(encoders.vpa_handle_encoder, "classes_"):
        features["vpa_handle_id"] = _safe_label_transform(
            features["vpa_handle"], encoders.vpa_handle_encoder
        )
    else:
        features["vpa_handle_id"] = features["vpa_handle"]

    # Merchant token/id
    features["merchant_token"] = features.get(
        "merchant_name", features.get("payee_name", "")
    ).apply(_normalize_name)
    if encoders and hasattr(encoders.merchant_name_encoder, "classes_"):
        features["merchant_id"] = _safe_label_transform(
            features["merchant_token"], encoders.merchant_name_encoder
        )
    else:
        features["merchant_id"] = features["merchant_token"]

    # Simple derived
    features["is_round_amount"] = (
        (features["amount"] % 50 == 0) | (features["amount"] % 100 == 0)
    ).astype(int)

    # Optional P2P/memo features (lightweight keyword flags)
    memo = features.get("memo", "")
    if isinstance(memo, pd.Series):
        memo_l = memo.astype(str).str.lower()
    else:
        memo_l = pd.Series(["" for _ in range(len(features))])

    def has_kw(s: pd.Series, kw: str) -> pd.Series:
        return s.str.contains(rf"\b{re.escape(kw)}\b", regex=True)

    features["memo_has_loan"] = (
        has_kw(memo_l, "loan") | has_kw(memo_l, "lend") | has_kw(memo_l, "borrow")
    ).astype(int)
    features["memo_has_repay"] = (
        has_kw(memo_l, "repay") | has_kw(memo_l, "return")
    ).astype(int)
    features["memo_has_gift"] = has_kw(memo_l, "gift").astype(int)
    features["memo_has_tip"] = has_kw(memo_l, "tip").astype(int)
    # Direction if present: 'send'/'receive'
    if "direction" in features.columns:
        dir_l = features["direction"].astype(str).str.lower()
        features["direction_send"] = (dir_l == "send").astype(int)
        features["direction_receive"] = (dir_l == "receive").astype(int)
    else:
        features["direction_send"] = 0
        features["direction_receive"] = 0

    # Default user features (can be overridden by feature store)
    defaults = {
        "user_food_ratio_30d": 0.25,
        "user_transport_ratio_30d": 0.15,
        "user_shopping_ratio_30d": 0.30,
        "user_avg_transaction_amount": 500.0,
        "user_transaction_frequency_score": 0.5,
        "payee_seen_before": 0,
        "merchant_known": 0,
        "caps_state": "OK",
    }
    for k, v in defaults.items():
        if k not in features.columns:
            features[k] = v
    # caps state int
    cap_map = {"OK": 0, "Near": 1, "Over": 2}
    features["caps_state_int"] = (
        features["caps_state"].map(cap_map).fillna(0).astype(int)
    )

    # Merchant catalog priors
    catalog = _load_merchant_catalog()
    features["catalog_category_id"] = 0.0
    features["catalog_confidence"] = 0.0
    if catalog is not None:
        idx = features["merchant_token"].map(_normalize_name)
        # Some normalized names may appear multiple times; collapse to first
        m = (
            catalog.set_index("normalized_name")["canonical_category"]
            .groupby(level=0)
            .first()
        )
        # Map directly to a series aligned with feature rows
        mapped_canon = idx.map(
            m
        )  # index = features index, values = canonical_category or NaN
        if mapped_canon.notna().any():
            canon_ids = {
                "Food & Dining": 1.0,
                "Shopping": 2.0,
                "Transport": 3.0,
                "Healthcare": 4.0,
                "Personal": 5.0,
                "Bills": 6.0,
            }
            cat_ids = mapped_canon.map(canon_ids).fillna(0.0)
            features.loc[:, "catalog_category_id"] = cat_ids.values
            hit_mask = cat_ids > 0
            features.loc[hit_mask, "catalog_confidence"] = 0.95
            features.loc[hit_mask, "merchant_known"] = 1
            features.loc[hit_mask, "payee_seen_before"] = features.loc[
                hit_mask, "payee_seen_before"
            ].where(features.loc[hit_mask, "payee_seen_before"] > 0, 1)

    # Community label priors (strong hint for local merchants)
    try:
        canon_ids = {
            "Food & Dining": 1.0,
            "Shopping": 2.0,
            "Transport": 3.0,
            "Healthcare": 4.0,
            "Personal": 5.0,
            "Bills": 6.0,
        }
        for i in range(len(features)):
            lbl = get_label(
                str(features.at[i, "merchant_token"]),
                (
                    str(features.at[i, "vpa_handle"])
                    if "vpa_handle" in features.columns
                    else ""
                ),
            )
            if lbl and float(lbl.get("confidence", 0.0)) >= 0.8:
                features.at[i, "catalog_category_id"] = canon_ids.get(
                    lbl.get("category_name", ""), features.at[i, "catalog_category_id"]
                )
                features.at[i, "catalog_confidence"] = max(
                    0.95, float(lbl.get("confidence", 0.9))
                )
                features.at[i, "merchant_known"] = 1
                if features.at[i, "payee_seen_before"] <= 0:
                    features.at[i, "payee_seen_before"] = 1
    except Exception:
        pass

    # One-hot flags from catalog id
    cid = features["catalog_category_id"].fillna(0.0)
    features["catalog_is_food"] = (cid == 1.0).astype(float)
    features["catalog_is_shopping"] = (cid == 2.0).astype(float)
    features["catalog_is_transport"] = (cid == 3.0).astype(float)
    features["catalog_is_healthcare"] = (cid == 4.0).astype(float)
    features["catalog_is_personal"] = (cid == 5.0).astype(float)
    features["catalog_is_bills"] = (cid == 6.0).astype(float)

    # Novelty feature: high when merchant has no catalog prior and user hasn't seen payee
    # Additionally consider it novel if the merchant token is unknown to the encoder (when available)
    try:
        unknown_enc_series = pd.Series(False, index=features.index)
        if encoders is not None and hasattr(encoders.merchant_name_encoder, "classes_"):
            known = set([str(x) for x in encoders.merchant_name_encoder.classes_.tolist()])
            unknown_enc_series = ~features["merchant_token"].astype(str).isin(known)
        features["novelty_score"] = (
            ((features["catalog_confidence"].fillna(0.0) <= 0.0) & (features["payee_seen_before"].fillna(0) <= 0))
            | unknown_enc_series
        ).astype(float)
    except Exception:
        features["novelty_score"] = 0.0

    return features


def fit_encoders(df: pd.DataFrame, categories: List[str]) -> Encoders:
    enc = Encoders()
    feats = build_features(df)

    # Fit label encoders with <UNK>
    vpa_handles = feats["vpa_handle"].astype(str).unique().tolist() + ["<UNK>"]
    enc.vpa_handle_encoder.fit(vpa_handles)
    feats["vpa_handle_id"] = enc.vpa_handle_encoder.transform(feats["vpa_handle"])

    merchants = feats["merchant_token"].astype(str).unique().tolist() + ["<UNK>"]
    enc.merchant_name_encoder.fit(merchants)
    feats["merchant_id"] = enc.merchant_name_encoder.transform(feats["merchant_token"])

    enc.category_encoder.fit(categories)
    enc.category_mapping = {i: c for i, c in enumerate(categories)}

    # Amount scaler
    enc.amount_scaler.fit(feats[["log_amount"]])

    feature_columns = [
        "log_amount",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "vpa_handle_id",
        "merchant_id",
        "caps_state_int",
        "is_round_amount",
        "user_food_ratio_30d",
        "user_transport_ratio_30d",
        "user_shopping_ratio_30d",
        "user_avg_transaction_amount",
        "user_transaction_frequency_score",
        "payee_seen_before",
        "merchant_known",
        "catalog_category_id",
        "catalog_confidence",
        "novelty_score",
        "catalog_is_food",
        "catalog_is_shopping",
        "catalog_is_transport",
        "catalog_is_healthcare",
        "catalog_is_personal",
        "catalog_is_bills",
        # P2P/memo flags
        "memo_has_loan",
        "memo_has_repay",
        "memo_has_gift",
        "memo_has_tip",
        "direction_send",
        "direction_receive",
    ]

    categorical = ["vpa_handle_id", "merchant_id", "caps_state_int"]
    numerical = [c for c in feature_columns if c not in categorical]
    feature_dtypes = {
        c: ("int64" if c in categorical else "float64") for c in feature_columns
    }

    enc.feature_spec = FeatureSpec(
        feature_columns=feature_columns,
        categorical_features=categorical,
        numerical_features=numerical,
        derived_features=[
            "log_amount",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "is_round_amount",
            "novelty_score",
        ],
        feature_dtypes=feature_dtypes,
    )
    return enc


def prepare_training_data(
    df: pd.DataFrame, enc: Encoders
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    feats = build_features(df, enc)

    # Scale log amount
    if "log_amount" in feats.columns:
        feats["log_amount"] = enc.amount_scaler.transform(
            feats[["log_amount"]]
        ).flatten()

    # Coerce dtypes
    for c in enc.feature_spec.categorical_features:
        if c in feats.columns:
            feats[c] = (
                pd.to_numeric(feats[c], errors="coerce").fillna(0).astype("int64")
            )
    for c in enc.feature_spec.numerical_features:
        if c in feats.columns:
            feats[c] = (
                pd.to_numeric(feats[c], errors="coerce").fillna(0.0).astype("float64")
            )

    X = feats[enc.feature_spec.feature_columns].copy()
    y = None
    if "category" in df.columns:
        y = pd.Series(enc.category_encoder.transform(df["category"]))
    return X, y


def validate_features(X: pd.DataFrame, spec: FeatureSpec) -> bool:
    missing = set(spec.feature_columns) - set(X.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    extra = set(X.columns) - set(spec.feature_columns)
    if extra:
        logger.debug(f"Extra columns ignored: {extra}")
    for c in spec.feature_columns:
        expected = spec.feature_dtypes[c]
        if str(X[c].dtype) != expected:
            logger.warning(
                f"Column {c} dtype mismatch: expected {expected}, got {X[c].dtype}"
            )
    # Nulls
    if X[spec.feature_columns].isnull().sum().sum() > 0:
        raise ValueError("Nulls present in feature matrix")
    return True
