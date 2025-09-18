"""
Advanced XGBoost Auto-Tagger with 85+ Features
Targets 80%+ accuracy using comprehensive feature engineering
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import mlflow
import mlflow.xgboost
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
import pickle

from .advanced_features import AdvancedFeatureEngineer
from .features import Encoders
from .calibration import MulticlassPlatt

logger = logging.getLogger(__name__)


class AdvancedXGBoostTagger:
    """
    Advanced XGBoost payment categorization with 85+ features
    Targets 80%+ accuracy with sophisticated feature engineering
    """

    def __init__(self):
        self.model = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.encoders = None
        self.calibrator = None
        self.feature_importance = None

        # Advanced XGBoost hyperparameters (tuned for 85+ features)
        self.xgb_params = {
            "objective": "multi:softprob",
            "eval_metric": ["mlogloss", "merror"],
            "max_depth": 10,  # Deeper for more complex patterns
            "min_child_weight": 5,  # Higher to prevent overfitting
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "colsample_bylevel": 0.8,  # Additional regularization
            "learning_rate": 0.08,  # Slower learning for stability
            "n_estimators": 300,  # More trees for complex features
            "reg_alpha": 0.1,  # L1 regularization
            "reg_lambda": 1.0,  # L2 regularization
            "random_state": 42,
            "tree_method": "hist",
            "max_leaves": 256,  # Control tree complexity
            "grow_policy": "lossguide",  # Better splits for complex data
        }

        # Category mapping aligned with business requirements
        self.categories = [
            "Food & Dining",
            "Transport",
            "Shopping",
            "Bills & Utilities",
            "Entertainment",
            "Healthcare",
            "Personal",
            "Other",
        ]

    def prepare_advanced_features(
        self, df: pd.DataFrame, fit_encoders: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare 85+ advanced features for training/prediction"""

        # Create all advanced features
        features_df = self.feature_engineer.create_all_advanced_features(df)

        if fit_encoders:
            # Fit encoders during training
            self.encoders = self._fit_advanced_encoders(features_df, self.categories)

        if self.encoders is None:
            raise ValueError(
                "Encoders not fitted. Call with fit_encoders=True during training."
            )

        # Apply encoders
        X, y = self._apply_encoders(features_df, df)

        return X, y

    def _fit_advanced_encoders(
        self, features_df: pd.DataFrame, categories: List[str]
    ) -> Encoders:
        """Fit encoders for advanced features"""
        encoders = Encoders()

        # Fit label encoders for categorical features
        merchant_tokens = features_df["merchant_token"].astype(
            str
        ).unique().tolist() + ["<UNK>"]
        encoders.merchant_name_encoder.fit(merchant_tokens)

        vpa_handles = features_df["vpa_handle"].astype(str).unique().tolist() + [
            "<UNK>"
        ]
        encoders.vpa_handle_encoder.fit(vpa_handles)

        # Category encoder
        encoders.category_encoder.fit(categories)
        encoders.category_mapping = {i: c for i, c in enumerate(categories)}

        # Amount scaler
        encoders.amount_scaler.fit(features_df[["log_amount"]])

        # Store feature specification
        encoders.feature_spec = self.feature_engineer.create_advanced_feature_spec(
            categories
        )

        return encoders

    def _apply_encoders(
        self, features_df: pd.DataFrame, original_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Apply fitted encoders to features"""

        # Encode categorical features
        features_df["merchant_token_encoded"] = self._safe_transform(
            features_df["merchant_token"], self.encoders.merchant_name_encoder
        )
        features_df["vpa_handle_encoded"] = self._safe_transform(
            features_df["vpa_handle"], self.encoders.vpa_handle_encoder
        )

        # Scale numerical features
        if "log_amount" in features_df.columns:
            features_df["log_amount_scaled"] = self.encoders.amount_scaler.transform(
                features_df[["log_amount"]]
            ).flatten()

        # Select final feature set
        feature_columns = self.encoders.feature_spec.feature_columns.copy()

        # Replace original columns with encoded versions
        if "merchant_token" in feature_columns:
            feature_columns[feature_columns.index("merchant_token")] = (
                "merchant_token_encoded"
            )
        if "vpa_handle" in feature_columns:
            feature_columns[feature_columns.index("vpa_handle")] = "vpa_handle_encoded"
        if "log_amount" in feature_columns:
            feature_columns[feature_columns.index("log_amount")] = "log_amount_scaled"

        # Ensure all features exist
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0

        X = features_df[feature_columns].copy()

        # Handle target variable
        y = None
        if "category" in original_df.columns:
            y = pd.Series(
                self.encoders.category_encoder.transform(original_df["category"])
            )

        # Final data type conversion
        X = self._ensure_dtypes(X)

        return X, y

    def _safe_transform(self, series: pd.Series, encoder: LabelEncoder) -> pd.Series:
        """Safely transform categorical data with unknown handling"""
        values = series.astype(str).copy()
        known_classes = set(encoder.classes_)

        # Replace unknown values with <UNK>
        mask = ~values.isin(known_classes)
        values.loc[mask] = "<UNK>"

        return pd.Series(encoder.transform(values), index=series.index)

    def _ensure_dtypes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure proper data types for XGBoost"""
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            elif X[col].dtype == "bool":
                X[col] = X[col].astype(int)

        # Fill any remaining NaNs
        X = X.fillna(0)

        return X

    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train advanced XGBoost model with comprehensive evaluation"""

        logger.info(f"Training advanced XGBoost model on {len(df)} samples...")

        # Prepare features
        X, y = self.prepare_advanced_features(df, fit_encoders=True)

        logger.info(f"Created {X.shape[1]} features for training")

        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )

        # Create XGBoost datasets
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X.columns.tolist())
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=X.columns.tolist())

        # Training with early stopping
        evals = [(dtrain, "train"), (dval, "val")]

        self.model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=self.xgb_params["n_estimators"],
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50,
        )

        # Predictions for evaluation
        y_pred_proba = self.model.predict(dval)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="weighted"
        )

        # Feature importance
        self.feature_importance = self.model.get_score(importance_type="weight")

        # Calibration for better probability estimates
        self.calibrator = MulticlassPlatt()
        self.calibrator.fit(y_pred_proba, y_val)

        # Cross-validation for robustness
        cv_scores = self._cross_validate(X, y)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "feature_count": X.shape[1],
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
        }

        logger.info(
            f"Training complete - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})"
        )

        return metrics

    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5
    ) -> np.ndarray:
        """Perform cross-validation for robust evaluation"""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train fold model
            dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
            dval = xgb.DMatrix(X_fold_val, label=y_fold_val)

            fold_model = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=100,  # Reduced for CV
                evals=[(dval, "val")],
                early_stopping_rounds=20,
                verbose_eval=False,
            )

            # Evaluate
            y_pred = np.argmax(fold_model.predict(dval), axis=1)
            score = accuracy_score(y_fold_val, y_pred)
            scores.append(score)

        return np.array(scores)

    def predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict categories with confidence scores"""

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare features
        X, _ = self.prepare_advanced_features(df, fit_encoders=False)

        # Make predictions
        dtest = xgb.DMatrix(X, feature_names=X.columns.tolist())
        proba = self.model.predict(dtest)

        # Apply calibration if available
        if self.calibrator:
            proba = self.calibrator.predict_proba(proba)

        # Convert to results
        results = []
        for i, prob_dist in enumerate(proba):
            predicted_class = np.argmax(prob_dist)
            confidence = float(prob_dist[predicted_class])
            category = self.encoders.category_mapping[predicted_class]

            # Top 3 predictions
            top_indices = np.argsort(prob_dist)[::-1][:3]
            alternatives = [
                {
                    "category": self.encoders.category_mapping[idx],
                    "confidence": float(prob_dist[idx]),
                }
                for idx in top_indices[1:]
            ]

            results.append(
                {
                    "category": category,
                    "confidence": confidence,
                    "alternatives": alternatives,
                    "feature_count": X.shape[1],
                }
            )

        return results

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top feature importance scores"""
        if self.feature_importance is None:
            return {}

        # Sort by importance
        sorted_features = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        return dict(sorted_features[:top_n])

    def save_model(self, version: str = "2.0") -> str:
        """Save advanced model to MLflow"""

        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(self.xgb_params)
            mlflow.log_param(
                "feature_count", self.encoders.feature_spec.feature_columns.__len__()
            )
            mlflow.log_param("model_version", version)
            mlflow.log_param("model_type", "advanced_xgboost")

            # Log model
            mlflow.xgboost.log_model(self.model, "model")

            # Save encoders and feature spec
            artifacts_dir = Path("training/model_artifacts/advanced")
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Save encoders
            encoders_path = artifacts_dir / "encoders.pkl"
            with open(encoders_path, "wb") as f:
                pickle.dump(self.encoders, f)
            mlflow.log_artifact(str(encoders_path))

            # Save feature importance
            if self.feature_importance:
                importance_path = artifacts_dir / "feature_importance.json"
                with open(importance_path, "w") as f:
                    json.dump(self.feature_importance, f, indent=2)
                mlflow.log_artifact(str(importance_path))

            # Save model directly
            model_path = artifacts_dir / "xgb_advanced_model.json"
            self.model.save_model(str(model_path))
            mlflow.log_artifact(str(model_path))

            logger.info(f"Advanced model saved to MLflow run {run.info.run_id}")

        return run.info.run_id

    def load_model(self, model_path: str, encoders_path: str):
        """Load saved advanced model"""

        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        # Load encoders
        with open(encoders_path, "rb") as f:
            self.encoders = pickle.load(f)

        # Recreate feature engineer
        self.feature_engineer = AdvancedFeatureEngineer()

        logger.info(
            f"Advanced model loaded with {len(self.encoders.feature_spec.feature_columns)} features"
        )

    def explain_prediction(self, df: pd.DataFrame, index: int = 0) -> Dict[str, Any]:
        """Explain a single prediction using feature contributions"""

        if self.model is None:
            raise ValueError("Model not trained.")

        # Prepare features
        X, _ = self.prepare_advanced_features(df, fit_encoders=False)

        # Get prediction
        prediction = self.predict(df.iloc[[index]])[0]

        # Feature groups for better interpretation
        feature_groups = self.feature_engineer.get_feature_importance_groups()

        # Calculate feature contributions (simplified)
        feature_values = X.iloc[index].to_dict()
        top_features = self.get_feature_importance(top_n=10)

        explanation = {
            "prediction": prediction,
            "top_contributing_features": [
                {
                    "feature": feat,
                    "value": feature_values.get(feat, 0),
                    "importance": imp,
                }
                for feat, imp in top_features.items()
                if feat in feature_values
            ],
            "feature_groups": {
                group: [feat for feat in features if feat in feature_values]
                for group, features in feature_groups.items()
            },
        }

        return explanation
