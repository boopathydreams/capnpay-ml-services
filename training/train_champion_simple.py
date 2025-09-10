#!/usr/bin/env python3
"""
Simple Champion Model Training - Fixed Version
Train XGBoost model with real merchant data without complex features
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.xgboost


def main():
    parser = argparse.ArgumentParser(description="Train Champion XGBoost Model")
    parser.add_argument("--csv", required=True, help="Path to training CSV")
    parser.add_argument("--version", default="1.0.0", help="Model version")
    args = parser.parse_args()

    # Set MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("auto_tagging")

    print(f"📥 Loading data: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"🚀 Training Champion on {len(df)} rows...")

    # Simple feature engineering
    print("🔧 Engineering features...")

    # 1. Text features from merchant name
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    merchant_features = vectorizer.fit_transform(df["merchant_name"].fillna(""))

    # 2. Amount features
    df["amount_log"] = np.log1p(df["amount"])
    df["amount_scaled"] = (df["amount"] - df["amount"].mean()) / df["amount"].std()

    # 3. Temporal features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Combine features
    numerical_features = df[
        ["amount_log", "amount_scaled", "hour", "day_of_week", "month"]
    ].values
    X = np.hstack([merchant_features.toarray(), numerical_features])

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df["category"])

    print(f"✅ Feature matrix: {X.shape}, Target classes: {len(le.classes_)}")
    print(f"📊 Categories: {list(le.classes_)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train XGBoost
    with mlflow.start_run(run_name=f"Champion_v{args.version}"):
        print("🎯 Training XGBoost Champion...")

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"🎯 Champion Accuracy: {accuracy:.3f}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("n_features", X.shape[1])
        mlflow.log_metric("n_samples", len(df))

        # Log model
        mlflow.xgboost.log_model(model, "model")

        # Save artifacts
        os.makedirs("../artifacts", exist_ok=True)
        joblib.dump(model, f"../artifacts/champion_xgb_v{args.version}.joblib")
        joblib.dump(le, f"../artifacts/champion_label_encoder_v{args.version}.joblib")
        joblib.dump(
            vectorizer, f"../artifacts/champion_vectorizer_v{args.version}.joblib"
        )

        # Classification report
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        print("\n📈 Classification Report:")
        print(report)

        # Log report as artifact
        with open(f"../artifacts/champion_report_v{args.version}.txt", "w") as f:
            f.write(report)

        mlflow.log_artifact(f"../artifacts/champion_report_v{args.version}.txt")

        print(f"✅ Champion model trained! Accuracy: {accuracy:.1%}")

        # Promote to production if accuracy > 85%
        if accuracy > 0.85:
            print("🚀 Promoting Champion to PRODUCTION!")
            # Register model
            model_uri = mlflow.get_artifact_uri("model")
            mv = mlflow.register_model(model_uri, "auto_tagging_champion")

            # Transition to Production
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name="auto_tagging_champion", version=mv.version, stage="Production"
            )
        else:
            print(f"📝 Champion accuracy {accuracy:.1%} < 85%, keeping in Staging")

    return accuracy


if __name__ == "__main__":
    accuracy = main()
    print(f"\n🎉 Training complete! Final accuracy: {accuracy:.1%}")
