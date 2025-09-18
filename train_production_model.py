#!/usr/bin/env python3
"""
Train and Deploy Enhanced ML Model
Complete training pipeline with model persistence
"""

import pandas as pd
from models.enhanced_auto_tagger import EnhancedAutoTagger
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_deploy_model():
    """Train the enhanced model and save it for production use"""

    print("🚀 Training Enhanced ML Model for Production")
    print("=" * 50)

    # Load training data
    print("📊 Loading training data...")
    df = pd.read_csv("data/training_data.csv")
    print(f"✅ Loaded {len(df)} training samples")
    print(f"📋 Categories: {df['category'].value_counts().to_dict()}")

    # Initialize enhanced tagger
    print("\n🧠 Initializing Enhanced Auto-Tagger...")
    enhanced_tagger = EnhancedAutoTagger(use_advanced_features=True)
    print(f"📚 Merchant catalog: {len(enhanced_tagger.merchant_catalog)} merchants")

    # Train the model
    print("\n🔄 Training ML model...")
    metrics = enhanced_tagger.train(df, target_accuracy=0.75)

    print(f"\n📈 Training Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Test the trained model
    print("\n🧪 Testing trained model...")
    test_data = pd.DataFrame(
        [
            {
                "merchant_name": "Tea Point",
                "amount": 25.0,
                "timestamp": "2025-01-15 14:30:00",
                "user_id": "test_user",
            },
            {
                "merchant_name": "Starbucks",
                "amount": 120.0,
                "timestamp": "2025-01-15 19:00:00",
                "user_id": "test_user",
            },
            {
                "merchant_name": "Uber",
                "amount": 85.0,
                "timestamp": "2025-01-15 20:00:00",
                "user_id": "test_user",
            },
        ]
    )

    predictions = enhanced_tagger.predict(test_data)

    print("\n🎯 Test Predictions:")
    for i, (_, row) in enumerate(test_data.iterrows()):
        pred = predictions[i]
        source = pred.get("prediction_source", "ml_model")
        print(
            f"  {row['merchant_name']:15} → {pred['category']:15} (conf: {pred['confidence']:.3f}, source: {source})"
        )

    # Save the model
    print(f"\n💾 Model training status: {enhanced_tagger.is_trained}")
    if enhanced_tagger.is_trained:
        print("✅ Model successfully trained and ready for production!")

        # Note: Enhanced tagger uses the underlying XGBoost model
        # The model is automatically saved by the XGBoost tagger
        underlying_model_info = enhanced_tagger.tagger
        if hasattr(underlying_model_info, "model") and underlying_model_info.model:
            print("✅ Underlying XGBoost model is trained and ready")
        else:
            print("⚠️ Underlying XGBoost model may not be properly trained")
    else:
        print("❌ Model training failed!")

    return enhanced_tagger, metrics


if __name__ == "__main__":
    trained_model, results = train_and_deploy_model()
