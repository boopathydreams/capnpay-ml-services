#!/usr/bin/env python3
"""Test the trained model predictions"""

import pandas as pd
from models.enhanced_auto_tagger import EnhancedAutoTagger
import logging

logging.basicConfig(level=logging.INFO)


def test_trained_model():
    print("🧪 Testing Trained Model")
    print("=" * 30)

    # Load training data and train model
    print("Training model...")
    df = pd.read_csv("data/training_data.csv")
    enhanced_tagger = EnhancedAutoTagger(use_advanced_features=True)
    enhanced_tagger.train(df, target_accuracy=0.75)

    # Test data
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

    print(f"\nModel trained: {enhanced_tagger.is_trained}")
    print("Making predictions...")

    predictions = enhanced_tagger.predict(test_data)

    print(f"\nPredictions (type: {type(predictions)}, length: {len(predictions)}):")
    for i, pred in enumerate(predictions):
        print(f"Prediction {i}: {pred}")
        print(f"  Type: {type(pred)}")
        print(f"  Keys: {pred.keys() if isinstance(pred, dict) else 'N/A'}")


if __name__ == "__main__":
    test_trained_model()
