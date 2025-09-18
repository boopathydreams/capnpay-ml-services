#!/usr/bin/env python3
"""Test tea merchant predictions with improved intelligence"""

from models.enhanced_auto_tagger import EnhancedAutoTagger
import pandas as pd


def test_tea_predictions():
    # Test data with various tea merchants
    test_data = pd.DataFrame(
        [
            {
                "merchant_name": "Tea Point",
                "amount": 25.0,
                "timestamp": "2025-01-15 14:30:00",
            },
            {
                "merchant_name": "Chai Wala",
                "amount": 15.0,
                "timestamp": "2025-01-15 16:45:00",
            },
            {
                "merchant_name": "Tea Garden Restaurant",
                "amount": 85.0,
                "timestamp": "2025-01-15 18:00:00",
            },
            {
                "merchant_name": "Starbucks",  # Should be in catalog
                "amount": 120.0,
                "timestamp": "2025-01-15 19:00:00",
            },
            {
                "merchant_name": "Café Coffee Day",  # Should be in catalog
                "amount": 80.0,
                "timestamp": "2025-01-15 20:00:00",
            },
        ]
    )

    print("🍵 Testing Intelligent Tea Merchant Predictions")
    print("=" * 50)

    enhanced_tagger = EnhancedAutoTagger(use_advanced_features=True)
    predictions = enhanced_tagger.predict(test_data)

    print(f"\nMerchant Catalog Size: {len(enhanced_tagger.merchant_catalog)} merchants")
    print("\nPredictions:")
    for i, (_, row) in enumerate(test_data.iterrows()):
        pred = predictions[i]
        source = pred.get("prediction_source", "unknown")
        print(
            f'{row["merchant_name"]:20} | ₹{row["amount"]:7.2f} | {pred["category"]:15} | Conf: {pred["confidence"]:.3f} | Source: {source}'
        )


if __name__ == "__main__":
    test_tea_predictions()
