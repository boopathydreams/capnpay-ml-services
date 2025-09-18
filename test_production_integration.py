#!/usr/bin/env python3
"""Test complete production integration"""

import pandas as pd
from models.enhanced_auto_tagger import EnhancedAutoTagger
import logging

logging.basicConfig(level=logging.INFO)


def test_production_integration():
    print("🚀 Testing Complete Production Integration")
    print("=" * 50)

    # Test 1: Fresh instance (should load saved model)
    print("\n📊 Test 1: Fresh Enhanced Tagger (should auto-load model)")
    fresh_tagger = EnhancedAutoTagger(use_advanced_features=True)
    print(f"  Model trained status: {fresh_tagger.is_trained}")
    print(f"  Merchant catalog size: {len(fresh_tagger.merchant_catalog)}")

    # Test 2: Production-style prediction
    print("\n🎯 Test 2: Production Predictions")
    test_data = pd.DataFrame(
        [
            {
                "merchant_name": "Tea Point",
                "amount": 25.0,
                "timestamp": "2025-01-15 14:30:00",
                "user_id": "test_user",
            },
            {
                "merchant_name": "Unknown Chai Shop",  # Not in catalog
                "amount": 15.0,
                "timestamp": "2025-01-15 16:45:00",
                "user_id": "test_user",
            },
            {
                "merchant_name": "Starbucks",  # In catalog
                "amount": 120.0,
                "timestamp": "2025-01-15 19:00:00",
                "user_id": "test_user",
            },
            {
                "merchant_name": "Uber",  # Transport
                "amount": 85.0,
                "timestamp": "2025-01-15 20:00:00",
                "user_id": "test_user",
            },
            {
                "merchant_name": "Random New Service",  # Unknown
                "amount": 200.0,
                "timestamp": "2025-01-15 21:00:00",
                "user_id": "test_user",
            },
        ]
    )

    predictions = fresh_tagger.predict(test_data)

    print("\n📈 Prediction Results:")
    print(
        f"{'Merchant':<20} | {'Category':<15} | {'Conf':<5} | {'Source':<15} | {'Review':<6}"
    )
    print("-" * 70)

    for i, (_, row) in enumerate(test_data.iterrows()):
        pred = predictions[i]
        review = "✓" if not pred.get("requires_review", False) else "⚠️"
        print(
            f"{row['merchant_name']:<20} | {pred['category']:<15} | {pred['confidence']:<5.3f} | {pred['prediction_source']:<15} | {review:<6}"
        )

    # Test 3: API-style batch prediction
    print("\n🔄 Test 3: API-style Batch Prediction")
    api_data = [
        {
            "user_id": "user_001",
            "merchant_name": "Tea Garden",
            "amount": 45.0,
            "timestamp": "2025-01-15T14:30:00",
        },
        {
            "user_id": "user_002",
            "merchant_name": "Chai Wala",
            "amount": 20.0,
            "timestamp": "2025-01-15T16:45:00",
        },
    ]

    batch_predictions = fresh_tagger.predict_batch(api_data)

    print("\nBatch Prediction Results:")
    for i, pred in enumerate(batch_predictions):
        merchant = api_data[i]["merchant_name"]
        print(
            f"  {merchant} → {pred['category']} ({pred['confidence']:.3f} confidence)"
        )

    # Test 4: Response format validation
    print("\n✅ Test 4: Response Format Validation")
    sample_pred = predictions[0]
    required_fields = [
        "category",
        "confidence",
        "prediction_source",
        "requires_review",
        "model_version",
    ]

    print("Required fields check:")
    for field in required_fields:
        status = "✓" if field in sample_pred else "❌"
        value = sample_pred.get(field, "MISSING")
        print(f"  {field}: {status} ({value})")

    print(f"\n🎉 Production Integration Test Complete!")
    print(
        f"   - Model Status: {'✅ Trained' if fresh_tagger.is_trained else '❌ Not Trained'}"
    )
    print(f"   - Catalog Size: {len(fresh_tagger.merchant_catalog)} merchants")
    print(
        f"   - Tea Classification: {'✅ Fixed' if predictions[0]['category'] == 'Food & Dining' else '❌ Still Broken'}"
    )


if __name__ == "__main__":
    test_production_integration()
