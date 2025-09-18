#!/usr/bin/env python3
"""Test backend ML service integration"""

import requests
import json


def test_backend_integration():
    print("🔗 Testing Backend → ML Service Integration")
    print("=" * 50)

    # Test ML service endpoint
    ml_service_url = "http://localhost:8001/predict/auto-tag"

    test_request = {
        "transactions": [
            {
                "user_id": "test_user_001",
                "merchant_name": "Tea Point",
                "amount": 25.0,
                "timestamp": "2025-01-15T14:30:00",
            },
            {
                "user_id": "test_user_001",
                "merchant_name": "Chai Wala",
                "amount": 15.0,
                "timestamp": "2025-01-15T16:45:00",
            },
            {
                "user_id": "test_user_001",
                "merchant_name": "Starbucks",
                "amount": 120.0,
                "timestamp": "2025-01-15T19:00:00",
            },
        ]
    }

    try:
        print("📡 Sending request to ML service...")
        response = requests.post(ml_service_url, json=test_request, timeout=30)

        if response.status_code == 200:
            result = response.json()
            print("✅ ML Service Response:")
            print(f"   Processing time: {result.get('processing_time_ms', 0):.2f}ms")
            print(f"   Model version: {result.get('model_version', 'unknown')}")

            predictions = result.get("predictions", [])
            print(f"\n📊 Predictions ({len(predictions)} total):")

            for i, pred in enumerate(predictions):
                merchant = test_request["transactions"][i]["merchant_name"]
                category = pred.get("category", "Unknown")
                confidence = pred.get("confidence", 0)
                source = pred.get("prediction_source", "unknown")

                print(
                    f"   {merchant:<15} → {category:<15} ({confidence:.3f} via {source})"
                )

        else:
            print(f"❌ ML Service Error: {response.status_code}")
            print(f"   Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("⚠️ ML Service not running. Start it with:")
        print("   cd ml-services && source .venv/bin/activate && python main.py")

    except Exception as e:
        print(f"❌ Error testing ML service: {e}")


if __name__ == "__main__":
    test_backend_integration()
