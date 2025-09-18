#!/usr/bin/env python3
"""
Enhanced XGBoost Model Training and Testing Script
Tests the advanced features integration and measures performance improvements
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import mlflow
import json
from datetime import datetime
from typing import Dict, Any

# Setup paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "models"))
sys.path.append(str(current_dir))

from models.enhanced_auto_tagger import EnhancedAutoTagger, create_test_data

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compare_model_performance(
    test_data: pd.DataFrame, test_size: int = 500
) -> Dict[str, Any]:
    """Compare standard vs enhanced model performance"""

    logger.info(f"Comparing model performance on {test_size} samples...")

    # Use subset for faster comparison
    comparison_data = test_data.head(test_size)

    results = {
        "test_samples": len(comparison_data),
        "timestamp": datetime.now().isoformat(),
    }

    # Test Standard Model
    logger.info("Training standard model...")
    try:
        standard_tagger = EnhancedAutoTagger(use_advanced_features=False)
        standard_metrics = standard_tagger.train(comparison_data, target_accuracy=0.70)

        # Test predictions
        sample_data = comparison_data.head(10).drop(columns=["category"])
        standard_predictions = standard_tagger.predict(sample_data)

        results["standard"] = {
            "training_metrics": standard_metrics,
            "prediction_count": len(standard_predictions),
            "avg_confidence": np.mean(
                [p.get("confidence", 0) for p in standard_predictions]
            ),
            "status": "success",
        }

        logger.info(
            f"Standard model - Accuracy: {standard_metrics.get('accuracy', 0):.4f}"
        )

    except Exception as e:
        logger.error(f"Standard model failed: {e}")
        results["standard"] = {"status": "failed", "error": str(e)}

    # Test Enhanced Model
    logger.info("Training enhanced model with advanced features...")
    try:
        enhanced_tagger = EnhancedAutoTagger(use_advanced_features=True)
        enhanced_metrics = enhanced_tagger.train(comparison_data, target_accuracy=0.75)

        # Test predictions
        sample_data = comparison_data.head(10).drop(columns=["category"])
        enhanced_predictions = enhanced_tagger.predict(sample_data)

        results["enhanced"] = {
            "training_metrics": enhanced_metrics,
            "prediction_count": len(enhanced_predictions),
            "avg_confidence": np.mean(
                [p.get("confidence", 0) for p in enhanced_predictions]
            ),
            "status": "success",
        }

        logger.info(
            f"Enhanced model - Accuracy: {enhanced_metrics.get('accuracy', 0):.4f}"
        )

        # Feature insights
        insights = enhanced_tagger.get_feature_insights()
        results["enhanced"]["feature_insights"] = insights

    except Exception as e:
        logger.error(f"Enhanced model failed: {e}")
        results["enhanced"] = {"status": "failed", "error": str(e)}

    # Calculate comparison metrics
    if (
        results["standard"]["status"] == "success"
        and results["enhanced"]["status"] == "success"
    ):
        # Safely get accuracy metrics
        standard_metrics = results["standard"]["training_metrics"]
        enhanced_metrics = results["enhanced"]["training_metrics"]

        # Handle case where metrics might be string or dict
        standard_acc = 0
        enhanced_acc = 0

        if isinstance(standard_metrics, dict):
            standard_acc = standard_metrics.get("accuracy", 0)
        if isinstance(enhanced_metrics, dict):
            enhanced_acc = enhanced_metrics.get("accuracy", 0)

        improvement = enhanced_acc - standard_acc
        relative_improvement = (
            (improvement / standard_acc * 100) if standard_acc > 0 else 0
        )

        results["comparison"] = {
            "accuracy_improvement": improvement,
            "relative_improvement_percent": relative_improvement,
            "recommended_model": (
                "enhanced" if enhanced_acc > standard_acc else "standard"
            ),
            "significant_improvement": improvement > 0.05,  # 5% threshold
        }

        logger.info(f"Performance Comparison:")
        logger.info(f"  Standard: {standard_acc:.4f}")
        logger.info(f"  Enhanced: {enhanced_acc:.4f}")
        logger.info(f"  Improvement: {improvement:.4f} ({relative_improvement:.2f}%)")

    return results


def test_prediction_quality(
    model: EnhancedAutoTagger, test_data: pd.DataFrame, n_samples: int = 20
) -> Dict[str, Any]:
    """Test prediction quality on sample data"""

    sample_data = test_data.head(n_samples)
    test_transactions = sample_data.drop(columns=["category"])
    true_categories = sample_data["category"].tolist()

    predictions = model.predict(test_transactions)

    # Calculate accuracy
    correct_predictions = 0
    prediction_details = []

    for i, (pred, true_cat) in enumerate(zip(predictions, true_categories)):
        predicted_cat = pred.get("category", "Unknown")
        confidence = pred.get("confidence", 0)
        is_correct = predicted_cat == true_cat

        if is_correct:
            correct_predictions += 1

        prediction_details.append(
            {
                "transaction_index": i,
                "merchant": sample_data.iloc[i]["merchant_name"],
                "amount": sample_data.iloc[i]["amount"],
                "true_category": true_cat,
                "predicted_category": predicted_cat,
                "confidence": confidence,
                "correct": is_correct,
            }
        )

    accuracy = correct_predictions / len(predictions)
    avg_confidence = np.mean([p.get("confidence", 0) for p in predictions])

    return {
        "sample_accuracy": accuracy,
        "average_confidence": avg_confidence,
        "correct_predictions": correct_predictions,
        "total_predictions": len(predictions),
        "prediction_details": prediction_details[:5],  # First 5 for review
    }


def main():
    """Main training and testing workflow"""

    logger.info("🚀 Starting Enhanced XGBoost Model Training and Testing")

    # Create test data
    logger.info("📊 Creating synthetic test data...")
    test_data = create_test_data(n_samples=1500)
    logger.info(f"Created {len(test_data)} test transactions")

    # Display data summary
    print("\n📈 Test Data Summary:")
    print(f"Total Transactions: {len(test_data)}")
    print(f"Categories: {test_data['category'].nunique()}")
    print(f"Unique Merchants: {test_data['merchant_name'].nunique()}")
    print(
        f"Amount Range: ₹{test_data['amount'].min():.2f} - ₹{test_data['amount'].max():.2f}"
    )
    print(
        f"Date Range: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}"
    )

    print("\nCategory Distribution:")
    category_counts = test_data["category"].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} ({count/len(test_data)*100:.1f}%)")

    # Compare model performance
    print("\n⚡ Comparing Model Performance...")
    comparison_results = compare_model_performance(test_data, test_size=800)

    # Display results
    print("\n📊 Performance Comparison Results:")

    if comparison_results["standard"]["status"] == "success":
        std_metrics = comparison_results["standard"]["training_metrics"]
        print(f"\n🔶 Standard Model:")
        print(f"  Accuracy: {std_metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {std_metrics.get('precision', 0):.4f}")
        print(f"  Recall: {std_metrics.get('recall', 0):.4f}")
        print(f"  F1-Score: {std_metrics.get('f1_score', 0):.4f}")
        print(f"  Training Samples: {std_metrics.get('training_samples', 'unknown')}")

    if comparison_results["enhanced"]["status"] == "success":
        enh_metrics = comparison_results["enhanced"]["training_metrics"]
        print(f"\n🔷 Enhanced Model (Advanced Features):")
        print(f"  Accuracy: {enh_metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {enh_metrics.get('precision', 0):.4f}")
        print(f"  Recall: {enh_metrics.get('recall', 0):.4f}")
        print(f"  F1-Score: {enh_metrics.get('f1_score', 0):.4f}")
        print(f"  Feature Count: {enh_metrics.get('feature_count', 'unknown')}")
        print(f"  Training Samples: {enh_metrics.get('training_samples', 'unknown')}")

        # Feature insights
        insights = comparison_results["enhanced"].get("feature_insights", {})
        if "feature_groups" in insights:
            print(f"\n🎯 Advanced Feature Groups:")
            for group, count in insights["feature_groups"].items():
                print(f"  {group}: {count} features")
            print(
                f"  Total Advanced Features: {insights.get('total_features', 'unknown')}"
            )

    # Performance comparison
    if "comparison" in comparison_results:
        comp = comparison_results["comparison"]
        print(f"\n📈 Performance Improvement:")
        print(f"  Accuracy Improvement: {comp['accuracy_improvement']:+.4f}")
        print(f"  Relative Improvement: {comp['relative_improvement_percent']:+.2f}%")
        print(f"  Recommended Model: {comp['recommended_model'].upper()}")
        print(
            f"  Significant Improvement: {'✅ YES' if comp['significant_improvement'] else '❌ NO'}"
        )

    # Test prediction quality on enhanced model if successful
    if comparison_results["enhanced"]["status"] == "success":
        print("\n🎯 Testing Prediction Quality...")
        enhanced_tagger = EnhancedAutoTagger(use_advanced_features=True)
        enhanced_tagger.train(test_data.head(800), target_accuracy=0.75)

        quality_results = test_prediction_quality(
            enhanced_tagger, test_data.tail(200), n_samples=10
        )

        print(f"\n🎪 Prediction Quality Results:")
        print(f"  Sample Accuracy: {quality_results['sample_accuracy']:.4f}")
        print(f"  Average Confidence: {quality_results['average_confidence']:.4f}")
        print(
            f"  Correct Predictions: {quality_results['correct_predictions']}/{quality_results['total_predictions']}"
        )

        print(f"\n🔍 Sample Predictions:")
        for detail in quality_results["prediction_details"]:
            status = "✅" if detail["correct"] else "❌"
            print(
                f"  {status} {detail['merchant'][:20]:<20} | ₹{detail['amount']:<8.2f} | "
                f"True: {detail['true_category']:<15} | Pred: {detail['predicted_category']:<15} | "
                f"Conf: {detail['confidence']:.3f}"
            )

    # Save results
    results_file = Path("training/advanced_model_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(comparison_results, f, indent=2, default=str)

    logger.info(f"✅ Results saved to {results_file}")

    # Final recommendation
    if (
        "comparison" in comparison_results
        and comparison_results["comparison"]["significant_improvement"]
    ):
        print(f"\n🎉 RECOMMENDATION: Deploy Enhanced Model with Advanced Features")
        print(
            f"   Achieved {comparison_results['comparison']['relative_improvement_percent']:.1f}% improvement over standard model"
        )
    else:
        print(f"\n⚠️ RECOMMENDATION: Continue with Standard Model")
        print(f"   Advanced features did not provide significant improvement")

    print(f"\n✨ Enhanced XGBoost Model Testing Complete!")


if __name__ == "__main__":
    main()
