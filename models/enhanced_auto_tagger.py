"""
Enhanced Auto-Tagging with Advanced Features Integration
Seamlessly upgrades existing XGBoost model to use 85+ advanced features
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
import json

# Add models to path
sys.path.append(str(Path(__file__).parent))

from .auto_tagging import XGBoostAutoTagger
from .advanced_features import AdvancedFeatureEngineer

logger = logging.getLogger(__name__)


class EnhancedAutoTagger:
    """
    Enhanced Auto-Tagger that seamlessly integrates 85+ advanced features
    with the existing XGBoost pipeline for improved accuracy
    """

    def __init__(self, use_advanced_features: bool = True):
        self.use_advanced_features = use_advanced_features

        if use_advanced_features:
            logger.info("Initializing Enhanced Auto-Tagger with 85+ advanced features")
            # Use enhanced XGBoost with advanced features
            self.tagger = XGBoostAutoTagger()
            self.feature_engineer = AdvancedFeatureEngineer()
        else:
            logger.info("Initializing standard Auto-Tagger")
            self.tagger = XGBoostAutoTagger()

        self.is_trained = False
        self.model_version = (
            "enhanced_v2.0" if use_advanced_features else "standard_v1.0"
        )

        # Load merchant intelligence from catalog
        self.merchant_catalog = self._load_merchant_catalog()

        # Try to load saved model if available
        self._load_saved_model()

    def _load_saved_model(self):
        """Load saved model artifacts if available"""
        try:
            from pathlib import Path

            model_dir = (
                Path(__file__).parent.parent
                / "training"
                / "model_artifacts"
                / "champion"
            )
            model_file = model_dir / "xgb_model.json"

            if model_file.exists():
                logger.info("Loading saved model from artifacts...")
                # Load model into the underlying tagger
                if hasattr(self.tagger, "load_model"):
                    success = self.tagger.load_model(
                        "Local"
                    )  # This loads from model_artifacts
                    if success:
                        self.is_trained = True
                        logger.info("✅ Successfully loaded pre-trained model")
                    else:
                        logger.warning("⚠️ Failed to load model, will use fallback")
                else:
                    logger.warning("⚠️ Model loading not supported by underlying tagger")
            else:
                logger.info("ℹ️ No saved model found, will train when needed")

        except Exception as e:
            logger.error(f"Failed to load saved model: {e}")
            self.is_trained = False

    def _load_merchant_catalog(self) -> Dict[str, str]:
        """Load merchant category catalog for intelligent lookups"""
        try:
            import pandas as pd
            from pathlib import Path

            catalog_path = (
                Path(__file__).parent.parent / "data" / "merchant_category.csv"
            )
            if not catalog_path.exists():
                logger.warning("Merchant catalog not found, using fallback rules only")
                return {}

            df = pd.read_csv(catalog_path)

            # Create mapping from normalized merchant names to categories
            catalog = {}
            category_map = {
                "Food & Dining": "Food & Dining",
                "Transport": "Transport",
                "Shopping": "Shopping",
                "Bills & Utilities": "Bills & Utilities",
                "Digital Subscriptions": "Entertainment",
                "Healthcare & Pharmacy": "Healthcare",
                "Education & Coaching": "Education",
                "Finance & Insurance": "Finance",
                "Groceries & Kirana": "Shopping",
                "Travel & Hospitality": "Transport",
                "Fashion & Lifestyle": "Shopping",
                "Government & NGOs": "Other",
                "Home & Services": "Other",
            }

            for _, row in df.iterrows():
                merchant_name = str(row["Merchant Name"]).lower().strip()
                category = category_map.get(row["Category"], "Other")
                catalog[merchant_name] = category

            logger.info(f"Loaded {len(catalog)} merchants from catalog")
            return catalog

        except Exception as e:
            logger.error(f"Failed to load merchant catalog: {e}")
            return {}

    def _enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced features to the dataframe"""
        if not self.use_advanced_features:
            return df

        try:
            # Create advanced features
            enhanced_df = self.feature_engineer.create_all_advanced_features(df)

            # Combine with original features from standard pipeline
            # The standard XGBoost will extract its own features, we just need data
            return enhanced_df

        except Exception as e:
            logger.warning(f"Failed to create advanced features: {e}")
            return df

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data"""

        required_columns = ["amount", "timestamp", "merchant_name"]

        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Clean data
        df = df.copy()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["merchant_name"] = df["merchant_name"].fillna("Unknown").astype(str)

        # Remove invalid rows
        initial_len = len(df)
        df = df.dropna(subset=["amount", "timestamp"])

        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} invalid rows")

        # Add user_id if missing (for feature engineering)
        if "user_id" not in df.columns:
            df["user_id"] = "default_user"

        return df

    def train(
        self, transactions: pd.DataFrame, target_accuracy: float = 0.80
    ) -> Dict[str, Any]:
        """Train the enhanced model with advanced features"""

        logger.info(
            f"Training {self.model_version} on {len(transactions)} transactions"
        )

        # Validate data
        transactions = self.validate_data(transactions)

        if "category" not in transactions.columns:
            raise ValueError("Training data must include 'category' column")

        # Enhance with advanced features if enabled
        if self.use_advanced_features:
            transactions = self._enhance_features(transactions)
            logger.info(f"Enhanced with {len(transactions.columns)} total features")

        # Train the underlying model
        metrics = self.tagger.train(transactions)
        self.is_trained = True

        # Check if target accuracy is met
        accuracy = metrics.get("accuracy", 0)
        if accuracy >= target_accuracy:
            logger.info(
                f"✅ Target accuracy achieved: {accuracy:.4f} >= {target_accuracy}"
            )
        else:
            logger.warning(
                f"⚠️ Target accuracy not met: {accuracy:.4f} < {target_accuracy}"
            )

        # Enhanced metrics
        enhanced_metrics = {
            **metrics,
            "model_version": self.model_version,
            "advanced_features": self.use_advanced_features,
            "target_accuracy_met": accuracy >= target_accuracy,
            "feature_engineering_level": (
                "advanced" if self.use_advanced_features else "standard"
            ),
        }

        return enhanced_metrics

    def predict_batch(
        self, transaction_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Predict categories for a batch of transactions (list of dicts)"""

        # Convert list of dicts to DataFrame
        df = pd.DataFrame(transaction_data)

        # Call the main predict method
        return self.predict(df)

    def predict(self, transactions: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict categories for transactions"""

        # Validate data
        transactions = self.validate_data(transactions)

        # Enhance with advanced features if enabled
        if self.use_advanced_features:
            transactions = self._enhance_features(transactions)

        # Make predictions using underlying tagger or fallback
        try:
            if (
                self.is_trained
                and hasattr(self.tagger, "predict")
                and callable(self.tagger.predict)
            ):
                predictions = self.tagger.predict(transactions)
            else:
                # Use fallback prediction logic when model not trained
                logger.info("Model not trained, using fallback prediction logic")
                predictions = self._fallback_predict(transactions)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            predictions = self._fallback_predict(transactions)

        # Enhance predictions with metadata and normalize format
        normalized_predictions = []
        for i, pred in enumerate(predictions):
            if isinstance(pred, dict):
                # Normalize response format
                normalized_pred = self._normalize_prediction_format(pred, i)
                normalized_predictions.append(normalized_pred)

        return normalized_predictions

    def _normalize_prediction_format(
        self, pred: Dict[str, Any], index: int
    ) -> Dict[str, Any]:
        """Normalize prediction format between ML model and fallback"""

        # Extract category - handle both ML model and fallback formats
        category = pred.get("predicted_category") or pred.get("category", "Other")
        confidence = pred.get("confidence", 0.5)

        # Map internal categories to display categories
        display_category = self._map_to_display_category(category)

        # Convert numpy types to Python native types to avoid serialization errors
        confidence = float(confidence)  # Ensure it's Python float, not numpy.float32

        # Process alternatives to ensure all values are JSON serializable
        alternatives = pred.get("top_predictions", pred.get("alternatives", []))
        normalized_alternatives = []
        for alt in alternatives:
            if isinstance(alt, dict):
                normalized_alt = {
                    "category": self._map_to_display_category(
                        alt.get("category", "Other")
                    ),
                    "confidence": float(
                        alt.get("confidence", 0.0)
                    ),  # Convert numpy types
                }
                normalized_alternatives.append(normalized_alt)
            else:
                normalized_alternatives.append(alt)

        # Standardized response format
        normalized = {
            "category": display_category,  # Always use 'category' as the main field
            "confidence": confidence,
            "alternatives": normalized_alternatives,
            "prediction_source": pred.get(
                "prediction_source", "ml_model" if self.is_trained else "fallback"
            ),
            "requires_review": bool(pred.get("requires_review", confidence < 0.7)),
            "model_version": self.model_version,
            "advanced_features": bool(self.use_advanced_features),
            "transaction_index": int(index),
        }

        return normalized

    def _map_to_display_category(self, internal_category: str) -> str:
        """Map internal ML categories to user-facing display categories"""
        category_map = {
            "food": "Food & Dining",
            "transport": "Transport",
            "shopping": "Shopping",
            "utilities": "Bills & Utilities",
            "entertainment": "Entertainment",
            "education": "Education",
            "investment": "Finance",
            "healthcare": "Healthcare",
            "other": "Other",
        }

        # Handle both internal and display categories
        return category_map.get(internal_category.lower(), internal_category)

    def _fallback_predict(self, transactions: pd.DataFrame) -> List[Dict[str, Any]]:
        """Intelligent fallback prediction using merchant catalog + rules"""
        logger.info("Using intelligent fallback prediction with merchant catalog")

        predictions = []
        for _, row in transactions.iterrows():
            merchant = str(row.get("merchant_name", "")).strip()
            merchant_lower = merchant.lower()
            amount = float(row.get("amount", 0))

            # Strategy 1: Exact merchant catalog lookup
            if merchant_lower in self.merchant_catalog:
                category = self.merchant_catalog[merchant_lower]
                confidence = 0.9  # High confidence for exact matches
                logger.info(f"Exact catalog match: {merchant} → {category}")

            # Strategy 2: Fuzzy merchant catalog lookup
            elif self.merchant_catalog:
                best_match = None
                best_score = 0

                for catalog_merchant, catalog_category in self.merchant_catalog.items():
                    # Simple substring matching
                    if (
                        catalog_merchant in merchant_lower
                        or merchant_lower in catalog_merchant
                    ):
                        score = min(len(catalog_merchant), len(merchant_lower)) / max(
                            len(catalog_merchant), len(merchant_lower)
                        )
                        if (
                            score > best_score and score > 0.6
                        ):  # 60% similarity threshold
                            best_match = catalog_category
                            best_score = score

                if best_match:
                    category = best_match
                    confidence = 0.7 + (best_score * 0.2)  # 0.7-0.9 based on similarity
                    logger.info(
                        f"Fuzzy catalog match: {merchant} → {category} (score: {best_score:.2f})"
                    )
                else:
                    # Strategy 3: Keyword-based fallback
                    category, confidence = self._keyword_based_prediction(
                        merchant_lower, amount
                    )
            else:
                # Strategy 3: Keyword-based fallback
                category, confidence = self._keyword_based_prediction(
                    merchant_lower, amount
                )

            predictions.append(
                {
                    "category": category,
                    "confidence": confidence,
                    "alternatives": [],
                    "prediction_source": (
                        "merchant_catalog"
                        if merchant_lower in self.merchant_catalog
                        else "fuzzy_match" if confidence > 0.7 else "keyword_rules"
                    ),
                }
            )

        return predictions

    def _keyword_based_prediction(self, merchant_lower: str, amount: float) -> tuple:
        """Keyword-based prediction as final fallback"""

        # Food & Dining keywords
        if any(
            keyword in merchant_lower
            for keyword in [
                "food",
                "restaurant",
                "cafe",
                "coffee",
                "tea",
                "chai",
                "pizza",
                "burger",
                "kitchen",
                "dining",
                "bakery",
                "sweets",
                "snacks",
                "meal",
                "lunch",
                "dinner",
            ]
        ):
            return "Food & Dining", 0.6

        # Transport keywords
        elif any(
            keyword in merchant_lower
            for keyword in [
                "uber",
                "ola",
                "taxi",
                "metro",
                "bus",
                "train",
                "petrol",
                "fuel",
                "parking",
            ]
        ):
            return "Transport", 0.6

        # Shopping keywords
        elif any(
            keyword in merchant_lower
            for keyword in [
                "amazon",
                "flipkart",
                "shopping",
                "store",
                "mart",
                "mall",
                "retail",
            ]
        ):
            return "Shopping", 0.6

        # Bills & Utilities keywords
        elif any(
            keyword in merchant_lower
            for keyword in [
                "electricity",
                "gas",
                "water",
                "mobile",
                "internet",
                "bill",
                "recharge",
            ]
        ):
            return "Bills & Utilities", 0.6

        # Entertainment keywords
        elif any(
            keyword in merchant_lower
            for keyword in [
                "movie",
                "cinema",
                "netflix",
                "entertainment",
                "game",
                "music",
            ]
        ):
            return "Entertainment", 0.6
        else:
            return "Other", 0.4

        return predictions

    def get_feature_insights(self) -> Dict[str, Any]:
        """Get insights about feature importance and model performance"""

        if not self.is_trained:
            return {"error": "Model not trained"}

        insights = {
            "model_version": self.model_version,
            "advanced_features": self.use_advanced_features,
        }

        if self.use_advanced_features:
            # Get feature groups info
            feature_groups = self.feature_engineer.get_feature_importance_groups()

            insights.update(
                {
                    "feature_groups": {
                        group: len(features)
                        for group, features in feature_groups.items()
                    },
                    "total_features": sum(
                        len(features) for features in feature_groups.values()
                    ),
                }
            )

        return insights

    def save_model(self, model_name: str = "enhanced_auto_tagger") -> str:
        """Save the trained model"""

        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save.")

        try:
            # Save using the underlying tagger's save method if available
            if hasattr(self.tagger, "save_model"):
                return self.tagger.save_model(version=self.model_version)
            else:
                logger.warning("No save_model method available")
                return "save_not_available"

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return "save_failed"


def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic test data for model validation"""

    np.random.seed(42)

    # Merchant categories and patterns
    merchants = {
        "Food & Dining": [
            "Zomato",
            "Swiggy",
            "McDonald's",
            "Subway",
            "Cafe Coffee Day",
        ],
        "Transport": ["Uber", "Ola", "Metro Card", "IRCTC", "Petrol Pump"],
        "Shopping": ["Amazon", "Flipkart", "Big Bazaar", "Reliance Store", "DMart"],
        "Bills & Utilities": [
            "Electricity Board",
            "Jio",
            "Airtel",
            "Gas Agency",
            "Water Board",
        ],
        "Entertainment": [
            "BookMyShow",
            "Netflix",
            "Spotify",
            "Gaming Store",
            "YouTube",
        ],
        "Other": [
            "ATM Withdrawal",
            "Bank Transfer",
            "Unknown Merchant",
            "Cash Deposit",
        ],
    }

    data = []
    user_ids = [f"user_{i}" for i in range(1, 51)]  # 50 users

    for i in range(n_samples):
        # Select category and merchant
        category = np.random.choice(list(merchants.keys()))
        merchant = np.random.choice(merchants[category])

        # Generate amount based on category
        if category == "Food & Dining":
            amount = np.random.exponential(300)
        elif category == "Transport":
            amount = np.random.exponential(150)
        elif category == "Shopping":
            amount = np.random.exponential(800)
        elif category == "Bills & Utilities":
            amount = np.random.exponential(1200)
        elif category == "Entertainment":
            amount = np.random.exponential(400)
        else:
            amount = np.random.exponential(500)

        amount = round(amount, 2)

        # Generate timestamp (last 90 days)
        days_ago = np.random.randint(0, 90)
        hours = np.random.randint(0, 24)
        timestamp = pd.Timestamp.now() - pd.Timedelta(days=days_ago, hours=hours)

        data.append(
            {
                "user_id": np.random.choice(user_ids),
                "merchant_name": merchant,
                "amount": amount,
                "timestamp": timestamp,
                "category": category,
            }
        )

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Demo: Enhanced Auto-Tagger with advanced features

    logging.basicConfig(level=logging.INFO)

    # Create test data
    logger.info("Creating test data...")
    test_data = create_test_data(n_samples=500)
    logger.info(f"Created {len(test_data)} test transactions")

    # Train enhanced model
    enhanced_tagger = EnhancedAutoTagger(use_advanced_features=True)
    metrics = enhanced_tagger.train(test_data, target_accuracy=0.75)

    print("\n🚀 Enhanced Auto-Tagger Training Results:")
    print(f"Model Version: {metrics['model_version']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Target Met: {metrics['target_accuracy_met']}")

    # Test predictions
    sample_data = test_data.head(5).drop(columns=["category"])
    predictions = enhanced_tagger.predict(sample_data)

    print("\n🎯 Sample Predictions:")
    for i, pred in enumerate(predictions):
        print(f"{i+1}. {pred['category']} (confidence: {pred['confidence']:.3f})")

    # Feature insights
    insights = enhanced_tagger.get_feature_insights()
    print(f"\n📊 Feature Insights:")
    print(f"Total Features: {insights.get('total_features', 'unknown')}")
    if "feature_groups" in insights:
        for group, count in insights["feature_groups"].items():
            print(f"  {group}: {count} features")

# Export instance for backward compatibility
enhanced_tagger = EnhancedAutoTagger(use_advanced_features=True)
