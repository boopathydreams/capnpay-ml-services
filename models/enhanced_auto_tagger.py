"""
Enhanced Ensemble Auto-Tagger for Cap'n Pay
Handles feature mismatch by using multiple prediction strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from pathlib import Path
import json

from .auto_tagging import XGBoostAutoTagger
from .ensemble import ChampionDeltaEnsemble, EnsembleConfig

logger = logging.getLogger(__name__)


class EnhancedAutoTagger:
    """
    Enhanced auto-tagger that combines multiple approaches:
    1. Champion-Delta Ensemble (when features match)
    2. XGBoost with feature engineering (fallback)
    3. Rule-based predictions (final fallback)
    """

    def __init__(self):
        self.ensemble = None
        self.xgb_tagger = XGBoostAutoTagger()
        self.initialize_ensemble()

    def initialize_ensemble(self):
        """Initialize ensemble if artifacts exist"""
        try:
            champ_dir = Path("model_artifacts/champion")
            delta_dir = Path("model_artifacts/delta")

            if (champ_dir / "xgb_model.json").exists():
                config = EnsembleConfig(alpha=0.7, confidence_threshold=0.6)
                self.ensemble = ChampionDeltaEnsemble(
                    champion_dir=champ_dir,
                    delta_dir=(
                        delta_dir if (delta_dir / "xgb_model.json").exists() else None
                    ),
                    config=config,
                )
                logger.info("✅ Ensemble model initialized successfully")
            else:
                logger.warning("⚠️ Ensemble artifacts not found, using XGBoost fallback")

        except Exception as e:
            logger.error(f"❌ Failed to initialize ensemble: {e}")
            self.ensemble = None

    def predict_single(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict category for a single transaction using best available method
        """

        # Method 1: Try ensemble first
        if self.ensemble:
            try:
                df = pd.DataFrame([transaction_data])
                result = self.ensemble.predict(df)

                return {
                    "predicted_category": result["category"],
                    "confidence": result["confidence"],
                    "top_predictions": result["top_predictions"],
                    "method": "champion_delta_ensemble",
                    "model_source": result.get("source", "champion"),
                    "requires_review": result["confidence"] < 0.7,
                }

            except Exception as e:
                logger.warning(f"Ensemble prediction failed: {e}")

        # Method 2: Try XGBoost with feature engineering
        try:
            if self.xgb_tagger.model is not None:
                df = pd.DataFrame([transaction_data])
                predictions = self.xgb_tagger.predict(df)

                if predictions:
                    pred = predictions[0]
                    return {
                        "predicted_category": pred.get("predicted_category", "other"),
                        "confidence": pred.get("confidence", 0.0),
                        "top_predictions": pred.get("top_predictions", []),
                        "method": "xgboost_engineered",
                        "model_source": "xgboost",
                        "requires_review": pred.get("confidence", 0.0) < 0.7,
                    }
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")

        # Method 3: Rule-based fallback
        merchant_name = transaction_data.get("merchant_name", "").lower()
        amount = transaction_data.get("amount", 0)

        # Enhanced rule-based prediction
        predicted_category = self._enhanced_rule_based_prediction(merchant_name, amount)

        return {
            "predicted_category": predicted_category,
            "confidence": 0.65,  # Moderate confidence for enhanced rules
            "top_predictions": [
                {"category": predicted_category, "confidence": 0.65},
                {"category": "other", "confidence": 0.35},
            ],
            "method": "enhanced_rule_based",
            "model_source": "rules",
            "requires_review": True,
        }

    def _enhanced_rule_based_prediction(self, merchant_name: str, amount: float) -> str:
        """
        Enhanced rule-based prediction with amount-based heuristics
        """
        merchant_name = merchant_name.lower()

        # Food delivery and restaurants
        food_keywords = [
            "zomato",
            "swiggy",
            "dominos",
            "pizza",
            "burger",
            "restaurant",
            "cafe",
            "food",
            "dining",
            "meal",
            "kitchen",
            "biryani",
            "dosa",
            "mcdonald",
            "kfc",
            "subway",
            "starbucks",
            "cafe coffee day",
            "ccd",
            "haldiram",
            "barbeque",
            "taco bell",
        ]
        if any(keyword in merchant_name for keyword in food_keywords):
            return "food"

        # Transport and mobility
        transport_keywords = [
            "uber",
            "ola",
            "taxi",
            "cab",
            "metro",
            "bus",
            "train",
            "auto",
            "petrol",
            "diesel",
            "fuel",
            "parking",
            "toll",
            "transport",
            "rapido",
            "meru",
            "bmtc",
            "dmrc",
            "indian oil",
            "hp petrol",
            "bpcl",
            "shell",
        ]
        if any(keyword in merchant_name for keyword in transport_keywords):
            return "transport"

        # Shopping and retail
        shopping_keywords = [
            "amazon",
            "flipkart",
            "myntra",
            "ajio",
            "nykaa",
            "bigbasket",
            "grofers",
            "blinkit",
            "zepto",
            "instamart",
            "dunzo",
            "shop",
            "store",
            "mart",
            "mall",
            "retail",
            "grocery",
            "supermarket",
            "clothing",
            "fashion",
            "electronics",
            "basket",
        ]
        if any(keyword in merchant_name for keyword in shopping_keywords):
            return "shopping"

        # Utilities and bills
        utility_keywords = [
            "electricity",
            "water",
            "gas",
            "internet",
            "wifi",
            "mobile",
            "recharge",
            "bill",
            "utility",
            "broadband",
            "jio",
            "airtel",
            "bsnl",
        ]
        if any(keyword in merchant_name for keyword in utility_keywords):
            return "utilities"

        # Entertainment
        entertainment_keywords = [
            "netflix",
            "spotify",
            "youtube",
            "prime",
            "hotstar",
            "movie",
            "cinema",
            "bookmyshow",
            "game",
            "entertainment",
            "music",
        ]
        if any(keyword in merchant_name for keyword in entertainment_keywords):
            return "entertainment"

        # Healthcare
        healthcare_keywords = [
            "hospital",
            "clinic",
            "pharmacy",
            "medical",
            "doctor",
            "health",
            "medicine",
            "apollo",
            "fortis",
            "max",
            "manipal",
        ]
        if any(keyword in merchant_name for keyword in healthcare_keywords):
            return "healthcare"

        # Financial services
        if amount > 10000:  # Large amounts often investments or loan payments
            finance_keywords = [
                "bank",
                "mutual",
                "fund",
                "sip",
                "insurance",
                "loan",
                "emi",
                "investment",
                "trading",
                "zerodha",
                "groww",
                "paytm money",
            ]
            if any(keyword in merchant_name for keyword in finance_keywords):
                return "investment"

        # Education
        education_keywords = [
            "school",
            "college",
            "university",
            "course",
            "education",
            "fees",
            "tuition",
            "academy",
            "coaching",
            "byju",
            "unacademy",
        ]
        if any(keyword in merchant_name for keyword in education_keywords):
            return "education"

        # Default to other
        return "other"

    def predict_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict categories for multiple transactions
        """
        results = []
        for i, transaction in enumerate(transactions):
            try:
                prediction = self.predict_single(transaction)
                prediction["transaction_index"] = i
                results.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict transaction {i}: {e}")
                # Return fallback prediction
                results.append(
                    {
                        "transaction_index": i,
                        "predicted_category": "other",
                        "confidence": 0.3,
                        "top_predictions": [{"category": "other", "confidence": 0.3}],
                        "method": "error_fallback",
                        "model_source": "fallback",
                        "requires_review": True,
                        "error": str(e),
                    }
                )

        return results


# Global instance
enhanced_tagger = EnhancedAutoTagger()
