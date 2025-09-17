"""
Enhanced Ensemble Auto-Tagger for Cap'n Pay
Handles feature mismatch by using multiple prediction strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
import re

from .auto_tagging import XGBoostAutoTagger
from .ensemble import ChampionDeltaEnsemble, EnsembleConfig
from .features import Encoders

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
        self._enc_cache: Optional[Encoders] = None
        self.initialize_ensemble()

    def initialize_ensemble(self):
        """Initialize ensemble if artifacts exist"""
        try:
            champ_dir = Path("training/model_artifacts/champion")
            delta_dir = Path("training/model_artifacts/delta")

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

    def _normalize(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return "unknown"
        s = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return re.sub(r"\s+", " ", s).strip()

    def _load_encoders(self) -> Optional[Encoders]:
        if self._enc_cache is not None:
            return self._enc_cache
        try:
            enc_path = Path("training/model_artifacts/champion/encoders.pkl")
            if enc_path.exists():
                self._enc_cache = Encoders.load(enc_path)
                return self._enc_cache
        except Exception as e:
            logger.warning(f"Failed to load encoders for novelty check: {e}")
        return None

    def _is_novel_merchant(self, merchant_name: Optional[str]) -> bool:
        norm = self._normalize(merchant_name or "")
        enc: Optional[Encoders] = None
        if self.ensemble and getattr(self.ensemble, "champion", None):
            enc = getattr(self.ensemble.champion, "encoders", None)
        if enc is None:
            enc = self._load_encoders()
        try:
            classes = (
                set(enc.merchant_name_encoder.classes_.tolist())
                if enc and hasattr(enc.merchant_name_encoder, "classes_")
                else set()
            )
            return norm not in classes
        except Exception:
            # If we cannot determine, treat as novel to be conservative
            return True

    def predict_single(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict category for a single transaction using best available method
        """

        # Method 1: Try ensemble first
        if self.ensemble:
            try:
                df = pd.DataFrame([transaction_data])
                result = self.ensemble.predict(df)
                top_preds = result.get("top_predictions") or result.get("topk", [])
                source = result.get("prediction_source", "champion")

                # Novelty dampening: if merchant unseen by encoders, cap confidence and flag review
                is_novel = self._is_novel_merchant(
                    transaction_data.get("merchant_name")
                )
                confidence = float(result["confidence"])
                if is_novel and confidence > 0.6:
                    # Scale down the full top_predictions proportionally for consistency
                    scale = 0.6 / confidence if confidence > 0 else 1.0
                    adjusted = []
                    for tp in top_preds:
                        try:
                            adjusted.append(
                                {
                                    "category": tp.get("category"),
                                    "confidence": float(tp.get("confidence", 0.0))
                                    * scale,
                                }
                            )
                        except Exception:
                            adjusted.append(tp)
                    top_preds = adjusted
                    confidence = 0.6
                return {
                    "predicted_category": result["category"],
                    "confidence": confidence,
                    "top_predictions": top_preds,
                    "method": "champion_delta_ensemble",
                    "model_source": source,
                    "requires_review": is_novel or (confidence < 0.7),
                    "novel_merchant": is_novel,
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
                    # Novelty dampening using same encoder cache
                    is_novel = self._is_novel_merchant(
                        transaction_data.get("merchant_name")
                    )
                    confidence = float(pred.get("confidence", 0.0))
                    top_preds = pred.get("top_predictions", [])
                    if is_novel and confidence > 0.6:
                        scale = 0.6 / confidence if confidence > 0 else 1.0
                        adjusted = []
                        for tp in top_preds:
                            try:
                                adjusted.append(
                                    {
                                        "category": tp.get("category"),
                                        "confidence": float(tp.get("confidence", 0.0))
                                        * scale,
                                    }
                                )
                            except Exception:
                                adjusted.append(tp)
                        top_preds = adjusted
                        confidence = 0.6
                    return {
                        "predicted_category": pred.get("predicted_category", "other"),
                        "confidence": confidence,
                        "top_predictions": top_preds,
                        "method": "xgboost_engineered",
                        "model_source": "xgboost",
                        "requires_review": is_novel or (confidence < 0.7),
                        "novel_merchant": is_novel,
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

    def _get_temporal_features(self, timestamp_str: str) -> Dict[str, float]:
        """
        Extract temporal features from transaction timestamp
        """
        try:
            from datetime import datetime

            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            hour = timestamp.hour
            day_of_week = timestamp.weekday()  # 0 = Monday
            is_weekend = day_of_week >= 5  # Saturday = 5, Sunday = 6

            features = {
                # Hour-based features
                "hour": hour,
                "is_morning": 1.0 if 6 <= hour <= 11 else 0.0,
                "is_lunch": 1.0 if 11 <= hour <= 14 else 0.0,
                "is_evening": 1.0 if 17 <= hour <= 21 else 0.0,
                "is_late_night": 1.0 if hour >= 22 or hour <= 5 else 0.0,
                # Day-based features
                "day_of_week": day_of_week,
                "is_weekend": 1.0 if is_weekend else 0.0,
                "is_weekday": 1.0 if not is_weekend else 0.0,
                "is_monday": 1.0 if day_of_week == 0 else 0.0,
                "is_friday": 1.0 if day_of_week == 4 else 0.0,
                # Combined patterns
                "weekend_morning": 1.0 if is_weekend and 6 <= hour <= 11 else 0.0,
                "weekday_commute": (
                    1.0
                    if not is_weekend and ((7 <= hour <= 10) or (17 <= hour <= 20))
                    else 0.0
                ),
                "weekend_evening": 1.0 if is_weekend and 17 <= hour <= 22 else 0.0,
            }

            return features

        except Exception as e:
            logger.warning(f"Failed to extract temporal features: {e}")
            return {
                "hour": 12,
                "is_morning": 0.0,
                "is_lunch": 0.0,
                "is_evening": 0.0,
                "is_late_night": 0.0,
                "day_of_week": 0,
                "is_weekend": 0.0,
                "is_weekday": 1.0,
                "is_monday": 0.0,
                "is_friday": 0.0,
                "weekend_morning": 0.0,
                "weekday_commute": 0.0,
                "weekend_evening": 0.0,
            }

    def _get_amount_pattern_features(self, amount: float) -> Dict[str, float]:
        """
        Extract amount-based pattern features
        """
        features = {
            "amount": amount,
            "amount_log": np.log1p(amount),  # log(1 + amount) to handle 0
            # Amount range indicators
            "is_micro": 1.0 if amount <= 50 else 0.0,
            "is_small": 1.0 if 50 < amount <= 500 else 0.0,
            "is_medium": 1.0 if 500 < amount <= 5000 else 0.0,
            "is_large": 1.0 if amount > 5000 else 0.0,
            # Common amount patterns
            "is_round_amount": 1.0 if amount % 100 == 0 else 0.0,
            "is_odd_amount": 1.0 if amount % 2 == 1 else 0.0,
            # Category-specific amount hints
            "likely_food_amount": 1.0 if 50 <= amount <= 1500 else 0.0,
            "likely_transport_amount": 1.0 if 10 <= amount <= 500 else 0.0,
            "likely_utility_amount": 1.0 if 100 <= amount <= 5000 else 0.0,
            "likely_shopping_amount": 1.0 if 200 <= amount <= 20000 else 0.0,
        }

        return features

    def _get_enhanced_merchant_features(self, merchant_name: str) -> Dict[str, float]:
        """
        Extract enhanced merchant-based features
        """
        if not merchant_name:
            merchant_name = ""

        merchant_lower = merchant_name.lower()
        features = {
            "merchant_length": len(merchant_name),
            "has_numbers": 1.0 if any(c.isdigit() for c in merchant_name) else 0.0,
            "has_special_chars": (
                1.0
                if any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?" for c in merchant_name)
                else 0.0
            ),
        }

        # Enhanced keyword matching with weights
        keyword_categories = {
            "food_strong": [
                "zomato",
                "swiggy",
                "dominos",
                "pizza",
                "restaurant",
                "cafe",
            ],
            "food_medium": ["food", "kitchen", "meal", "dining", "biryani", "dosa"],
            "transport_strong": ["uber", "ola", "rapido", "metro", "bmtc"],
            "transport_medium": ["taxi", "cab", "petrol", "diesel", "parking", "toll"],
            "shopping_strong": ["amazon", "flipkart", "myntra", "ajio"],
            "shopping_medium": ["shop", "store", "mall", "retail"],
            "entertainment_strong": ["netflix", "spotify", "amazon prime", "hotstar"],
            "entertainment_medium": ["movie", "cinema", "theater", "game"],
            "utility_strong": ["electricity", "water", "gas", "airtel", "jio"],
            "utility_medium": ["bill", "recharge", "utility", "internet"],
        }

        for category, keywords in keyword_categories.items():
            feature_name = f"keyword_{category}"
            features[feature_name] = (
                1.0 if any(kw in merchant_lower for kw in keywords) else 0.0
            )

        return features

    def predict_single_enhanced(
        self, transaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhanced prediction with temporal and pattern features
        """
        # Extract enhanced features
        merchant_name = transaction_data.get("merchant_name", "")
        amount = float(transaction_data.get("amount", 0))
        timestamp = transaction_data.get("timestamp", "")

        temporal_features = self._get_temporal_features(timestamp)
        amount_features = self._get_amount_pattern_features(amount)
        merchant_features = self._get_enhanced_merchant_features(merchant_name)

        # Get base prediction
        base_prediction = self.predict_single(transaction_data)

        # Enhance confidence based on temporal patterns
        enhanced_confidence = self._enhance_confidence_with_patterns(
            base_prediction, temporal_features, amount_features, merchant_features
        )

        # Update the prediction with enhanced confidence
        base_prediction["confidence"] = enhanced_confidence
        base_prediction["enhanced_features"] = {
            "temporal": temporal_features,
            "amount": amount_features,
            "merchant": merchant_features,
        }

        # Adjust requires_review based on enhanced confidence
        base_prediction["requires_review"] = enhanced_confidence < 0.7

        return base_prediction

    def _enhance_confidence_with_patterns(
        self,
        prediction: Dict[str, Any],
        temporal: Dict[str, float],
        amount: Dict[str, float],
        merchant: Dict[str, float],
    ) -> float:
        """
        Enhance prediction confidence using pattern analysis
        """
        base_confidence = float(prediction.get("confidence", 0.0))
        category = prediction.get("predicted_category", "")

        confidence_boost = 0.0

        # Food category enhancements
        if category == "food":
            # Strong food keywords boost
            if merchant.get("keyword_food_strong", 0) > 0:
                confidence_boost += 0.15
            elif merchant.get("keyword_food_medium", 0) > 0:
                confidence_boost += 0.08

            # Meal time patterns
            if (
                temporal.get("is_lunch", 0) > 0
                and amount.get("likely_food_amount", 0) > 0
            ):
                confidence_boost += 0.10
            elif (
                temporal.get("is_evening", 0) > 0
                and amount.get("likely_food_amount", 0) > 0
            ):
                confidence_boost += 0.08
            elif temporal.get("is_late_night", 0) > 0 and amount.get("is_small", 0) > 0:
                confidence_boost += 0.12  # Late night food delivery

        # Transport category enhancements
        elif category == "transport":
            if merchant.get("keyword_transport_strong", 0) > 0:
                confidence_boost += 0.15
            elif merchant.get("keyword_transport_medium", 0) > 0:
                confidence_boost += 0.08

            # Commute time patterns
            if (
                temporal.get("weekday_commute", 0) > 0
                and amount.get("likely_transport_amount", 0) > 0
            ):
                confidence_boost += 0.12

        # Shopping category enhancements
        elif category == "shopping":
            if merchant.get("keyword_shopping_strong", 0) > 0:
                confidence_boost += 0.15
            elif merchant.get("keyword_shopping_medium", 0) > 0:
                confidence_boost += 0.08

            # Weekend shopping patterns
            if (
                temporal.get("weekend_evening", 0) > 0
                and amount.get("likely_shopping_amount", 0) > 0
            ):
                confidence_boost += 0.10

        # Entertainment category enhancements
        elif category == "entertainment":
            if merchant.get("keyword_entertainment_strong", 0) > 0:
                confidence_boost += 0.15
            elif merchant.get("keyword_entertainment_medium", 0) > 0:
                confidence_boost += 0.08

            # Evening entertainment patterns
            if (
                temporal.get("is_evening", 0) > 0
                or temporal.get("weekend_evening", 0) > 0
            ):
                confidence_boost += 0.10

        # Bills/utility category enhancements
        elif category in ["bills", "utility"]:
            if merchant.get("keyword_utility_strong", 0) > 0:
                confidence_boost += 0.15
            elif merchant.get("keyword_utility_medium", 0) > 0:
                confidence_boost += 0.08

            # Utility amount patterns
            if amount.get("likely_utility_amount", 0) > 0:
                confidence_boost += 0.08

        # Apply boost and cap at reasonable levels
        enhanced_confidence = base_confidence + confidence_boost
        return min(
            enhanced_confidence, 0.96
        )  # Cap at 96% to leave room for uncertainty

    def predict_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict categories for multiple transactions using enhanced features
        """
        results = []
        for i, transaction in enumerate(transactions):
            try:
                # Use enhanced prediction with temporal and pattern features
                prediction = self.predict_single_enhanced(transaction)
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
