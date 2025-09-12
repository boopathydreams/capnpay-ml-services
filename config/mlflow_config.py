# MLflow Configuration for Cap'n Pay AI/ML
# This file sets up the model registry and experiment tracking

import os
from pathlib import Path

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlflow-artifacts")
MLFLOW_EXPERIMENT_NAME = "capnpay-ai-models"

# Model Registry Settings
MODEL_REGISTRY_SETTINGS = {
    "auto_tagging": {
        "name": "payment-auto-tagging",
        "description": "XGBoost ensemble for payment categorization",
        "target_accuracy": 0.92,
        "target_latency_ms": 150,
    },
    "behavioral_nudges": {
        "name": "behavioral-nudge-engine",
        "description": "Psychology-based spending nudge generator",
        "target_engagement": 0.85,
        "target_latency_ms": 50,
    },
    "voice_intelligence": {
        "name": "voice-intelligence-engine",
        "description": "Multi-provider speech-to-text with NLP analysis",
        "target_accuracy": 0.90,
        "target_latency_ms": 5000,
    },
    "trust_scoring": {
        "name": "trust-score-engine",
        "description": "7-dimension trust analysis for contacts",
        "target_accuracy": 0.88,
        "target_latency_ms": 100,
    },
    "financial_advisor": {
        "name": "ai-financial-advisor",
        "description": "GPT-4 powered financial advisory system",
        "target_satisfaction": 4.5,
        "target_latency_ms": 3000,
    },
    "merchant_intelligence": {
        "name": "merchant-intelligence-system",
        "description": "Community-driven merchant categorization",
        "target_consensus": 0.90,
        "target_latency_ms": 200,
    },
}

# Experiment Configuration
EXPERIMENTS = {
    "auto_tagging": {
        "baseline": "rule_based_tagging",
        "champion": "xgboost_ensemble",
        "models": ["xgboost", "lightgbm", "neural_network", "collaborative_filter"],
    },
    "behavioral_nudges": {
        "baseline": "random_nudges",
        "champion": "psychology_based_nudges",
        "models": [
            "loss_aversion",
            "social_proof",
            "commitment_device",
            "mental_accounting",
        ],
    },
    "voice_intelligence": {
        "baseline": "basic_stt",
        "champion": "multi_provider_stt_nlp",
        "models": [
            "openai_whisper",
            "google_stt",
            "emotion_classifier",
            "financial_ner",
        ],
    },
    "trust_scoring": {
        "baseline": "simple_trust_score",
        "champion": "multi_dimensional_trust",
        "models": [
            "transaction_history",
            "network_analysis",
            "behavioral_patterns",
            "fraud_detection",
        ],
    },
    "financial_advisor": {
        "baseline": "template_advice",
        "champion": "gpt4_personalized_advisor",
        "models": [
            "intent_classifier",
            "context_builder",
            "knowledge_base",
            "advice_generator",
        ],
    },
    "merchant_intelligence": {
        "baseline": "manual_tagging",
        "champion": "community_consensus",
        "models": [
            "merchant_classifier",
            "voting_system",
            "fraud_detector",
            "consensus_engine",
        ],
    },
}

# Feature Store Configuration
import urllib.parse


def parse_redis_url(redis_url: str | None = None) -> dict:
    """Parse Redis URL into individual components"""
    if not redis_url:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/1")

    parsed = urllib.parse.urlparse(redis_url)

    return {
        "redis_host": parsed.hostname or "localhost",
        "redis_port": parsed.port or 6379,
        "redis_db": int(parsed.path.lstrip("/")) if parsed.path.lstrip("/") else 0,
        "redis_password": parsed.password,
        "redis_username": parsed.username,
    }


# Parse Redis configuration from URL
_redis_config = parse_redis_url()

FEATURE_STORE_CONFIG = {
    **_redis_config,
    "feature_ttl": 3600,  # 1 hour TTL for features
    "batch_size": 1000,
}

# Performance Monitoring
MONITORING_CONFIG = {
    "model_drift_threshold": 0.05,
    "accuracy_threshold": {
        "auto_tagging": 0.90,
        "behavioral_nudges": 0.80,
        "voice_intelligence": 0.85,
        "trust_scoring": 0.85,
        "financial_advisor": 4.0,
        "merchant_intelligence": 0.85,
    },
    "latency_threshold_ms": {
        "auto_tagging": 200,
        "behavioral_nudges": 100,
        "voice_intelligence": 7000,
        "trust_scoring": 150,
        "financial_advisor": 5000,
        "merchant_intelligence": 300,
    },
}
