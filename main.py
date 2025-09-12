"""
Cap'n Pay ML Services API
FastAPI server for AI/ML model serving and inference
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import pandas as pd
import asyncio

from models.auto_tagging import auto_tagger
from models.enhanced_auto_tagger import enhanced_tagger
from models.ensemble import ChampionDeltaEnsemble, EnsembleConfig
from pathlib import Path
from models.behavioral_nudges import (
    BehavioralNudgesEngine,
    UserPsychProfile,
    NudgeResult,
)
from models.financial_advisor import financial_advisor, FinancialAdviceRequest
from models.trust_scoring import trust_engine, TrustScore, TrustLevel, RiskFlag
from models.voice_intelligence import (
    voice_intelligence,
    VoiceAnalysisResult,
    VoiceProcessingMode,
)
from models.merchant_intelligence import MerchantIntelligenceEngine
from core.feature_store import feature_store
from core.model_registry import model_registry
from data.sample_generator import data_generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AI engines
behavioral_engine = BehavioralNudgesEngine()
merchant_intelligence = MerchantIntelligenceEngine()

# FastAPI app
app = FastAPI(
    title="Cap'n Pay ML Services",
    description="Production ML API for payment auto-tagging, behavioral nudges, and AI financial advisory",
    version="1.0.0",
    docs_url="/ml-docs",
    redoc_url="/ml-redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ensemble lazy loader (Champion + optional Delta)
_ensemble_instance = None


def get_ensemble() -> ChampionDeltaEnsemble:
    global _ensemble_instance
    if _ensemble_instance is None:
        champ_dir = Path("training/model_artifacts/champion")
        delta_dir = Path("training/model_artifacts/delta")
        cfg = EnsembleConfig(alpha=0.7, confidence_threshold=0.6)
        _ensemble_instance = ChampionDeltaEnsemble(
            champion_dir=champ_dir,
            delta_dir=(delta_dir if (delta_dir / "xgb_model.json").exists() else None),
            config=cfg,
        )
    return _ensemble_instance


# Request/Response Models
class Transaction(BaseModel):
    transaction_id: Optional[str] = None
    user_id: str = Field(..., description="Unique user identifier")
    merchant_name: str = Field(..., description="Merchant name")
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "merchant_name": "Zomato",
                "amount": 450.75,
                "timestamp": "2024-01-15T12:30:00",
            }
        }


class PredictionRequest(BaseModel):
    transactions: List[Transaction]


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    processing_time_ms: float
    model_version: str


class TrainingRequest(BaseModel):
    transactions: List[Dict[str, Any]]
    model_version: Optional[str] = "v1"


class TrainingSample(BaseModel):
    user_id: str
    merchant_name: str
    amount: float
    timestamp: str
    category: str  # canonical category name
    vpa: str
    source: str = "confirmed"  # AUTO or MANUAL


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Any]


class FinancialAdviceAPIRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    query: str = Field(..., description="Financial question or topic")
    context_type: str = Field(
        default="general",
        description="Type of advice: spending_review, goal_setting, investment_advice, debt_management",
    )
    transaction_data: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Recent transaction data for context"
    )
    time_horizon: Optional[str] = Field(
        default="medium", description="Time horizon: short, medium, long"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "query": "How can I save money on my monthly expenses?",
                "context_type": "spending_review",
                "time_horizon": "short",
            }
        }


class FinancialAdviceAPIResponse(BaseModel):
    advice: str
    confidence_score: float
    advice_category: str
    actionable_steps: List[str]
    relevant_insights: List[str]
    follow_up_questions: List[str]
    risk_warnings: List[str]
    supporting_data: Dict[str, Any]
    processing_time_ms: float


# Behavioral Nudges Models
class ProposedTransaction(BaseModel):
    """Transaction being considered by user"""

    amount: float = Field(..., gt=0, description="Transaction amount")
    category: str = Field(..., description="Transaction category")
    merchant: str = Field(..., description="Merchant name")
    timestamp: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 1200.0,
                "category": "food",
                "merchant": "Zomato",
                "timestamp": "2024-01-15T19:30:00",
            }
        }


class UserContext(BaseModel):
    """User's financial context for nudge generation"""

    user_id: str = Field(..., description="User identifier")
    budgets: Dict[str, float] = Field(
        default_factory=dict, description="Category budgets"
    )
    monthly_spending: Dict[str, float] = Field(
        default_factory=dict, description="Month-to-date spending by category"
    )
    financial_goals: List[Dict[str, Any]] = Field(
        default_factory=list, description="User's financial goals"
    )
    transaction_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Recent transaction history"
    )
    spending_commitments: List[Dict[str, Any]] = Field(
        default_factory=list, description="User commitments"
    )
    savings_goal: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "budgets": {"food": 5000, "transport": 3000, "shopping": 4000},
                "monthly_spending": {"food": 3200, "transport": 1800, "shopping": 2100},
                "financial_goals": [
                    {
                        "name": "Emergency Fund",
                        "target_amount": 50000,
                        "current_amount": 12000,
                        "target_date": "2024-12-31",
                    }
                ],
                "transaction_history": [],
                "spending_commitments": [],
                "savings_goal": {"name": "Emergency Fund", "target_amount": 50000},
            }
        }


class NudgeRequest(BaseModel):
    """Request for behavioral nudge generation"""

    proposed_transaction: ProposedTransaction
    user_context: UserContext
    generate_alternatives: bool = Field(
        default=True, description="Whether to generate alternative suggestions"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "proposed_transaction": {
                    "amount": 1200.0,
                    "category": "food",
                    "merchant": "Zomato",
                },
                "user_context": {
                    "user_id": "user_001",
                    "budgets": {"food": 5000},
                    "monthly_spending": {"food": 4200},
                },
                "generate_alternatives": True,
            }
        }


class NudgeResponse(BaseModel):
    """Response containing personalized nudges"""

    nudges: List[Dict[str, Any]]
    psychological_profile: Dict[str, float]
    recommendations: List[str]
    processing_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "nudges": [
                    {
                        "nudge_type": "loss_aversion",
                        "intensity": "moderate",
                        "message": "⚠️ This ₹1200 purchase will put you ₹400 over your food budget",
                        "confidence": 0.85,
                        "recommended_action": "Delay purchase for 24 hours",
                    }
                ],
                "psychological_profile": {
                    "loss_aversion_sensitivity": 0.7,
                    "social_influence_susceptibility": 0.5,
                    "commitment_tendency": 0.8,
                },
                "recommendations": [
                    "Consider cooking at home today",
                    "Wait until next month for food budget reset",
                ],
                "processing_time_ms": 45.2,
            }
        }


class PsychAnalysisRequest(BaseModel):
    """Request for psychological profile analysis"""

    user_id: str
    transaction_history: List[Dict[str, Any]]

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "transaction_history": [
                    {
                        "amount": 500,
                        "category": "food",
                        "timestamp": "2024-01-01T12:00:00",
                        "type": "purchase",
                    },
                    {
                        "amount": 300,
                        "category": "transport",
                        "timestamp": "2024-01-02T09:00:00",
                        "type": "purchase",
                    },
                ],
            }
        }


# Trust Scoring Models
class TrustScoreRequest(BaseModel):
    """Request for trust score calculation"""

    user_id: str = Field(..., description="User identifier")
    transactions: List[Transaction] = Field(..., description="User transaction history")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "transactions": [
                    {
                        "user_id": "user_001",
                        "merchant_name": "Zomato",
                        "amount": 450.75,
                        "timestamp": "2024-01-15T12:30:00",
                    },
                    {
                        "user_id": "user_001",
                        "merchant_name": "Uber",
                        "amount": 280.50,
                        "timestamp": "2024-01-15T18:45:00",
                    },
                ],
            }
        }


class TrustScoreResponse(BaseModel):
    """Response containing trust score and analysis"""

    user_id: str
    overall_score: float = Field(
        ..., ge=0, le=1, description="Overall trust score (0-1)"
    )
    trust_level: str = Field(
        ..., description="Trust level: very_high, high, medium, low, very_low"
    )
    risk_flags: List[str] = Field(..., description="Identified risk flags")
    component_scores: Dict[str, float] = Field(
        ..., description="Individual component scores"
    )
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the score")
    recommendations: List[str] = Field(..., description="Security recommendations")
    monitoring_frequency: str = Field(
        ..., description="Recommended monitoring frequency"
    )
    transaction_limits: Dict[str, float] = Field(
        ..., description="Recommended transaction limits"
    )
    risk_level: str = Field(..., description="Overall risk level")
    factors: Dict[str, Any] = Field(..., description="Factors affecting trust score")
    processing_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "overall_score": 0.85,
                "trust_level": "high",
                "risk_flags": ["no_risk"],
                "component_scores": {
                    "transaction_history": 0.88,
                    "network_analysis": 0.82,
                    "behavioral_patterns": 0.85,
                    "community_reputation": 0.85,
                },
                "confidence": 0.78,
                "recommendations": [
                    "Enable additional verification for high-value transactions"
                ],
                "monitoring_frequency": "daily",
                "transaction_limits": {
                    "daily_limit": 42500.0,
                    "per_transaction_limit": 21250.0,
                    "monthly_limit": 850000.0,
                },
                "risk_level": "low",
                "processing_time_ms": 45.2,
            }
        }


class VoiceAnalysisRequest(BaseModel):
    """Voice Analysis Request Model"""

    user_id: str = Field(..., description="User ID")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    audio_url: Optional[str] = Field(None, description="URL to audio file")
    processing_mode: Optional[str] = Field(
        "auto",
        description="Processing mode: auto, openai_api, local_whisper, google_stt",
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "audio_data": "UklGRjIAAABXQVZFZm10IBIAAAABAAEA...",
                "context": {"transaction_amount": 1500, "merchant_name": "Swiggy"},
            }
        }


class VoiceAnalysisResponse(BaseModel):
    """Voice Analysis Response Model"""

    user_id: str = Field(..., description="User ID")
    transcript: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Transcription confidence")
    processing_mode: str = Field(..., description="Processing mode used")
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    entities: Dict[str, List[str]] = Field(
        ..., description="Financial entities extracted"
    )
    emotions: Dict[str, float] = Field(..., description="Emotion analysis scores")
    sentiment_score: float = Field(..., description="Overall sentiment score (-1 to 1)")
    insights: List[Dict[str, Any]] = Field(..., description="Financial insights")
    spending_mentions: List[Dict[str, Any]] = Field(
        ..., description="Spending mentions found"
    )
    financial_keywords: List[str] = Field(
        ..., description="Financial keywords detected"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    language: str = Field(..., description="Detected language")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "transcript": "I paid fifteen hundred rupees to Swiggy for dinner delivery",
                "confidence": 0.95,
                "processing_mode": "local_whisper",
                "duration_seconds": 3.5,
                "entities": {
                    "amounts": ["1500"],
                    "merchants": ["Swiggy"],
                    "categories": ["food_delivery"],
                },
                "emotions": {"positive": 0.7, "neutral": 0.25, "negative": 0.05},
                "sentiment_score": 0.6,
                "insights": [
                    {
                        "type": "spending_pattern",
                        "content": "Regular food delivery spending detected",
                    },
                    {
                        "type": "emotional_state",
                        "content": "Positive experience with transaction",
                    },
                ],
                "spending_mentions": [
                    {"amount": 1500, "merchant": "Swiggy", "category": "food_delivery"}
                ],
                "financial_keywords": ["paid", "rupees", "delivery"],
                "processing_time_ms": 850.3,
                "language": "en",
            }
        }


class MerchantAnalysisRequest(BaseModel):
    """Merchant Analysis Request Model"""

    user_id: str = Field(..., description="User ID")
    merchant_vpa: str = Field(..., description="Merchant VPA address")
    transaction_history: List[Dict[str, Any]] = Field(
        ..., description="User's transaction history with merchant"
    )
    market_data: Optional[Dict[str, Any]] = Field(
        None, description="Market data for comparison"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "merchant_vpa": "swiggy@paytm",
                "transaction_history": [
                    {
                        "amount": 450.0,
                        "timestamp": "2023-12-01T19:30:00",
                        "status": "success",
                        "category": "food_delivery",
                    }
                ],
                "market_data": {
                    "average_transaction": 400.0,
                    "market_category": "food_delivery",
                },
            }
        }


class MerchantAnalysisResponse(BaseModel):
    """Merchant Analysis Response Model"""

    user_id: str = Field(..., description="User ID")
    merchant_vpa: str = Field(..., description="Merchant VPA")
    risk_level: str = Field(..., description="Risk level assessment")
    trust_score: float = Field(..., description="Trust score (0.0 to 1.0)")
    insights: List[Dict[str, Any]] = Field(..., description="Merchant insights")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    pricing_analysis: Dict[str, Any] = Field(..., description="Pricing analysis")
    behavioral_patterns: Dict[str, Any] = Field(..., description="Behavioral patterns")
    competitive_analysis: Dict[str, Any] = Field(
        ..., description="Competitive analysis"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "merchant_vpa": "swiggy@paytm",
                "risk_level": "low",
                "trust_score": 0.85,
                "insights": [
                    {
                        "type": "pricing_optimization",
                        "title": "Competitive Pricing",
                        "description": "This merchant offers competitive pricing",
                        "confidence": 0.9,
                    }
                ],
                "recommendations": ["Consider loyalty programs for frequent orders"],
                "pricing_analysis": {
                    "fairness_score": 0.8,
                    "market_position": "competitive",
                },
                "behavioral_patterns": {"frequency": "weekly", "timing": "evening"},
                "competitive_analysis": {"ranking": "top_25_percent"},
                "processing_time_ms": 125.5,
            }
        }


# Startup events
@app.on_event("startup")
async def startup_event():
    """Initialize ML services on startup"""
    logger.info("🚀 Starting Cap'n Pay ML Services...")

    # Health check all services
    health = await get_system_health()
    logger.info(f"🟢 System health: {health['status']}")

    # Load models if available
    try:
        # Prefer local artifacts; avoid noisy MLflow errors when registry is empty
        champ_path = Path("training/model_artifacts/champion/xgb_model.json")
        if champ_path.exists():
            model_loaded = auto_tagger.load_model("Local")
            if model_loaded:
                logger.info("✅ Auto-tagging model loaded from local artifacts")
            else:
                logger.info("⚠️ Local artifacts present but failed to load; using fallback")
        else:
            logger.info("ℹ️ No local model artifacts found; using ensemble/rule-based fallback")
    except Exception as e:
        logger.warning(f"Model load skipped due to error: {e}")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    health = await get_system_health()
    return health


async def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health"""
    try:
        # Check feature store
        feature_health = feature_store.health_check()

        # Check MLflow
        experiments = model_registry.list_experiments()
        mlflow_health = {
            "status": "healthy" if experiments else "degraded",
            "experiments_count": len(experiments),
        }

        # Overall status
        overall_status = "healthy"
        if (
            feature_health["status"] != "healthy"
            or mlflow_health["status"] != "healthy"
        ):
            overall_status = "degraded"

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "feature_store": feature_health,
                "mlflow": mlflow_health,
                "auto_tagger": {
                    "status": (
                        "ready" if auto_tagger.model is not None else "training_mode"
                    ),
                    "model_loaded": auto_tagger.model is not None,
                },
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "services": {},
        }


# Auto-tagging endpoints
@app.post("/predict/auto-tag", response_model=PredictionResponse)
async def predict_categories(request: PredictionRequest):
    """
    Predict payment categories using Champion-Delta Ensemble XGBoost model

    Uses advanced ensemble approach for 92%+ accuracy
    Returns top 3 predictions with calibrated confidence scores
    """
    start_time = datetime.now()

    try:
        # Convert request to transaction data format
        transaction_data = []
        for txn in request.transactions:
            transaction_data.append(
                {
                    "user_id": txn.user_id,
                    "merchant_name": txn.merchant_name,
                    "amount": txn.amount,
                    "timestamp": txn.timestamp or datetime.now().isoformat(),
                }
            )

        # Use enhanced tagger for robust predictions
        predictions = enhanced_tagger.predict_batch(transaction_data)

        # Store features in Redis for future use
        for i, txn in enumerate(request.transactions):
            feature_data = {
                "last_transaction_amount": txn.amount,
                "last_merchant": txn.merchant_name,
                "last_transaction_time": txn.timestamp or datetime.now().isoformat(),
            }

            # Store asynchronously
            asyncio.create_task(
                asyncio.to_thread(
                    feature_store.store_features,
                    txn.user_id,
                    "payment_features",
                    feature_data,
                )
            )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return PredictionResponse(
            predictions=predictions,
            processing_time_ms=processing_time,
            model_version=(
                "champion_xgboost_v1" if auto_tagger.model else "rule_based_fallback"
            ),
        )

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/auto-tag/ensemble", response_model=PredictionResponse)
async def predict_categories_ensemble(request: PredictionRequest):
    """Predict categories using Champion+Delta ensemble (calibrated)."""
    start = datetime.now()
    ens = get_ensemble()
    try:
        preds = []
        for i, txn in enumerate(request.transactions):
            row = pd.DataFrame(
                [
                    {
                        "user_id": txn.user_id,
                        "merchant_name": txn.merchant_name,
                        "amount": txn.amount,
                        "timestamp": txn.timestamp or datetime.now().isoformat(),
                    }
                ]
            )
            res = ens.predict(row)
            preds.append(
                {
                    "transaction_index": i,
                    "predicted_category": res["category"],
                    "confidence": res["confidence"],
                    "top_predictions": res["topk"],
                    "requires_review": not res["meets_threshold"],
                    "method": res["prediction_source"],
                }
            )
        return PredictionResponse(
            predictions=preds,
            processing_time_ms=(datetime.now() - start).total_seconds() * 1000,
            model_version="ensemble_v1",
        )
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/auto-tag")
async def train_auto_tagger(
    request: TrainingRequest, background_tasks: BackgroundTasks
):
    """
    Train auto-tagging model with new data
    Training happens in background to avoid blocking
    """
    try:
        # Validate data
        if len(request.transactions) < 100:
            raise HTTPException(
                status_code=400, detail="Minimum 100 transactions required for training"
            )

        # Start background training
        background_tasks.add_task(
            train_model_background, request.transactions, request.model_version
        )

        return {
            "status": "training_started",
            "message": f"Training initiated with {len(request.transactions)} transactions",
            "estimated_time_minutes": max(5, len(request.transactions) // 200),
            "model_version": request.model_version,
        }

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/events/training-sample")
async def stream_training_sample(sample: TrainingSample):
    """
    Stream individual confirmed label to ML for delta training
    Collects samples for batch training when threshold is reached
    """
    try:
        # Convert to transaction format
        transaction_data = {
            "user_id": sample.user_id,
            "merchant_name": sample.merchant_name,
            "amount": sample.amount,
            "timestamp": sample.timestamp,
            "category": sample.category,
            "vpa": sample.vpa,
            "source": sample.source,
        }

        # For now, just log the sample (in production, this would be stored for batch training)
        logger.info(
            f"🎯 Training sample received: {sample.merchant_name} → {sample.category} (confidence: {sample.source})"
        )

        # TODO: In production, accumulate samfples and trigger delta training when threshold reached
        # This could store in a queue/database and periodically retrain the model

        return {
            "status": "accepted",
            "message": "Training sample logged for future delta training",
            "sample_id": f"{sample.user_id}_{sample.timestamp}",
        }

    except Exception as e:
        logger.error(f"Error processing training sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def train_model_background(transactions_data: List[Dict[str, Any]], version: str):
    """Background task for model training"""
    try:
        logger.info(
            f"🎯 Starting background training with {len(transactions_data)} transactions"
        )

        # Convert to DataFrame
        df = pd.DataFrame(transactions_data)

        # Train model
        metrics = auto_tagger.train(df)

        # Save model to MLflow
        run_id = auto_tagger.save_model(version)

        # Store training metrics
        logger.info(
            f"✅ Training completed! Accuracy: {metrics.get('accuracy', 0):.4f}"
        )
        logger.info(f"📊 Model saved with run_id: {run_id}")

    except Exception as e:
        logger.error(f"Background training failed: {e}")


# Behavioral Nudges endpoints
@app.post("/predict/behavioral-nudges", response_model=NudgeResponse)
async def generate_behavioral_nudges(request: NudgeRequest):
    """
    Generate personalized behavioral nudges for a proposed transaction

    Uses advanced behavioral psychology principles:
    - Loss Aversion (Kahneman & Tversky)
    - Social Proof (Cialdini)
    - Mental Accounting (Thaler)
    - Commitment Devices (Ariely)
    - Temporal Discounting
    - Anchoring Bias
    """
    start_time = datetime.now()

    try:
        # Convert request to internal format
        proposed_transaction = {
            "amount": request.proposed_transaction.amount,
            "category": request.proposed_transaction.category,
            "merchant": request.proposed_transaction.merchant,
            "timestamp": request.proposed_transaction.timestamp
            or datetime.now().isoformat(),
        }

        user_context = {
            "budgets": request.user_context.budgets,
            "monthly_spending": request.user_context.monthly_spending,
            "financial_goals": request.user_context.financial_goals,
            "transaction_history": request.user_context.transaction_history,
            "spending_commitments": request.user_context.spending_commitments,
            "savings_goal": request.user_context.savings_goal,
        }

        # Generate psychological profile
        psych_profile = behavioral_engine.analyze_user_psychology(
            request.user_context.user_id, request.user_context.transaction_history
        )

        # Generate nudges
        nudges = behavioral_engine.generate_nudge(
            user_id=request.user_context.user_id,
            proposed_transaction=proposed_transaction,
            user_context=user_context,
            psych_profile=psych_profile,
        )

        # Convert nudges to response format
        nudge_data = []
        for nudge in nudges:
            nudge_data.append(
                {
                    "nudge_type": nudge.nudge_type.value,
                    "intensity": nudge.intensity.value,
                    "message": nudge.message,
                    "psychological_principle": nudge.psychological_principle,
                    "confidence": nudge.confidence,
                    "recommended_action": nudge.recommended_action,
                    "context_data": nudge.context_data,
                }
            )

        # Generate recommendations
        recommendations = []
        if nudges:
            top_nudge = nudges[0]
            if top_nudge.nudge_type.value == "loss_aversion":
                recommendations.append(
                    "Consider the long-term impact on your financial goals"
                )
                recommendations.append("Wait 24 hours before making this purchase")
            elif top_nudge.nudge_type.value == "social_proof":
                recommendations.append("Compare with similar users' spending patterns")
                recommendations.append(
                    "Consider what financially successful people would do"
                )
            elif top_nudge.nudge_type.value == "mental_accounting":
                recommendations.append(
                    "Review your budget allocation for this category"
                )
                recommendations.append("Consider reallocating from another category")

        # Store behavioral features for future use
        behavioral_features = {
            "last_nudge_generated": datetime.now().isoformat(),
            "loss_aversion_sensitivity": psych_profile.loss_aversion_sensitivity,
            "social_influence_susceptibility": psych_profile.social_influence_susceptibility,
            "commitment_tendency": psych_profile.commitment_tendency,
            "spending_impulsiveness": psych_profile.spending_impulsiveness,
        }

        # Store asynchronously
        asyncio.create_task(
            asyncio.to_thread(
                feature_store.store_features,
                request.user_context.user_id,
                "behavioral_features",
                behavioral_features,
            )
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return NudgeResponse(
            nudges=nudge_data,
            psychological_profile={
                "loss_aversion_sensitivity": psych_profile.loss_aversion_sensitivity,
                "social_influence_susceptibility": psych_profile.social_influence_susceptibility,
                "commitment_tendency": psych_profile.commitment_tendency,
                "temporal_preference": psych_profile.temporal_preference,
                "anchoring_susceptibility": psych_profile.anchoring_susceptibility,
                "risk_tolerance": psych_profile.risk_tolerance,
                "spending_impulsiveness": psych_profile.spending_impulsiveness,
            },
            recommendations=recommendations,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error generating behavioral nudges: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/psychology")
async def analyze_user_psychology(request: PsychAnalysisRequest):
    """
    Analyze user's psychological spending profile from transaction history

    Returns detailed psychological insights:
    - Loss aversion sensitivity
    - Social influence susceptibility
    - Commitment tendency
    - Temporal preference (present vs future focus)
    - Anchoring susceptibility
    - Risk tolerance
    - Spending impulsiveness
    """
    try:
        psych_profile = behavioral_engine.analyze_user_psychology(
            request.user_id, request.transaction_history
        )

        # Generate insights based on profile
        insights = []

        if psych_profile.loss_aversion_sensitivity > 0.7:
            insights.append(
                "High loss aversion - responds well to savings-focused messaging"
            )
        elif psych_profile.loss_aversion_sensitivity < 0.3:
            insights.append(
                "Low loss aversion - may need stronger financial guardrails"
            )

        if psych_profile.social_influence_susceptibility > 0.7:
            insights.append(
                "Highly influenced by social proof - peer comparisons are effective"
            )
        elif psych_profile.social_influence_susceptibility < 0.3:
            insights.append("Less influenced by others - focus on personal goals")

        if psych_profile.spending_impulsiveness > 0.7:
            insights.append(
                "High impulsiveness - benefits from delayed gratification techniques"
            )
        elif psych_profile.spending_impulsiveness < 0.3:
            insights.append("Low impulsiveness - already shows good self-control")

        if psych_profile.commitment_tendency > 0.7:
            insights.append(
                "High commitment tendency - commitment devices will be effective"
            )
        elif psych_profile.commitment_tendency < 0.3:
            insights.append(
                "Low commitment tendency - may need external accountability"
            )

        return {
            "user_id": request.user_id,
            "psychological_profile": {
                "loss_aversion_sensitivity": psych_profile.loss_aversion_sensitivity,
                "social_influence_susceptibility": psych_profile.social_influence_susceptibility,
                "commitment_tendency": psych_profile.commitment_tendency,
                "temporal_preference": psych_profile.temporal_preference,
                "anchoring_susceptibility": psych_profile.anchoring_susceptibility,
                "risk_tolerance": psych_profile.risk_tolerance,
                "spending_impulsiveness": psych_profile.spending_impulsiveness,
            },
            "insights": insights,
            "recommended_nudge_types": [
                (
                    "loss_aversion"
                    if psych_profile.loss_aversion_sensitivity > 0.5
                    else None
                ),
                (
                    "social_proof"
                    if psych_profile.social_influence_susceptibility > 0.5
                    else None
                ),
                (
                    "commitment_device"
                    if psych_profile.commitment_tendency > 0.5
                    else None
                ),
                (
                    "temporal_discounting"
                    if psych_profile.temporal_preference > 0.6
                    else None
                ),
            ],
            "analysis_quality": {
                "transaction_count": len(request.transaction_history),
                "confidence": min(
                    1.0, len(request.transaction_history) / 50
                ),  # Higher confidence with more data
                "recommendations": (
                    "Collect more transaction data for better analysis"
                    if len(request.transaction_history) < 20
                    else "Sufficient data for reliable analysis"
                ),
            },
        }

    except Exception as e:
        logger.error(f"Error analyzing user psychology: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nudges/track-effectiveness")
async def track_nudge_effectiveness(
    user_id: str, nudge_data: Dict[str, Any], user_action: str, outcome: Dict[str, Any]
):
    """
    Track nudge effectiveness for continuous learning

    Args:
        user_id: User identifier
        nudge_data: The nudge that was shown (from previous generate_nudges response)
        user_action: What user did (purchased, delayed, skipped, etc.)
        outcome: Financial outcome data
    """
    try:
        # This would be used to improve the nudge engine over time
        effectiveness_data = {
            "user_id": user_id,
            "nudge_type": nudge_data.get("nudge_type"),
            "intensity": nudge_data.get("intensity"),
            "predicted_confidence": nudge_data.get("confidence"),
            "user_action": user_action,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Nudge effectiveness tracked: {effectiveness_data}")

        # Store effectiveness data for ML model improvement
        asyncio.create_task(
            asyncio.to_thread(
                feature_store.store_features,
                user_id,
                "nudge_effectiveness",
                effectiveness_data,
            )
        )

        return {
            "status": "tracked",
            "message": "Nudge effectiveness data recorded for model improvement",
            "data": effectiveness_data,
        }

    except Exception as e:
        logger.error(f"Error tracking nudge effectiveness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nudges/analytics/{user_id}")
async def get_nudge_analytics(user_id: str, days: int = Query(30, ge=1, le=365)):
    """Get nudge effectiveness analytics for a user"""
    try:
        analytics = behavioral_engine.get_nudge_analytics(user_id, days)

        return {
            "user_id": user_id,
            "period_days": days,
            "analytics": analytics,
            "retrieved_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error retrieving nudge analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Data generation endpoints (for testing)
@app.post("/data/generate-sample")
async def generate_sample_data(
    n_users: int = Query(50, ge=10, le=1000),
    n_transactions: int = Query(1000, ge=100, le=10000),
):
    """Generate sample transaction data for testing"""
    try:
        df = data_generator.generate_transactions(n_users, n_transactions)

        # Convert to JSON serializable format
        transactions = df.to_dict("records")

        return {
            "status": "success",
            "data": transactions,
            "summary": {
                "total_transactions": len(transactions),
                "users": n_users,
                "categories": df["category"].value_counts().to_dict(),
                "amount_stats": {
                    "mean": float(df["amount"].mean()),
                    "median": float(df["amount"].median()),
                    "min": float(df["amount"].min()),
                    "max": float(df["amount"].max()),
                },
            },
        }

    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Feature store endpoints
@app.get("/features/{user_id}")
async def get_user_features(user_id: str, feature_set: str = "payment_features"):
    """Get cached features for a user"""
    try:
        features = await asyncio.to_thread(
            feature_store.get_features, user_id, feature_set
        )

        if features is None:
            raise HTTPException(status_code=404, detail="Features not found")

        return {
            "user_id": user_id,
            "feature_set": feature_set,
            "features": features,
            "retrieved_at": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/stats/{feature_set}")
async def get_feature_stats(feature_set: str):
    """Get statistics for a feature set"""
    try:
        stats = await asyncio.to_thread(feature_store.get_feature_stats, feature_set)

        return {
            "feature_set": feature_set,
            "stats": stats,
            "retrieved_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error retrieving feature stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model registry endpoints
@app.get("/models/experiments")
async def list_experiments():
    """List all ML experiments"""
    try:
        experiments = model_registry.list_experiments()
        return {"experiments": experiments, "total_count": len(experiments)}

    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{component}/metrics")
async def get_model_metrics(component: str, stage: str = "Production"):
    """Get metrics for a specific model"""
    try:
        metrics = model_registry.get_model_metrics(component, stage)
        return {
            "component": component,
            "stage": stage,
            "metrics": metrics,
            "retrieved_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error retrieving model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Financial Advisor endpoints
@app.post("/advice/financial", response_model=FinancialAdviceAPIResponse)
async def get_financial_advice(request: FinancialAdviceAPIRequest):
    """
    Get personalized financial advice using GPT-4

    Provides intelligent financial guidance based on user context and spending patterns
    """
    start_time = datetime.now()

    try:
        # Convert API request to internal format
        advice_request = FinancialAdviceRequest(
            user_id=request.user_id,
            query=request.query,
            context_type=request.context_type,
            transaction_data=request.transaction_data,
            time_horizon=request.time_horizon,
        )

        # Get advice from GPT-4
        advice_response = await financial_advisor.get_financial_advice(advice_request)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return FinancialAdviceAPIResponse(
            advice=advice_response.advice,
            confidence_score=advice_response.confidence_score,
            advice_category=advice_response.advice_category,
            actionable_steps=advice_response.actionable_steps,
            relevant_insights=advice_response.relevant_insights,
            follow_up_questions=advice_response.follow_up_questions,
            risk_warnings=advice_response.risk_warnings,
            supporting_data=advice_response.supporting_data,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error getting financial advice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/advice/spending-analysis/{user_id}")
async def analyze_user_spending(
    user_id: str, days: int = Query(30, description="Number of days to analyze")
):
    """Analyze user spending patterns and provide insights"""
    try:
        # Get recent transactions (mock data for demo)
        sample_transactions = data_generator.generate_transactions(
            n_users=1, n_transactions=20
        )
        sample_transactions["user_id"] = user_id
        transactions = sample_transactions.to_dict("records")

        # Analyze patterns
        analysis = await financial_advisor.analyze_spending_patterns(
            user_id, transactions
        )

        return {
            "user_id": user_id,
            "analysis_period_days": days,
            "spending_insights": analysis,
            "analyzed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error analyzing spending: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/advice/goal-suggestions/{user_id}")
async def suggest_financial_goals(user_id: str):
    """Suggest appropriate financial goals based on user profile"""
    try:
        # Mock user profile (in production, would come from user service)
        user_profile = {
            "monthly_income": 75000,
            "age": 28,
            "employment_status": "employed",
            "dependents": 1,
            "risk_tolerance": "moderate",
        }

        # Get goal suggestions
        goals = await financial_advisor.suggest_financial_goals(user_profile)

        return {
            "user_id": user_id,
            "user_profile": user_profile,
            "suggested_goals": goals,
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error suggesting goals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Demo endpoint
@app.get("/demo/auto-tag")
async def demo_auto_tagging():
    """Demo endpoint showing auto-tagging capabilities"""
    try:
        # Generate sample transactions
        sample_data = data_generator.generate_transactions(n_users=5, n_transactions=20)

        # Convert to request format
        transactions = []
        for _, row in sample_data.iterrows():
            transactions.append(
                Transaction(
                    user_id=row["user_id"],
                    merchant_name=row["merchant_name"],
                    amount=row["amount"],
                    timestamp=row["timestamp"].isoformat(),
                )
            )

        # Get predictions
        request = PredictionRequest(transactions=transactions)
        predictions = await predict_categories(request)

        # Add actual categories for comparison
        demo_results = []
        for i, (_, row) in enumerate(sample_data.iterrows()):
            pred = (
                predictions.predictions[i] if i < len(predictions.predictions) else {}
            )
            demo_results.append(
                {
                    "transaction": {
                        "merchant": row["merchant_name"],
                        "amount": row["amount"],
                        "actual_category": row["category"],
                    },
                    "prediction": pred,
                }
            )

        return {
            "demo_results": demo_results,
            "model_info": {
                "version": predictions.model_version,
                "processing_time_ms": predictions.processing_time_ms,
            },
            "summary": {
                "total_predictions": len(demo_results),
                "high_confidence": sum(
                    1
                    for r in demo_results
                    if r["prediction"].get("confidence", 0) > 0.8
                ),
                "requires_review": sum(
                    1
                    for r in demo_results
                    if r["prediction"].get("requires_review", False)
                ),
            },
        }

    except Exception as e:
        logger.error(f"Error in demo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/demo/behavioral-nudges")
async def demo_behavioral_nudges():
    """Demo endpoint showing behavioral nudges capabilities"""
    try:
        # Create demo scenarios with different psychological profiles
        demo_scenarios = [
            {
                "name": "High Spender - Food Budget Exceeded",
                "proposed_transaction": {
                    "amount": 1500.0,
                    "category": "food",
                    "merchant": "Expensive Restaurant",
                },
                "user_context": {
                    "user_id": "demo_user_1",
                    "budgets": {"food": 4000, "entertainment": 2000},
                    "monthly_spending": {"food": 3800, "entertainment": 500},
                    "financial_goals": [
                        {
                            "name": "Emergency Fund",
                            "target_amount": 50000,
                            "current_amount": 15000,
                            "target_date": "2024-12-31",
                        }
                    ],
                    "transaction_history": [
                        {
                            "amount": 800,
                            "category": "food",
                            "timestamp": "2024-01-10T12:00:00",
                            "type": "purchase",
                        },
                        {
                            "amount": 1200,
                            "category": "food",
                            "timestamp": "2024-01-12T19:00:00",
                            "type": "purchase",
                        },
                        {
                            "amount": 600,
                            "category": "food",
                            "timestamp": "2024-01-14T13:00:00",
                            "type": "purchase",
                        },
                    ],
                },
            },
            {
                "name": "Impulsive Shopper - Social Proof Scenario",
                "proposed_transaction": {
                    "amount": 2500.0,
                    "category": "shopping",
                    "merchant": "Luxury Brand Store",
                },
                "user_context": {
                    "user_id": "demo_user_2",
                    "budgets": {"shopping": 3000, "food": 4000},
                    "monthly_spending": {"shopping": 2800, "food": 2100},
                    "transaction_history": [
                        {
                            "amount": 500,
                            "category": "shopping",
                            "timestamp": "2024-01-08T15:00:00",
                            "type": "purchase",
                        },
                        {
                            "amount": 800,
                            "category": "shopping",
                            "timestamp": "2024-01-08T15:30:00",
                            "type": "purchase",
                        },
                        {
                            "amount": 1200,
                            "category": "shopping",
                            "timestamp": "2024-01-08T16:00:00",
                            "type": "purchase",
                        },
                        {
                            "amount": 300,
                            "category": "shopping",
                            "timestamp": "2024-01-08T16:15:00",
                            "type": "purchase",
                        },
                    ],
                    "savings_goal": {"name": "Vacation Fund", "target_amount": 100000},
                },
            },
            {
                "name": "Goal-Oriented Saver - Commitment Device",
                "proposed_transaction": {
                    "amount": 3000.0,
                    "category": "entertainment",
                    "merchant": "Concert Tickets",
                },
                "user_context": {
                    "user_id": "demo_user_3",
                    "budgets": {"entertainment": 2000, "food": 3000},
                    "monthly_spending": {"entertainment": 1200, "food": 2400},
                    "financial_goals": [
                        {
                            "name": "House Down Payment",
                            "target_amount": 500000,
                            "current_amount": 180000,
                            "target_date": "2025-06-01",
                        }
                    ],
                    "spending_commitments": [
                        {
                            "category": "entertainment",
                            "limit": 2000,
                            "commitment_date": "2024-01-01",
                        }
                    ],
                    "transaction_history": [
                        {
                            "amount": 400,
                            "category": "entertainment",
                            "timestamp": "2024-01-05T20:00:00",
                            "type": "purchase",
                        },
                        {
                            "amount": 800,
                            "category": "entertainment",
                            "timestamp": "2024-01-12T19:00:00",
                            "type": "purchase",
                        },
                    ],
                },
            },
        ]

        demo_results = []

        for scenario in demo_scenarios:
            # Create request
            request = NudgeRequest(
                proposed_transaction=ProposedTransaction(
                    **scenario["proposed_transaction"]
                ),
                user_context=UserContext(**scenario["user_context"]),
                generate_alternatives=True,
            )

            # Generate nudges
            response = await generate_behavioral_nudges(request)

            demo_results.append(
                {
                    "scenario_name": scenario["name"],
                    "transaction": scenario["proposed_transaction"],
                    "context_summary": {
                        "budget_status": scenario["user_context"]["budgets"],
                        "current_spending": scenario["user_context"][
                            "monthly_spending"
                        ],
                        "has_goals": len(
                            scenario["user_context"].get("financial_goals", [])
                        )
                        > 0,
                    },
                    "psychological_profile": response.psychological_profile,
                    "nudges": response.nudges,
                    "top_recommendation": (
                        response.recommendations[0]
                        if response.recommendations
                        else "No specific recommendation"
                    ),
                    "processing_time_ms": response.processing_time_ms,
                }
            )

        return {
            "demo_title": "Behavioral Nudges Engine - Psychology-Based Financial Guidance",
            "demo_description": "Advanced behavioral psychology engine using research from Kahneman, Thaler, Cialdini, and Ariely",
            "scenarios": demo_results,
            "summary": {
                "total_scenarios": len(demo_results),
                "average_processing_time": sum(
                    r["processing_time_ms"] for r in demo_results
                )
                / len(demo_results),
                "nudge_types_demonstrated": list(
                    set(
                        nudge["nudge_type"]
                        for result in demo_results
                        for nudge in result["nudges"]
                    )
                ),
                "psychological_principles": [
                    "Loss Aversion (Kahneman & Tversky)",
                    "Social Proof (Cialdini)",
                    "Mental Accounting (Thaler)",
                    "Commitment Devices (Ariely)",
                    "Temporal Discounting",
                    "Anchoring Bias",
                ],
            },
            "model_info": {
                "engine": "BehavioralNudgesEngine",
                "version": "1.0.0",
                "target_effectiveness": "85%+ user engagement",
                "personalization": "Individual psychological profiling",
            },
        }

    except Exception as e:
        logger.error(f"Error in behavioral nudges demo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/demo/financial-advisor")
async def demo_financial_advisor():
    """Demo endpoint showing GPT-4 financial advisor capabilities"""
    try:
        # Demo scenarios
        demo_scenarios = [
            {
                "scenario": "Budget Review",
                "user_profile": "Tech Professional, ₹80K/month, Age 27",
                "query": "I'm spending too much on food delivery. How can I reduce this expense?",
                "context_type": "spending_review",
            },
            {
                "scenario": "Investment Planning",
                "user_profile": "Marketing Manager, ₹65K/month, Age 32",
                "query": "I want to start investing ₹10,000 per month. What should I invest in?",
                "context_type": "investment_advice",
            },
            {
                "scenario": "Goal Setting",
                "user_profile": "Software Engineer, ₹1.2L/month, Age 29",
                "query": "I want to buy a house in 3 years. How much should I save monthly?",
                "context_type": "goal_setting",
            },
        ]

        demo_results = []

        for scenario in demo_scenarios:
            # Create request
            advice_request = FinancialAdviceRequest(
                user_id=f"demo_user_{len(demo_results)}",
                query=scenario["query"],
                context_type=scenario["context_type"],
            )

            # Get advice (will use fallback since no OpenAI key)
            start_time = datetime.now()
            advice_response = await financial_advisor.get_financial_advice(
                advice_request
            )
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            demo_results.append(
                {
                    "scenario_info": scenario,
                    "advice_response": {
                        "advice": advice_response.advice,
                        "confidence_score": advice_response.confidence_score,
                        "actionable_steps": advice_response.actionable_steps,
                        "insights": advice_response.relevant_insights,
                        "warnings": advice_response.risk_warnings,
                        "follow_up": advice_response.follow_up_questions,
                    },
                    "processing_time_ms": processing_time,
                }
            )

        return {
            "demo_title": "GPT-4 Financial Advisor - Personalized Financial Guidance",
            "demo_description": "AI-powered financial advisory system providing personalized advice for Indian financial context",
            "demo_results": demo_results,
            "model_info": {
                "engine": "GPT-4 Financial Advisor",
                "version": "1.0.0",
                "target_satisfaction": "4.5/5 user satisfaction",
                "response_time": "<3000ms",
            },
            "summary": {
                "total_scenarios": len(demo_results),
                "average_processing_time": sum(
                    r["processing_time_ms"] for r in demo_results
                )
                / len(demo_results),
                "advice_categories": list(
                    set(s["context_type"] for s in demo_scenarios)
                ),
                "features_demonstrated": [
                    "Personalized financial advice",
                    "Indian financial context",
                    "Actionable recommendations",
                    "Risk assessment",
                    "Goal-based planning",
                ],
            },
        }

    except Exception as e:
        logger.error(f"Error in financial advisor demo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Trust Scoring Endpoints
@app.post("/predict/trust-score", response_model=TrustScoreResponse)
async def calculate_trust_score(request: TrustScoreRequest):
    """Calculate comprehensive trust score for a user based on transaction history"""
    start_time = datetime.now()

    try:
        # Convert Pydantic models to dictionaries for the trust engine
        transactions = [
            {
                "user_id": txn.user_id,
                "merchant_name": txn.merchant_name,
                "amount": txn.amount,
                "timestamp": txn.timestamp or datetime.now().isoformat(),
            }
            for txn in request.transactions
        ]

        # Calculate trust score
        trust_score = trust_engine.calculate_trust_score(request.user_id, transactions)

        # Get risk profile with recommendations
        risk_profile = trust_engine.get_user_risk_profile(request.user_id, transactions)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return TrustScoreResponse(
            user_id=trust_score.user_id,
            overall_score=trust_score.overall_score,
            trust_level=trust_score.trust_level.value,
            risk_flags=[flag.value for flag in trust_score.risk_flags],
            component_scores=trust_score.component_scores,
            confidence=trust_score.confidence,
            recommendations=risk_profile["recommendations"],
            monitoring_frequency=risk_profile["monitoring_frequency"],
            transaction_limits=risk_profile["transaction_limits"],
            risk_level=risk_profile["risk_level"],
            factors=trust_score.factors,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error calculating trust score: {e}")
        raise HTTPException(
            status_code=500, detail=f"Trust score calculation failed: {str(e)}"
        )


@app.get("/demo/trust-scoring")
async def demo_trust_scoring():
    """Demonstrate trust scoring capabilities with various user scenarios"""
    try:
        demo_scenarios = [
            {
                "user_id": "trusted_user_001",
                "description": "High-trust user with consistent spending patterns",
                "transaction_count": 25,
            },
            {
                "user_id": "new_user_002",
                "description": "New user with limited transaction history",
                "transaction_count": 5,
            },
            {
                "user_id": "suspicious_user_003",
                "description": "User with unusual spending patterns",
                "transaction_count": 30,
            },
        ]

        demo_results = []

        for scenario in demo_scenarios:
            start_time = datetime.now()

            # Generate sample transactions based on scenario
            sample_df = data_generator.generate_transactions(
                n_users=1, n_transactions=scenario["transaction_count"]
            )

            # Convert DataFrame to list of dicts and update user IDs
            sample_transactions = sample_df.to_dict("records")
            for txn in sample_transactions:
                txn["user_id"] = scenario["user_id"]

            # Convert to trust engine format
            transactions = [
                {
                    "user_id": txn["user_id"],
                    "merchant_name": txn["merchant_name"],
                    "amount": txn["amount"],
                    "timestamp": txn["timestamp"],
                }
                for txn in sample_transactions
            ]

            # Calculate trust score
            trust_score = trust_engine.calculate_trust_score(
                scenario["user_id"], transactions
            )
            risk_profile = trust_engine.get_user_risk_profile(
                scenario["user_id"], transactions
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            demo_results.append(
                {
                    "scenario": scenario,
                    "trust_analysis": {
                        "overall_score": trust_score.overall_score,
                        "trust_level": trust_score.trust_level.value,
                        "risk_flags": [flag.value for flag in trust_score.risk_flags],
                        "component_scores": trust_score.component_scores,
                        "confidence": trust_score.confidence,
                        "risk_level": risk_profile["risk_level"],
                        "recommended_limits": risk_profile["transaction_limits"],
                        "monitoring_frequency": risk_profile["monitoring_frequency"],
                    },
                    "key_factors": trust_score.factors,
                    "processing_time_ms": processing_time,
                }
            )

        return {
            "demo_title": "Trust Scoring Engine - Multi-Dimensional Risk Assessment",
            "demo_description": "Advanced trust scoring using transaction history, network analysis, behavioral patterns, and community reputation",
            "demo_results": demo_results,
            "model_info": {
                "engine": "TrustScoringEngine",
                "version": "1.0.0",
                "target_accuracy": "88%+ fraud detection",
                "target_latency": "<100ms",
                "components": [
                    "Transaction History Analysis (40% weight)",
                    "Network Analysis (25% weight)",
                    "Behavioral Patterns (20% weight)",
                    "Community Reputation (15% weight)",
                ],
            },
            "summary": {
                "total_scenarios": len(demo_results),
                "average_processing_time": sum(
                    r["processing_time_ms"] for r in demo_results
                )
                / len(demo_results),
                "trust_levels_demonstrated": list(
                    set(r["trust_analysis"]["trust_level"] for r in demo_results)
                ),
                "risk_levels_demonstrated": list(
                    set(r["trust_analysis"]["risk_level"] for r in demo_results)
                ),
                "features_demonstrated": [
                    "Multi-dimensional trust scoring",
                    "Risk flag identification",
                    "Dynamic transaction limits",
                    "Behavioral anomaly detection",
                    "Confidence-based scoring",
                ],
            },
        }

    except Exception as e:
        logger.error(f"Error in trust scoring demo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Voice Intelligence Endpoints
@app.post("/predict/voice-analysis", response_model=VoiceAnalysisResponse)
async def analyze_voice_data(request: VoiceAnalysisRequest):
    """Analyze voice data for financial insights and emotional context"""
    start_time = datetime.now()

    try:
        # Handle audio data conversion
        audio_data = None
        if request.audio_data:
            import base64

            audio_data = base64.b64decode(request.audio_data)
        elif request.audio_url:
            # For now, raise an error - URL handling would require additional implementation
            raise HTTPException(
                status_code=400,
                detail="Audio URL processing not yet implemented. Please use base64 audio_data.",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either audio_data (base64) or audio_url must be provided",
            )

        # Analyze voice data using the voice intelligence engine
        # Convert processing_mode string to VoiceProcessingMode enum
        from models.voice_intelligence import VoiceProcessingMode

        preferred_mode = VoiceProcessingMode.LOCAL_WHISPER  # default

        if request.processing_mode == "openai_api":
            preferred_mode = VoiceProcessingMode.OPENAI_API
        elif request.processing_mode == "google_stt":
            preferred_mode = VoiceProcessingMode.GOOGLE_STT
        elif request.processing_mode == "local_whisper":
            preferred_mode = VoiceProcessingMode.LOCAL_WHISPER

        # Debug logging
        import os

        logger.info(f"Processing voice analysis with mode: {preferred_mode}")
        logger.info(f"OpenAI API Key available: {bool(os.getenv('OPENAI_API_KEY'))}")

        # Create a fresh Voice Intelligence instance to ensure environment variables are loaded
        from models.voice_intelligence import VoiceIntelligenceEngine

        fresh_voice_intelligence = VoiceIntelligenceEngine()

        result = await fresh_voice_intelligence.process_voice_memo(
            audio_data=audio_data,
            user_id=request.user_id,
            context=request.context,
            preferred_mode=preferred_mode,
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return VoiceAnalysisResponse(
            user_id=request.user_id,
            transcript=result.transcript,
            confidence=result.confidence,
            processing_mode=result.processing_mode.value,
            duration_seconds=result.duration_seconds,
            entities=result.entities,
            emotions={
                emotion.value: score for emotion, score in result.emotions.items()
            },
            sentiment_score=result.sentiment_score,
            insights=result.insights,
            spending_mentions=result.spending_mentions,
            financial_keywords=result.financial_keywords,
            processing_time_ms=processing_time,
            language=result.language,
        )

    except Exception as e:
        logger.error(f"Error analyzing voice data: {e}")
        raise HTTPException(status_code=500, detail=f"Voice analysis failed: {str(e)}")


@app.get("/demo/voice-intelligence")
async def demo_voice_intelligence():
    """Demonstrate voice intelligence capabilities with sample scenarios"""
    try:
        demo_scenarios = [
            {
                "scenario_name": "Happy Food Delivery Payment",
                "transcript": "I just paid fifteen hundred rupees to Swiggy for dinner, and it was absolutely delicious!",
                "context": {"transaction_amount": 1500, "merchant_name": "Swiggy"},
                "expected_insights": [
                    "positive_experience",
                    "food_delivery_satisfaction",
                ],
            },
            {
                "scenario_name": "Stressed Bill Payment",
                "transcript": "Ugh, I had to pay two thousand five hundred for electricity bill again, this is getting expensive",
                "context": {
                    "transaction_amount": 2500,
                    "merchant_name": "Electricity Board",
                },
                "expected_insights": ["financial_stress", "utility_cost_concern"],
            },
            {
                "scenario_name": "Excited Shopping Purchase",
                "transcript": "I'm so excited! Just bought this amazing dress for three thousand rupees from Myntra",
                "context": {"transaction_amount": 3000, "merchant_name": "Myntra"},
                "expected_insights": [
                    "positive_shopping_experience",
                    "discretionary_spending",
                ],
            },
            {
                "scenario_name": "Concerned Investment Discussion",
                "transcript": "I'm thinking about investing ten thousand in mutual funds, but I'm not sure if it's the right time",
                "context": {"transaction_amount": 10000, "category": "investment"},
                "expected_insights": [
                    "investment_hesitation",
                    "financial_planning_interest",
                ],
            },
        ]

        demo_results = []
        start_time = datetime.now()

        for scenario in demo_scenarios:
            # Simulate voice analysis for demo purposes
            demo_result = await voice_intelligence.analyze_text_for_demo(
                user_id=f"demo_user_{len(demo_results)+1}",
                transcript=scenario["transcript"],
                context=scenario["context"],
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            demo_results.append(
                {
                    "scenario": scenario,
                    "voice_analysis": {
                        "transcript": demo_result.transcript,
                        "confidence": demo_result.confidence,
                        "processing_mode": demo_result.processing_mode.value,
                        "duration_seconds": demo_result.duration_seconds,
                        "entities": demo_result.entities,
                        "emotions": {
                            emotion.value: score
                            for emotion, score in demo_result.emotions.items()
                        },
                        "sentiment_score": demo_result.sentiment_score,
                        "insights": demo_result.insights,
                        "spending_mentions": demo_result.spending_mentions,
                        "financial_keywords": demo_result.financial_keywords,
                        "language": demo_result.language,
                    },
                    "processing_time_ms": processing_time,
                }
            )

        return {
            "demo_title": "Voice Intelligence Engine - Multi-Modal Speech Analysis",
            "demo_description": "Advanced voice processing with financial context understanding, emotion analysis, and behavioral insights",
            "demo_results": demo_results,
            "model_info": {
                "engine": "VoiceIntelligenceEngine",
                "version": "1.0.0",
                "target_accuracy": "95%+ transcription, 85%+ emotion detection",
                "target_latency": "<2s per analysis",
                "capabilities": [
                    "Multi-modal Speech-to-Text (Whisper, OpenAI, Google)",
                    "Financial Entity Extraction",
                    "Emotion and Stress Analysis",
                    "Voice Characteristics Analysis",
                    "Financial Insight Generation",
                ],
            },
            "summary": {
                "total_scenarios": len(demo_results),
                "average_processing_time": sum(
                    r["processing_time_ms"] for r in demo_results
                )
                / len(demo_results),
                "emotions_detected": list(
                    set(
                        emotion
                        for r in demo_results
                        for emotion in r["voice_analysis"]["emotions"].keys()
                    )
                ),
                "entity_types_detected": list(
                    set(
                        entity_type
                        for r in demo_results
                        for entity_type in r["voice_analysis"]["entities"].keys()
                    )
                ),
                "average_sentiment": sum(
                    r["voice_analysis"]["sentiment_score"] for r in demo_results
                )
                / len(demo_results),
                "languages_detected": list(
                    set(r["voice_analysis"]["language"] for r in demo_results)
                ),
                "features_demonstrated": [
                    "Real-time voice transcription",
                    "Emotional state detection",
                    "Financial entity extraction",
                    "Context-aware insights",
                    "Sentiment analysis",
                    "Spending pattern detection",
                ],
            },
        }

    except Exception as e:
        logger.error(f"Error in voice intelligence demo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/merchant-analysis", response_model=MerchantAnalysisResponse)
async def analyze_merchant_intelligence(request: MerchantAnalysisRequest):
    """Analyze merchant behavior, risk, and opportunities"""
    start_time = datetime.now()

    try:
        # Perform comprehensive merchant analysis
        analysis = await merchant_intelligence.analyze_merchant(
            merchant_vpa=request.merchant_vpa,
            user_id=request.user_id,
            transaction_history=request.transaction_history,
            market_data=request.market_data,
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return MerchantAnalysisResponse(
            user_id=request.user_id,
            merchant_vpa=request.merchant_vpa,
            risk_level=analysis.merchant_profile.risk_level.value,
            trust_score=analysis.merchant_profile.trust_score,
            insights=[
                {
                    "type": insight.insight_type,
                    "title": insight.title,
                    "description": insight.description,
                    "impact_score": insight.impact_score,
                    "confidence": insight.confidence,
                    "recommendation": insight.recommendation,
                    "estimated_savings": insight.estimated_savings,
                    "risk_mitigation": insight.risk_mitigation,
                }
                for insight in analysis.insights
            ],
            recommendations=analysis.recommendations,
            pricing_analysis=analysis.pricing_insights,
            behavioral_patterns=analysis.behavioral_patterns,
            competitive_analysis=analysis.competitive_analysis,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error analyzing merchant {request.merchant_vpa}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Merchant analysis failed: {str(e)}"
        )


@app.get("/demo/merchant-intelligence")
async def demo_merchant_intelligence():
    """Demonstrate merchant intelligence capabilities"""
    try:
        # Create demo merchant scenarios
        demo_scenarios = [
            {
                "scenario_name": "High-Frequency Food Delivery",
                "merchant_vpa": "swiggy@paytm",
                "transaction_history": [
                    {
                        "amount": 450.0,
                        "timestamp": "2023-12-01T19:30:00",
                        "status": "success",
                    },
                    {
                        "amount": 380.0,
                        "timestamp": "2023-12-03T20:15:00",
                        "status": "success",
                    },
                    {
                        "amount": 520.0,
                        "timestamp": "2023-12-05T19:45:00",
                        "status": "success",
                    },
                    {
                        "amount": 410.0,
                        "timestamp": "2023-12-07T20:00:00",
                        "status": "success",
                    },
                    {
                        "amount": 470.0,
                        "timestamp": "2023-12-09T19:20:00",
                        "status": "success",
                    },
                ],
                "expected_insights": [
                    "High frequency merchant - consider budget caps",
                    "Consistent pricing - reliable service",
                    "Peak evening orders - predictable pattern",
                ],
            },
            {
                "scenario_name": "Overpriced Retail Merchant",
                "merchant_vpa": "expensivestore@upi",
                "transaction_history": [
                    {
                        "amount": 2500.0,
                        "timestamp": "2023-11-15T14:30:00",
                        "status": "success",
                    },
                    {
                        "amount": 1800.0,
                        "timestamp": "2023-11-20T15:00:00",
                        "status": "success",
                        "disputed": True,
                    },
                    {
                        "amount": 3200.0,
                        "timestamp": "2023-11-25T16:30:00",
                        "status": "success",
                    },
                ],
                "market_data": {"average_transaction": 1500.0},
                "expected_insights": [
                    "Pricing above market average",
                    "Dispute history suggests service issues",
                    "Consider alternative merchants",
                ],
            },
            {
                "scenario_name": "Trusted Regular Merchant",
                "merchant_vpa": "grocerystore@payments",
                "transaction_history": [
                    {
                        "amount": 1200.0,
                        "timestamp": "2023-11-01T10:30:00",
                        "status": "success",
                    },
                    {
                        "amount": 980.0,
                        "timestamp": "2023-11-08T11:00:00",
                        "status": "success",
                    },
                    {
                        "amount": 1150.0,
                        "timestamp": "2023-11-15T10:45:00",
                        "status": "success",
                    },
                    {
                        "amount": 1050.0,
                        "timestamp": "2023-11-22T11:15:00",
                        "status": "success",
                    },
                    {
                        "amount": 1300.0,
                        "timestamp": "2023-11-29T10:20:00",
                        "status": "success",
                    },
                ],
                "expected_insights": [
                    "High loyalty score - reliable merchant",
                    "Consistent weekly pattern",
                    "Good candidate for loyalty programs",
                ],
            },
        ]

        demo_results = []
        for scenario in demo_scenarios:
            # Simulate merchant analysis
            analysis = await merchant_intelligence.analyze_merchant(
                merchant_vpa=scenario["merchant_vpa"],
                user_id="demo_user",
                transaction_history=scenario["transaction_history"],
                market_data=scenario.get("market_data"),
            )

            demo_results.append(
                {
                    "scenario": scenario["scenario_name"],
                    "merchant": scenario["merchant_vpa"],
                    "risk_level": analysis.merchant_profile.risk_level.value,
                    "trust_score": round(analysis.merchant_profile.trust_score, 2),
                    "key_insights": [
                        insight.title for insight in analysis.insights[:3]
                    ],
                    "top_recommendation": (
                        analysis.recommendations[0]
                        if analysis.recommendations
                        else "Monitor transactions"
                    ),
                    "pricing_fairness": round(
                        analysis.merchant_profile.pricing_fairness, 2
                    ),
                    "loyalty_score": round(analysis.merchant_profile.loyalty_score, 2),
                }
            )

        return {
            "service": "Merchant Intelligence",
            "version": "1.0.0",
            "demo_scenarios": demo_results,
            "capabilities": [
                "Risk assessment and trust scoring",
                "Pricing fairness analysis",
                "Behavioral pattern detection",
                "Competitive analysis",
                "Loyalty and frequency scoring",
                "Actionable recommendations",
                "Market comparison insights",
            ],
            "use_cases": [
                "Merchant risk evaluation",
                "Spending optimization",
                "Fraud detection",
                "Loyalty program recommendations",
                "Budget management",
                "Alternative merchant suggestions",
            ],
        }

    except Exception as e:
        logger.error(f"Error in merchant intelligence demo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
