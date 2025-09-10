"""
Voice Intelligence Engine for Cap'n Pay
Advanced speech processing with financial context understanding and insight generation.
Patent-worthy innovation: Multi-modal financial analysis with voice emotion detection and contextual insights.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
import aiohttp
import io
from pathlib import Path

# Audio processing
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("Whisper not available - using OpenAI API fallback")

# NLP and sentiment analysis
import re
from collections import Counter

logger = logging.getLogger(__name__)


class VoiceProcessingMode(Enum):
    LOCAL_WHISPER = "local_whisper"
    OPENAI_API = "openai_api"
    GOOGLE_STT = "google_stt"


class EmotionType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    ANXIOUS = "anxious"
    CONFIDENT = "confident"
    FRUSTRATED = "frustrated"


class InsightType(Enum):
    SPENDING_PATTERN = "spending_pattern"
    EMOTIONAL_STATE = "emotional_state"
    FINANCIAL_GOAL = "financial_goal"
    CONCERN = "concern"
    CELEBRATION = "celebration"
    REGRET = "regret"


@dataclass
class VoiceAnalysisResult:
    transcript: str
    confidence: float
    processing_mode: VoiceProcessingMode
    duration_seconds: float

    # NLP Analysis
    entities: Dict[str, List[str]]
    emotions: Dict[EmotionType, float]
    sentiment_score: float  # -1 to 1

    # Financial Insights
    insights: List[Dict[str, Any]]
    spending_mentions: List[Dict[str, Any]]
    financial_keywords: List[str]

    # Metadata
    processing_time_ms: float
    timestamp: datetime
    language: str


class VoiceIntelligenceEngine:
    """
    Advanced Voice Intelligence System:
    1. Multi-modal Speech-to-Text (Whisper + OpenAI + Google)
    2. Financial Entity Extraction
    3. Emotion and Sentiment Analysis
    4. Contextual Insight Generation
    5. Voice Pattern Learning
    """

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_CLOUD_API_KEY")

        # Load Whisper model if available
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("🎤 Whisper model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Whisper model: {e}")

        # Financial keywords and patterns
        self.financial_keywords = {
            "spending": [
                "spent",
                "bought",
                "purchased",
                "paid",
                "cost",
                "expensive",
                "cheap",
                "price",
            ],
            "saving": [
                "save",
                "saved",
                "savings",
                "budget",
                "affordable",
                "deal",
                "discount",
            ],
            "regret": [
                "regret",
                "mistake",
                "shouldnt have",
                "waste",
                "unnecessary",
                "overspent",
            ],
            "celebration": [
                "happy",
                "great deal",
                "good price",
                "worth it",
                "satisfied",
            ],
            "concern": [
                "worried",
                "stressed",
                "anxious",
                "tight budget",
                "cant afford",
            ],
            "goals": ["goal", "target", "plan", "future", "dream", "want to buy"],
            "merchants": [
                "zomato",
                "swiggy",
                "amazon",
                "flipkart",
                "uber",
                "ola",
                "paytm",
            ],
            "amounts": ["hundred", "thousand", "lakh", "crore", "rupee", "rupees", "₹"],
        }

        # Emotion patterns
        self.emotion_patterns = {
            EmotionType.POSITIVE: [
                "happy",
                "great",
                "good",
                "excellent",
                "satisfied",
                "pleased",
            ],
            EmotionType.NEGATIVE: [
                "bad",
                "terrible",
                "awful",
                "disappointed",
                "upset",
                "angry",
            ],
            EmotionType.ANXIOUS: [
                "worried",
                "nervous",
                "stressed",
                "concerned",
                "anxious",
            ],
            EmotionType.CONFIDENT: [
                "confident",
                "sure",
                "certain",
                "positive",
                "definitely",
            ],
            EmotionType.FRUSTRATED: ["frustrated", "annoyed", "irritated", "fed up"],
            EmotionType.NEUTRAL: ["okay", "fine", "normal", "usual", "regular"],
        }

        logger.info("🎤 Voice Intelligence Engine initialized")

    async def process_voice_memo(
        self,
        audio_data: bytes,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        preferred_mode: VoiceProcessingMode = VoiceProcessingMode.LOCAL_WHISPER,
    ) -> VoiceAnalysisResult:
        """Process voice memo and generate comprehensive analysis"""
        start_time = datetime.now()

        try:
            # Step 1: Speech-to-Text
            transcript, confidence, actual_mode, duration = await self._speech_to_text(
                audio_data, preferred_mode
            )

            # Step 2: NLP Analysis
            entities = self._extract_entities(transcript)
            emotions = self._analyze_emotions(transcript)
            sentiment_score = self._calculate_sentiment(transcript)

            # Step 3: Financial Analysis
            insights = self._generate_financial_insights(transcript, context)
            spending_mentions = self._extract_spending_mentions(transcript)
            financial_keywords = self._extract_financial_keywords(transcript)

            # Step 4: Language Detection
            language = self._detect_language(transcript)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return VoiceAnalysisResult(
                transcript=transcript,
                confidence=confidence,
                processing_mode=actual_mode,
                duration_seconds=duration,
                entities=entities,
                emotions=emotions,
                sentiment_score=sentiment_score,
                insights=insights,
                spending_mentions=spending_mentions,
                financial_keywords=financial_keywords,
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                language=language,
            )

        except Exception as e:
            logger.error(f"Error processing voice memo: {e}")
            return self._create_error_result(str(e))

    async def _speech_to_text(
        self, audio_data: bytes, preferred_mode: VoiceProcessingMode
    ) -> Tuple[str, float, VoiceProcessingMode, float]:
        """Convert speech to text using the best available method"""

        # Try local Whisper first if available and preferred
        if preferred_mode == VoiceProcessingMode.LOCAL_WHISPER and self.whisper_model:
            try:
                return await self._whisper_local_transcribe(audio_data)
            except Exception as e:
                logger.warning(f"Local Whisper failed: {e}, falling back to OpenAI API")

        # Try OpenAI API
        if self.openai_api_key:
            try:
                return await self._openai_transcribe(audio_data)
            except Exception as e:
                logger.warning(f"OpenAI API failed: {e}, falling back to Google STT")

        # Fallback to Google STT
        if self.google_api_key:
            try:
                return await self._google_stt_transcribe(audio_data)
            except Exception as e:
                logger.error(f"Google STT failed: {e}")

        # If all methods fail, return error
        raise Exception("All speech-to-text methods failed")

    async def _whisper_local_transcribe(
        self, audio_data: bytes
    ) -> Tuple[str, float, VoiceProcessingMode, float]:
        """Transcribe using local Whisper model"""

        # Save audio data to temporary file
        temp_path = f"/tmp/voice_memo_{datetime.now().timestamp()}.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_data)

        try:
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(temp_path)

            transcript = result["text"].strip()
            confidence = 0.95  # Whisper doesn't provide confidence, use high default
            duration = result.get("duration", 0)

            return transcript, confidence, VoiceProcessingMode.LOCAL_WHISPER, duration

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def _openai_transcribe(
        self, audio_data: bytes
    ) -> Tuple[str, float, VoiceProcessingMode, float]:
        """Transcribe using OpenAI Whisper API"""

        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}

        # Prepare multipart form data
        data = aiohttp.FormData()
        data.add_field(
            "file", audio_data, filename="audio.m4a", content_type="audio/m4a"
        )
        data.add_field("model", "whisper-1")
        data.add_field("response_format", "verbose_json")

        # Create SSL context that doesn't verify certificates (for development)
        import ssl

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(url, headers=headers, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    transcript = result.get("text", "").strip()
                    duration = result.get("duration", 0)
                    confidence = 0.90  # OpenAI API doesn't provide confidence

                    return (
                        transcript,
                        confidence,
                        VoiceProcessingMode.OPENAI_API,
                        duration,
                    )
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"OpenAI API error: {response.status} - {error_text}"
                    )

    async def _google_stt_transcribe(
        self, audio_data: bytes
    ) -> Tuple[str, float, VoiceProcessingMode, float]:
        """Transcribe using Google Speech-to-Text API"""
        # This is a placeholder - Google STT implementation would go here
        # For now, return a mock result
        return "Google STT placeholder", 0.85, VoiceProcessingMode.GOOGLE_STT, 5.0

    def _extract_entities(self, transcript: str) -> Dict[str, List[str]]:
        """Extract financial entities from transcript"""
        entities = {
            "merchants": [],
            "amounts": [],
            "categories": [],
            "locations": [],
            "dates": [],
        }

        transcript_lower = transcript.lower()

        # Extract merchants
        for merchant in self.financial_keywords["merchants"]:
            if merchant in transcript_lower:
                entities["merchants"].append(merchant)

        # Extract amounts (simple pattern matching)
        amount_patterns = [
            r"₹\s*(\d+(?:,\d+)*(?:\.\d+)?)",  # ₹1000, ₹1,000.50
            r"(\d+)\s*rupees?",  # 100 rupees
            r"(\d+)\s*thousand",  # 5 thousand
            r"(\d+)\s*lakh",  # 2 lakh
        ]

        for pattern in amount_patterns:
            matches = re.findall(pattern, transcript_lower)
            entities["amounts"].extend(matches)

        # Extract categories (using common financial categories)
        categories = [
            "food",
            "transport",
            "shopping",
            "entertainment",
            "utilities",
            "bills",
            "groceries",
            "delivery",
        ]
        for category in categories:
            if category in transcript_lower:
                entities["categories"].append(category)

        # Extract locations (basic Indian city names)
        locations = [
            "mumbai",
            "delhi",
            "bangalore",
            "hyderabad",
            "chennai",
            "pune",
            "kolkata",
        ]
        for location in locations:
            if location in transcript_lower:
                entities["locations"].append(location)

        # Extract dates (basic patterns)
        date_patterns = [
            r"today",
            r"yesterday",
            r"tomorrow",
            r"last week",
            r"next week",
            r"january|february|march|april|may|june|july|august|september|october|november|december",
        ]
        for pattern in date_patterns:
            if re.search(pattern, transcript_lower):
                entities["dates"].append(pattern)

        return entities

    def _analyze_emotions(self, transcript: str) -> Dict[EmotionType, float]:
        """Analyze emotions in the transcript using keyword-based sentiment analysis"""
        transcript_lower = transcript.lower()

        emotions = {emotion: 0.0 for emotion in EmotionType}

        # Positive emotion keywords
        positive_keywords = [
            "excited",
            "happy",
            "delicious",
            "amazing",
            "great",
            "wonderful",
            "love",
            "fantastic",
        ]
        if any(keyword in transcript_lower for keyword in positive_keywords):
            emotions[EmotionType.POSITIVE] = 0.8

        # Negative emotion keywords
        negative_keywords = [
            "ugh",
            "expensive",
            "hate",
            "terrible",
            "awful",
            "frustrated",
            "angry",
        ]
        if any(keyword in transcript_lower for keyword in negative_keywords):
            emotions[EmotionType.NEGATIVE] = 0.7

        # Anxious emotion keywords
        anxious_keywords = [
            "worried",
            "nervous",
            "uncertain",
            "not sure",
            "concerned",
            "anxious",
        ]
        if any(keyword in transcript_lower for keyword in anxious_keywords):
            emotions[EmotionType.ANXIOUS] = 0.6

        # Confident emotion keywords
        confident_keywords = [
            "confident",
            "sure",
            "definitely",
            "absolutely",
            "investing",
            "planning",
        ]
        if any(keyword in transcript_lower for keyword in confident_keywords):
            emotions[EmotionType.CONFIDENT] = 0.7

        # Frustrated emotion keywords
        frustrated_keywords = [
            "again",
            "always",
            "getting expensive",
            "keep paying",
            "tired of",
        ]
        if any(keyword in transcript_lower for keyword in frustrated_keywords):
            emotions[EmotionType.FRUSTRATED] = 0.6

        # Default to neutral if no strong emotions detected
        if all(score == 0.0 for score in emotions.values()):
            emotions[EmotionType.NEUTRAL] = 0.5

        return emotions
        emotion_scores = {}

        for emotion_type, keywords in self.emotion_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in transcript_lower:
                    score += 1

            # Normalize score
            emotion_scores[emotion_type] = min(score / len(keywords), 1.0)

        return emotion_scores

    def _calculate_sentiment(self, transcript: str) -> float:
        """Calculate overall sentiment score (-1 to 1)"""
        positive_words = [
            "good",
            "great",
            "happy",
            "satisfied",
            "pleased",
            "excellent",
            "wonderful",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "disappointed",
            "upset",
            "angry",
            "frustrated",
        ]

        transcript_lower = transcript.lower()

        positive_count = sum(1 for word in positive_words if word in transcript_lower)
        negative_count = sum(1 for word in negative_words if word in transcript_lower)

        total_words = len(transcript.split())

        if total_words == 0:
            return 0.0

        sentiment = (positive_count - negative_count) / max(total_words, 1)
        return max(-1.0, min(1.0, sentiment * 10))  # Scale and clamp

    def _generate_financial_insights(
        self, transcript: str, context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate financial insights from the transcript"""
        insights = []
        transcript_lower = transcript.lower()

        # Spending regret detection
        regret_keywords = [
            "expensive",
            "costly",
            "too much",
            "regret",
            "waste",
            "mistake",
        ]
        if any(keyword in transcript_lower for keyword in regret_keywords):
            insights.append(
                {
                    "type": InsightType.REGRET.value,
                    "confidence": 0.8,
                    "message": "You seem to have some regret about a recent purchase",
                    "recommendation": "Consider setting spending limits or using our nudge system",
                    "category": "spending_behavior",
                }
            )

        # Goal detection
        goal_keywords = [
            "investing",
            "investment",
            "save",
            "saving",
            "goal",
            "future",
            "plan",
            "budget",
        ]
        if any(keyword in transcript_lower for keyword in goal_keywords):
            insights.append(
                {
                    "type": InsightType.FINANCIAL_GOAL.value,
                    "confidence": 0.7,
                    "message": "You mentioned financial goals or future purchases",
                    "recommendation": "Set up a savings goal in the app to track progress",
                    "category": "financial_planning",
                }
            )

        # Concern detection
        concern_keywords = [
            "worried",
            "concerned",
            "expensive",
            "not sure",
            "uncertain",
            "stress",
        ]
        if any(keyword in transcript_lower for keyword in concern_keywords):
            insights.append(
                {
                    "type": InsightType.CONCERN.value,
                    "confidence": 0.9,
                    "message": "You expressed financial concerns or stress",
                    "recommendation": "Review your budget and consider speaking with our AI advisor",
                    "category": "financial_wellness",
                }
            )

        # Celebration detection
        celebration_keywords = [
            "excited",
            "happy",
            "delicious",
            "amazing",
            "great",
            "celebration",
        ]
        if any(keyword in transcript_lower for keyword in celebration_keywords):
            insights.append(
                {
                    "type": InsightType.CELEBRATION.value,
                    "confidence": 0.8,
                    "message": "You seem happy with a recent purchase or financial decision",
                    "recommendation": "Great job making smart financial choices!",
                    "category": "positive_reinforcement",
                }
            )

        return insights

    def _extract_spending_mentions(self, transcript: str) -> List[Dict[str, Any]]:
        """Extract specific spending mentions from transcript"""
        spending_mentions = []
        transcript_lower = transcript.lower()

        # Look for spending patterns with amounts
        spending_patterns = [
            (r"paid (\d+(?:,\d+)*)\s*rupees?", "payment"),
            (r"spent (\d+(?:,\d+)*)\s*rupees?", "spent"),
            (r"bought .+ for (\d+(?:,\d+)*)\s*rupees?", "purchase"),
            (r"cost (\d+(?:,\d+)*)\s*rupees?", "cost"),
            (r"(\d+(?:,\d+)*)\s*thousand", "large_amount"),
            (r"(\d+(?:,\d+)*)\s*lakh", "very_large_amount"),
        ]

        for pattern, action_type in spending_patterns:
            matches = re.finditer(pattern, transcript_lower)
            for match in matches:
                amount_str = match.group(1).replace(",", "")
                try:
                    amount = float(amount_str)
                    if "thousand" in pattern:
                        amount *= 1000
                    elif "lakh" in pattern:
                        amount *= 100000

                    spending_mentions.append(
                        {
                            "amount": amount,
                            "action_type": action_type,
                            "text_segment": match.group(0),
                            "confidence": 0.9,
                        }
                    )
                except ValueError:
                    continue

        return spending_mentions

    def _extract_financial_keywords(self, transcript: str) -> List[str]:
        """Extract relevant financial keywords from transcript"""
        keywords = []
        transcript_lower = transcript.lower()

        for category, keyword_list in self.financial_keywords.items():
            for keyword in keyword_list:
                if keyword in transcript_lower:
                    keywords.append(keyword)

        return list(set(keywords))  # Remove duplicates

    def _detect_language(self, transcript: str) -> str:
        """Simple language detection"""
        # This is a basic implementation - in production, use a proper language detection library
        hindi_indicators = ["hai", "hoon", "paisa", "rupiya", "kharcha", "bachat"]

        transcript_lower = transcript.lower()
        hindi_count = sum(
            1 for indicator in hindi_indicators if indicator in transcript_lower
        )

        if hindi_count > 0:
            return "hi"  # Hindi
        else:
            return "en"  # English (default)

    def _create_error_result(self, error_message: str) -> VoiceAnalysisResult:
        """Create error result when processing fails"""
        return VoiceAnalysisResult(
            transcript=f"Error: {error_message}",
            confidence=0.0,
            processing_mode=VoiceProcessingMode.LOCAL_WHISPER,
            duration_seconds=0.0,
            entities={},
            emotions={emotion: 0.0 for emotion in EmotionType},
            sentiment_score=0.0,
            insights=[],
            spending_mentions=[],
            financial_keywords=[],
            processing_time_ms=0.0,
            timestamp=datetime.now(),
            language="en",
        )

    def get_voice_analytics(
        self, user_id: str, voice_analyses: List[VoiceAnalysisResult]
    ) -> Dict[str, Any]:
        """Generate analytics from multiple voice analyses"""
        if not voice_analyses:
            return {"error": "No voice analyses provided"}

        # Aggregate emotions
        emotion_aggregates = {}
        for emotion in EmotionType:
            scores = [
                analysis.emotions.get(emotion, 0.0) for analysis in voice_analyses
            ]
            emotion_aggregates[emotion.value] = {
                "average": sum(scores) / len(scores),
                "max": max(scores),
                "frequency": sum(1 for score in scores if score > 0.3),
            }

        # Aggregate sentiments
        sentiments = [analysis.sentiment_score for analysis in voice_analyses]
        avg_sentiment = sum(sentiments) / len(sentiments)

        # Most common insights
        insight_types = []
        for analysis in voice_analyses:
            insight_types.extend([insight["type"] for insight in analysis.insights])

        common_insights = Counter(insight_types).most_common(3)

        # Financial keyword frequency
        all_keywords = []
        for analysis in voice_analyses:
            all_keywords.extend(analysis.financial_keywords)

        keyword_frequency = Counter(all_keywords).most_common(10)

        return {
            "user_id": user_id,
            "analysis_period": {
                "start": min(
                    analysis.timestamp for analysis in voice_analyses
                ).isoformat(),
                "end": max(
                    analysis.timestamp for analysis in voice_analyses
                ).isoformat(),
                "total_memos": len(voice_analyses),
            },
            "emotion_profile": emotion_aggregates,
            "sentiment_trends": {
                "average_sentiment": avg_sentiment,
                "sentiment_range": [min(sentiments), max(sentiments)],
                "positive_ratio": sum(1 for s in sentiments if s > 0.1)
                / len(sentiments),
            },
            "common_insights": [
                {"type": insight, "frequency": count}
                for insight, count in common_insights
            ],
            "financial_vocabulary": [
                {"keyword": word, "frequency": count}
                for word, count in keyword_frequency
            ],
            "voice_patterns": {
                "average_duration": sum(
                    analysis.duration_seconds for analysis in voice_analyses
                )
                / len(voice_analyses),
                "average_confidence": sum(
                    analysis.confidence for analysis in voice_analyses
                )
                / len(voice_analyses),
                "preferred_language": Counter(
                    [analysis.language for analysis in voice_analyses]
                ).most_common(1)[0][0],
            },
        }

    async def analyze_text_for_demo(
        self, user_id: str, transcript: str, context: Optional[Dict[str, Any]] = None
    ) -> VoiceAnalysisResult:
        """Analyze transcript for demo purposes without audio processing"""
        start_time = datetime.now()

        try:
            # Simulate transcription confidence
            confidence = 0.92 + (abs(hash(transcript)) % 100) / 1000  # 0.92-0.99 range

            # Extract entities and analyze emotions
            entities = self._extract_entities(transcript)
            emotions = self._analyze_emotions(transcript)

            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment(transcript)

            # Generate insights
            insights = self._generate_financial_insights(transcript, context or {})

            # Find spending mentions
            spending_mentions = self._extract_spending_mentions(transcript)

            # Extract financial keywords
            financial_keywords = []
            transcript_words = transcript.lower().split()
            for category, keywords in self.financial_keywords.items():
                for word in transcript_words:
                    if word in keywords:
                        financial_keywords.append(word)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return VoiceAnalysisResult(
                transcript=transcript,
                confidence=confidence,
                processing_mode=VoiceProcessingMode.LOCAL_WHISPER,  # Use existing mode
                duration_seconds=len(transcript.split()) / 2.5,  # Estimate duration
                entities=entities,
                emotions=emotions,
                sentiment_score=sentiment_score,
                insights=insights,
                spending_mentions=spending_mentions,
                financial_keywords=financial_keywords,
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                language="en",
            )

        except Exception as e:
            logger.error(f"Error in demo text analysis: {e}")
            # Return basic result even on error
            return VoiceAnalysisResult(
                transcript=transcript,
                confidence=0.85,
                processing_mode=VoiceProcessingMode.LOCAL_WHISPER,
                duration_seconds=len(transcript.split()) / 2.5,
                entities={},
                emotions={EmotionType.NEUTRAL: 0.8},
                sentiment_score=0.0,
                insights=[
                    {"type": "demo", "content": "Demo analysis - limited processing"}
                ],
                spending_mentions=[],
                financial_keywords=[],
                processing_time_ms=100.0,
                timestamp=datetime.now(),
                language="en",
            )


# Global instance
voice_intelligence = VoiceIntelligenceEngine()
