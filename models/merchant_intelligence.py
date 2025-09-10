"""
Merchant Intelligence Engine - 6th AI Component
Advanced merchant behavior analysis, risk assessment, and relationship optimization
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class MerchantRiskLevel(Enum):
    """Merchant risk classification levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MerchantCategory(Enum):
    """Merchant business categories"""

    FOOD_DELIVERY = "food_delivery"
    GROCERY = "grocery"
    RETAIL = "retail"
    TRANSPORT = "transport"
    ENTERTAINMENT = "entertainment"
    UTILITIES = "utilities"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    OTHER = "other"


@dataclass
class MerchantProfile:
    """Comprehensive merchant profile"""

    merchant_id: str
    vpa_address: str
    business_name: str
    category: MerchantCategory
    risk_level: MerchantRiskLevel
    trust_score: float  # 0.0 to 1.0
    transaction_count: int
    total_volume: float
    avg_transaction_size: float
    frequency_score: float  # How often users transact
    loyalty_score: float  # User retention rate
    pricing_fairness: float  # Price competitiveness
    service_quality: float  # Based on user feedback
    compliance_score: float  # Regulatory compliance
    growth_trend: float  # Business growth indicator
    peak_hours: List[int]  # Hours of peak activity
    seasonal_patterns: Dict[str, float]  # Monthly patterns
    user_sentiment: float  # Overall user sentiment
    dispute_rate: float  # Dispute frequency
    refund_rate: float  # Refund frequency
    last_updated: datetime


@dataclass
class MerchantInsight:
    """Actionable merchant insights"""

    insight_type: str
    title: str
    description: str
    impact_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    recommendation: str
    estimated_savings: Optional[float] = None
    risk_mitigation: Optional[str] = None


@dataclass
class MerchantAnalysis:
    """Complete merchant analysis result"""

    merchant_profile: MerchantProfile
    insights: List[MerchantInsight]
    risk_factors: List[str]
    opportunities: List[str]
    recommendations: List[str]
    competitive_analysis: Dict[str, Any]
    pricing_insights: Dict[str, Any]
    behavioral_patterns: Dict[str, Any]


class MerchantIntelligenceEngine:
    """Advanced merchant intelligence and analysis system"""

    def __init__(self):
        """Initialize the merchant intelligence engine"""
        self.scaler = StandardScaler()
        self.merchant_clusters = None
        self.category_benchmarks = {}
        logger.info("Merchant Intelligence Engine initialized")

    async def analyze_merchant(
        self,
        merchant_vpa: str,
        user_id: str,
        transaction_history: List[Dict[str, Any]],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> MerchantAnalysis:
        """Perform comprehensive merchant analysis"""
        try:
            # Build merchant profile
            profile = await self._build_merchant_profile(
                merchant_vpa, transaction_history, market_data
            )

            # Generate insights
            insights = await self._generate_merchant_insights(
                profile, transaction_history
            )

            # Risk assessment
            risk_factors = self._assess_risk_factors(profile, transaction_history)

            # Identify opportunities
            opportunities = self._identify_opportunities(profile, market_data)

            # Generate recommendations
            recommendations = self._generate_recommendations(profile, insights)

            # Competitive analysis
            competitive_analysis = await self._perform_competitive_analysis(profile)

            # Pricing insights
            pricing_insights = self._analyze_pricing_patterns(
                profile, transaction_history
            )

            # Behavioral patterns
            behavioral_patterns = self._analyze_behavioral_patterns(transaction_history)

            return MerchantAnalysis(
                merchant_profile=profile,
                insights=insights,
                risk_factors=risk_factors,
                opportunities=opportunities,
                recommendations=recommendations,
                competitive_analysis=competitive_analysis,
                pricing_insights=pricing_insights,
                behavioral_patterns=behavioral_patterns,
            )

        except Exception as e:
            logger.error(f"Error analyzing merchant {merchant_vpa}: {e}")
            raise

    async def _build_merchant_profile(
        self,
        merchant_vpa: str,
        transactions: List[Dict[str, Any]],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> MerchantProfile:
        """Build comprehensive merchant profile"""

        # Extract transaction metrics
        total_transactions = len(transactions)
        total_volume = sum(tx.get("amount", 0) for tx in transactions)
        avg_transaction = total_volume / max(total_transactions, 1)

        # Analyze transaction patterns
        frequency_score = self._calculate_frequency_score(transactions)
        loyalty_score = self._calculate_loyalty_score(transactions)

        # Risk assessment
        risk_level = self._assess_merchant_risk(transactions)
        trust_score = self._calculate_trust_score(transactions, risk_level)

        # Business category detection
        category = self._detect_merchant_category(merchant_vpa, transactions)

        # Time-based patterns
        peak_hours = self._identify_peak_hours(transactions)
        seasonal_patterns = self._analyze_seasonal_patterns(transactions)

        # Quality metrics
        dispute_rate = self._calculate_dispute_rate(transactions)
        refund_rate = self._calculate_refund_rate(transactions)
        user_sentiment = self._analyze_user_sentiment(transactions)

        # Business metrics
        growth_trend = self._calculate_growth_trend(transactions)
        pricing_fairness = self._assess_pricing_fairness(transactions, market_data)
        service_quality = self._assess_service_quality(transactions)
        compliance_score = self._assess_compliance(merchant_vpa, transactions)

        return MerchantProfile(
            merchant_id=merchant_vpa.split("@")[0],
            vpa_address=merchant_vpa,
            business_name=self._extract_business_name(merchant_vpa),
            category=category,
            risk_level=risk_level,
            trust_score=trust_score,
            transaction_count=total_transactions,
            total_volume=total_volume,
            avg_transaction_size=avg_transaction,
            frequency_score=frequency_score,
            loyalty_score=loyalty_score,
            pricing_fairness=pricing_fairness,
            service_quality=service_quality,
            compliance_score=compliance_score,
            growth_trend=growth_trend,
            peak_hours=peak_hours,
            seasonal_patterns=seasonal_patterns,
            user_sentiment=user_sentiment,
            dispute_rate=dispute_rate,
            refund_rate=refund_rate,
            last_updated=datetime.now(),
        )

    async def _generate_merchant_insights(
        self, profile: MerchantProfile, transactions: List[Dict[str, Any]]
    ) -> List[MerchantInsight]:
        """Generate actionable merchant insights"""
        insights = []

        # Pricing optimization insights
        if profile.pricing_fairness < 0.7:
            insights.append(
                MerchantInsight(
                    insight_type="pricing_optimization",
                    title="Overpriced Merchant Detected",
                    description=f"This merchant charges {(1-profile.pricing_fairness)*100:.1f}% above market average",
                    impact_score=0.8,
                    confidence=0.9,
                    recommendation="Consider finding alternative merchants or negotiate pricing",
                    estimated_savings=profile.avg_transaction_size * 0.15,
                )
            )

        # Frequency pattern insights
        if profile.frequency_score > 0.8:
            insights.append(
                MerchantInsight(
                    insight_type="spending_pattern",
                    title="High-Frequency Merchant",
                    description="You transact with this merchant very frequently",
                    impact_score=0.7,
                    confidence=0.95,
                    recommendation="Consider setting up automatic caps or budgets for this merchant",
                    risk_mitigation="Monitor for overspending patterns",
                )
            )

        # Trust and risk insights
        if profile.risk_level in [MerchantRiskLevel.HIGH, MerchantRiskLevel.CRITICAL]:
            insights.append(
                MerchantInsight(
                    insight_type="risk_warning",
                    title="High-Risk Merchant Alert",
                    description=f"This merchant has elevated risk indicators (Trust Score: {profile.trust_score:.2f})",
                    impact_score=0.9,
                    confidence=0.85,
                    recommendation="Exercise caution and monitor transactions closely",
                    risk_mitigation="Consider transaction limits and increased verification",
                )
            )

        # Growth trend insights
        if profile.growth_trend > 0.5:
            insights.append(
                MerchantInsight(
                    insight_type="market_opportunity",
                    title="Growing Business Opportunity",
                    description="This merchant shows strong growth patterns",
                    impact_score=0.6,
                    confidence=0.8,
                    recommendation="Potential for loyalty programs or bulk discounts",
                )
            )

        # Service quality insights
        if profile.service_quality < 0.6:
            insights.append(
                MerchantInsight(
                    insight_type="service_warning",
                    title="Service Quality Concern",
                    description=f"Service quality indicators below average ({profile.service_quality:.2f})",
                    impact_score=0.7,
                    confidence=0.8,
                    recommendation="Monitor service quality and consider alternatives",
                    risk_mitigation="Document any service issues for dispute resolution",
                )
            )

        return insights

    def _assess_merchant_risk(
        self, transactions: List[Dict[str, Any]]
    ) -> MerchantRiskLevel:
        """Assess merchant risk level"""
        risk_factors = 0

        # High transaction variance
        amounts = [tx.get("amount", 0) for tx in transactions]
        if len(amounts) > 1:
            cv = np.std(amounts) / max(np.mean(amounts), 1)
            if cv > 2.0:
                risk_factors += 1

        # Failed transactions
        failed_count = sum(1 for tx in transactions if tx.get("status") == "failed")
        failure_rate = failed_count / max(len(transactions), 1)
        if failure_rate > 0.1:
            risk_factors += 1

        # Unusual timing patterns
        hours = [
            datetime.fromisoformat(tx.get("timestamp", "2023-01-01T12:00:00")).hour
            for tx in transactions
            if tx.get("timestamp")
        ]
        if hours:
            night_transactions = sum(1 for h in hours if h < 6 or h > 22)
            if night_transactions / len(hours) > 0.3:
                risk_factors += 1

        # Map risk factors to levels
        if risk_factors >= 3:
            return MerchantRiskLevel.CRITICAL
        elif risk_factors == 2:
            return MerchantRiskLevel.HIGH
        elif risk_factors == 1:
            return MerchantRiskLevel.MEDIUM
        else:
            return MerchantRiskLevel.LOW

    def _calculate_trust_score(
        self, transactions: List[Dict[str, Any]], risk_level: MerchantRiskLevel
    ) -> float:
        """Calculate merchant trust score"""
        base_score = 0.8

        # Adjust based on risk level
        risk_adjustments = {
            MerchantRiskLevel.LOW: 0.1,
            MerchantRiskLevel.MEDIUM: 0.0,
            MerchantRiskLevel.HIGH: -0.2,
            MerchantRiskLevel.CRITICAL: -0.4,
        }

        trust_score = base_score + risk_adjustments[risk_level]

        # Success rate adjustment
        if transactions:
            success_count = sum(
                1 for tx in transactions if tx.get("status") == "success"
            )
            success_rate = success_count / len(transactions)
            trust_score = trust_score * success_rate

        return max(0.0, min(1.0, trust_score))

    def _detect_merchant_category(
        self, vpa: str, transactions: List[Dict[str, Any]]
    ) -> MerchantCategory:
        """Detect merchant business category"""
        vpa_lower = vpa.lower()

        # VPA-based detection
        category_keywords = {
            MerchantCategory.FOOD_DELIVERY: [
                "swiggy",
                "zomato",
                "ubereats",
                "foodpanda",
                "food",
            ],
            MerchantCategory.GROCERY: [
                "bigbasket",
                "grofers",
                "blinkit",
                "zepto",
                "grocery",
                "fresh",
            ],
            MerchantCategory.RETAIL: [
                "amazon",
                "flipkart",
                "myntra",
                "ajio",
                "store",
                "mart",
            ],
            MerchantCategory.TRANSPORT: [
                "uber",
                "ola",
                "rapido",
                "metro",
                "taxi",
                "transport",
            ],
            MerchantCategory.ENTERTAINMENT: [
                "bookmyshow",
                "netflix",
                "spotify",
                "hotstar",
                "movie",
            ],
            MerchantCategory.UTILITIES: [
                "electricity",
                "gas",
                "water",
                "mobile",
                "broadband",
                "bill",
            ],
            MerchantCategory.HEALTHCARE: [
                "pharmacy",
                "hospital",
                "clinic",
                "doctor",
                "medical",
            ],
            MerchantCategory.EDUCATION: [
                "school",
                "college",
                "university",
                "course",
                "education",
            ],
        }

        for category, keywords in category_keywords.items():
            if any(keyword in vpa_lower for keyword in keywords):
                return category

        return MerchantCategory.OTHER

    def _calculate_frequency_score(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate transaction frequency score"""
        if len(transactions) < 2:
            return 0.0

        # Analyze transaction intervals
        timestamps = []
        for tx in transactions:
            if tx.get("timestamp"):
                try:
                    timestamps.append(datetime.fromisoformat(tx["timestamp"]))
                except:
                    continue

        if len(timestamps) < 2:
            return 0.0

        timestamps.sort()
        intervals = [
            (timestamps[i + 1] - timestamps[i]).days for i in range(len(timestamps) - 1)
        ]

        if not intervals:
            return 0.0

        avg_interval = np.mean(intervals)

        # Score based on frequency (lower interval = higher frequency = higher score)
        if avg_interval <= 1:
            return 1.0
        elif avg_interval <= 7:
            return 0.8
        elif avg_interval <= 30:
            return 0.6
        elif avg_interval <= 90:
            return 0.4
        else:
            return 0.2

    def _calculate_loyalty_score(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate user loyalty score to merchant"""
        if not transactions:
            return 0.0

        # Time span analysis
        timestamps = []
        for tx in transactions:
            if tx.get("timestamp"):
                try:
                    timestamps.append(datetime.fromisoformat(tx["timestamp"]))
                except:
                    continue

        if len(timestamps) < 2:
            return 0.3  # Default for single transaction

        timestamps.sort()
        time_span = (timestamps[-1] - timestamps[0]).days

        # Loyalty based on transaction count and time span
        if time_span == 0:
            return 0.5

        transactions_per_month = (len(transactions) * 30) / max(time_span, 1)

        if transactions_per_month >= 4:
            return 1.0
        elif transactions_per_month >= 2:
            return 0.8
        elif transactions_per_month >= 1:
            return 0.6
        else:
            return 0.4

    def _identify_peak_hours(self, transactions: List[Dict[str, Any]]) -> List[int]:
        """Identify peak transaction hours"""
        hours = []
        for tx in transactions:
            if tx.get("timestamp"):
                try:
                    dt = datetime.fromisoformat(tx["timestamp"])
                    hours.append(dt.hour)
                except:
                    continue

        if not hours:
            return []

        # Count transactions per hour
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        # Find peak hours (above average)
        avg_count = np.mean(list(hour_counts.values()))
        peak_hours = [hour for hour, count in hour_counts.items() if count > avg_count]

        return sorted(peak_hours)

    def _analyze_seasonal_patterns(
        self, transactions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze seasonal transaction patterns"""
        monthly_counts = {}

        for tx in transactions:
            if tx.get("timestamp"):
                try:
                    dt = datetime.fromisoformat(tx["timestamp"])
                    month = dt.strftime("%B")
                    monthly_counts[month] = monthly_counts.get(month, 0) + 1
                except:
                    continue

        if not monthly_counts:
            return {}

        # Normalize to percentages
        total = sum(monthly_counts.values())
        return {month: count / total for month, count in monthly_counts.items()}

    def _calculate_dispute_rate(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate dispute rate"""
        if not transactions:
            return 0.0

        disputes = sum(1 for tx in transactions if tx.get("disputed", False))
        return disputes / len(transactions)

    def _calculate_refund_rate(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate refund rate"""
        if not transactions:
            return 0.0

        refunds = sum(1 for tx in transactions if tx.get("refunded", False))
        return refunds / len(transactions)

    def _analyze_user_sentiment(self, transactions: List[Dict[str, Any]]) -> float:
        """Analyze user sentiment towards merchant"""
        sentiment_sum = 0.0
        sentiment_count = 0

        for tx in transactions:
            # Check for sentiment indicators
            if tx.get("rating"):
                sentiment_sum += tx["rating"] / 5.0  # Normalize to 0-1
                sentiment_count += 1
            elif tx.get("disputed", False):
                sentiment_sum += 0.1  # Negative sentiment
                sentiment_count += 1
            elif tx.get("refunded", False):
                sentiment_sum += 0.2  # Negative sentiment
                sentiment_count += 1
            else:
                sentiment_sum += 0.7  # Neutral-positive sentiment
                sentiment_count += 1

        return sentiment_sum / max(sentiment_count, 1)

    def _calculate_growth_trend(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate merchant growth trend"""
        if len(transactions) < 4:
            return 0.5  # Neutral

        # Group by month and calculate trend
        monthly_volumes = {}
        for tx in transactions:
            if tx.get("timestamp") and tx.get("amount"):
                try:
                    dt = datetime.fromisoformat(tx["timestamp"])
                    month_key = dt.strftime("%Y-%m")
                    monthly_volumes[month_key] = (
                        monthly_volumes.get(month_key, 0) + tx["amount"]
                    )
                except:
                    continue

        if len(monthly_volumes) < 2:
            return 0.5

        # Calculate growth trend
        months = sorted(monthly_volumes.keys())
        volumes = [monthly_volumes[month] for month in months]

        if len(volumes) >= 2:
            growth = (volumes[-1] - volumes[0]) / max(volumes[0], 1)
            return min(1.0, max(0.0, 0.5 + growth))

        return 0.5

    def _assess_pricing_fairness(
        self, transactions: List[Dict[str, Any]], market_data: Optional[Dict[str, Any]]
    ) -> float:
        """Assess pricing fairness compared to market"""
        if not transactions:
            return 0.5

        avg_amount = np.mean([tx.get("amount", 0) for tx in transactions])

        # If no market data, use transaction variance as fairness indicator
        if not market_data:
            amounts = [tx.get("amount", 0) for tx in transactions]
            if len(amounts) > 1:
                cv = np.std(amounts) / max(np.mean(amounts), 1)
                return max(0.0, min(1.0, 1.0 - cv / 2))
            return 0.7

        # Compare with market benchmarks
        market_avg = market_data.get("average_transaction", avg_amount)
        if market_avg == 0:
            return 0.5

        ratio = avg_amount / market_avg
        if ratio <= 0.9:
            return 1.0  # Below market - good
        elif ratio <= 1.1:
            return 0.8  # At market - fair
        elif ratio <= 1.3:
            return 0.6  # Above market - concerns
        else:
            return 0.3  # Significantly above market - unfair

    def _assess_service_quality(self, transactions: List[Dict[str, Any]]) -> float:
        """Assess service quality based on transaction patterns"""
        if not transactions:
            return 0.5

        quality_score = 0.8  # Base score

        # Success rate
        success_count = sum(1 for tx in transactions if tx.get("status") == "success")
        success_rate = success_count / len(transactions)
        quality_score *= success_rate

        # Dispute rate penalty
        dispute_rate = self._calculate_dispute_rate(transactions)
        quality_score *= 1.0 - dispute_rate

        # Refund rate penalty
        refund_rate = self._calculate_refund_rate(transactions)
        quality_score *= 1.0 - refund_rate * 0.5

        return max(0.0, min(1.0, quality_score))

    def _assess_compliance(self, vpa: str, transactions: List[Dict[str, Any]]) -> float:
        """Assess regulatory compliance score"""
        compliance_score = 0.8  # Base compliance score

        # Check for compliance indicators
        large_transactions = sum(
            1 for tx in transactions if tx.get("amount", 0) > 50000
        )
        if large_transactions > 0:
            compliance_score += 0.1  # Properly handling large transactions

        # Regular transaction patterns indicate good business practices
        if len(transactions) >= 10:
            compliance_score += 0.1

        return min(1.0, compliance_score)

    def _extract_business_name(self, vpa: str) -> str:
        """Extract business name from VPA"""
        # Simple extraction - in real implementation, would use merchant database
        name_part = vpa.split("@")[0]
        return name_part.replace(".", " ").replace("_", " ").title()

    def _assess_risk_factors(
        self, profile: MerchantProfile, transactions: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify specific risk factors"""
        risk_factors = []

        if profile.trust_score < 0.5:
            risk_factors.append("Low trust score indicates reliability concerns")

        if profile.dispute_rate > 0.1:
            risk_factors.append("High dispute rate suggests service issues")

        if profile.refund_rate > 0.05:
            risk_factors.append("Elevated refund rate may indicate quality problems")

        if profile.pricing_fairness < 0.6:
            risk_factors.append("Pricing significantly above market average")

        if profile.compliance_score < 0.7:
            risk_factors.append("Potential regulatory compliance concerns")

        return risk_factors

    def _identify_opportunities(
        self, profile: MerchantProfile, market_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify opportunities for optimization"""
        opportunities = []

        if profile.loyalty_score > 0.8:
            opportunities.append("High loyalty - negotiate volume discounts")

        if profile.frequency_score > 0.7:
            opportunities.append("Frequent usage - explore loyalty programs")

        if profile.growth_trend > 0.6:
            opportunities.append(
                "Growing merchant - potential partnership opportunities"
            )

        if profile.service_quality > 0.8:
            opportunities.append(
                "High service quality - reliable for important transactions"
            )

        return opportunities

    def _generate_recommendations(
        self, profile: MerchantProfile, insights: List[MerchantInsight]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Risk-based recommendations
        if profile.risk_level == MerchantRiskLevel.HIGH:
            recommendations.append("Set transaction limits for this merchant")
            recommendations.append("Enable transaction notifications")

        # Loyalty-based recommendations
        if profile.loyalty_score > 0.8:
            recommendations.append("Explore exclusive offers or discounts")

        # Frequency-based recommendations
        if profile.frequency_score > 0.8:
            recommendations.append("Consider setting monthly spending caps")

        # Quality-based recommendations
        if profile.service_quality < 0.6:
            recommendations.append("Monitor service quality and consider alternatives")

        return recommendations

    async def _perform_competitive_analysis(
        self, profile: MerchantProfile
    ) -> Dict[str, Any]:
        """Perform competitive analysis"""
        return {
            "category_ranking": "Top 25%",  # Placeholder
            "price_competitiveness": profile.pricing_fairness,
            "service_comparison": (
                "Above Average" if profile.service_quality > 0.7 else "Below Average"
            ),
            "market_share_estimate": 0.15,  # Placeholder
            "alternatives_available": True,
        }

    def _analyze_pricing_patterns(
        self, profile: MerchantProfile, transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze pricing patterns and trends"""
        amounts = [tx.get("amount", 0) for tx in transactions if tx.get("amount")]

        if not amounts:
            return {}

        return {
            "average_transaction": np.mean(amounts),
            "median_transaction": np.median(amounts),
            "price_volatility": (
                np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 0
            ),
            "price_trend": "stable",  # Simplified
            "discount_frequency": 0.1,  # Placeholder
            "peak_pricing_hours": profile.peak_hours,
        }

    def _analyze_behavioral_patterns(
        self, transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze user behavioral patterns with merchant"""
        return {
            "transaction_timing": "Regular business hours",
            "spending_consistency": "Consistent",
            "payment_method_preference": "UPI",
            "transaction_size_pattern": "Consistent medium amounts",
            "frequency_pattern": "Weekly regular",
        }
