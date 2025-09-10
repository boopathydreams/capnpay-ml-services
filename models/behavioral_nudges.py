"""
Behavioral Nudges Engine for Cap'n Pay

Uses behavioral psychology principles to create intelligent spending nudges:
1. Loss Aversion - Frame spending as losses vs. gains
2. Social Proof - Compare with similar users
3. Mental Accounting - Category-specific budgeting psychology
4. Commitment Devices - Help users commit to financial goals
5. Temporal Discounting - Make future consequences feel immediate
6. Anchoring - Use reference points to influence decisions

Patent-worthy innovation: Real-time psychology-driven financial nudges
"""

import json
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)
import logging

logger = logging.getLogger(__name__)


class NudgeType(Enum):
    LOSS_AVERSION = "loss_aversion"
    SOCIAL_PROOF = "social_proof"
    MENTAL_ACCOUNTING = "mental_accounting"
    COMMITMENT_DEVICE = "commitment_device"
    TEMPORAL_DISCOUNTING = "temporal_discounting"
    ANCHORING = "anchoring"


class NudgeIntensity(Enum):
    GENTLE = "gentle"
    MODERATE = "moderate"
    STRONG = "strong"


@dataclass
class NudgeResult:
    """A behavioral nudge recommendation"""

    nudge_type: NudgeType
    intensity: NudgeIntensity
    message: str
    psychological_principle: str
    confidence: float
    recommended_action: str
    context_data: Dict[str, Any]


@dataclass
class UserPsychProfile:
    """User's psychological spending profile"""

    loss_aversion_sensitivity: float  # 0-1, higher = more sensitive to losses
    social_influence_susceptibility: float  # 0-1, higher = more influenced by peers
    commitment_tendency: float  # 0-1, higher = more likely to stick to goals
    temporal_preference: float  # 0-1, higher = more present-focused
    anchoring_susceptibility: float  # 0-1, higher = more influenced by reference points
    risk_tolerance: float  # 0-1, higher = more risk-tolerant
    spending_impulsiveness: float  # 0-1, higher = more impulsive


class BehavioralNudgesEngine:
    """
    Advanced behavioral psychology engine for financial decision nudges

    Research-backed psychology principles:
    - Kahneman & Tversky's Prospect Theory
    - Thaler's Mental Accounting
    - Cialdini's Social Proof
    - Ariely's Predictably Irrational behaviors
    """

    def __init__(self):
        self.nudge_templates = self._load_nudge_templates()
        self.social_benchmarks = {}
        self.commitment_contracts = {}

    def _load_nudge_templates(self) -> Dict[str, Dict]:
        """Load psychology-based nudge message templates"""
        return {
            "loss_aversion": {
                "gentle": [
                    "You could save ₹{amount} this month by skipping this purchase 💰",
                    "This purchase means ₹{amount} less towards your {goal} goal 🎯",
                ],
                "moderate": [
                    "⚠️ This ₹{amount} purchase will put you ₹{overspend} over your {category} budget",
                    "You're about to lose ₹{amount} from your savings progress 📉",
                ],
                "strong": [
                    "🚨 STOP: This purchase will cost you ₹{total_impact} in missed opportunities!",
                    "⛔ You'll lose {days_delay} days of progress toward your {goal} goal!",
                ],
            },
            "social_proof": {
                "gentle": [
                    "83% of users like you skip purchases like this 👥",
                    "Similar users spend 25% less on {category} this month 📊",
                ],
                "moderate": [
                    "Users in your income group spend ₹{benchmark} less on {category} 👥",
                    "You're spending 40% more than similar users this month 📈",
                ],
                "strong": [
                    "🏆 Top savers in your group avoid purchases like this 95% of the time!",
                    "⚡ You're in the top 10% of spenders - join the savers instead!",
                ],
            },
            "mental_accounting": {
                "gentle": [
                    "Consider moving ₹{amount} from your {from_bucket} to {to_bucket} budget 🗂️",
                    "This fits better in your {suggested_category} mental budget 🧠",
                ],
                "moderate": [
                    "⚖️ This purchase unbalances your {category} vs {other_category} spending",
                    "You've already allocated ₹{allocated} for {category} this month",
                ],
                "strong": [
                    "🚨 This violates your {category} budget rules by ₹{violation}!",
                    "⛔ Emergency: This breaks your carefully planned budget structure!",
                ],
            },
            "commitment_device": {
                "gentle": [
                    "Remember your commitment to save ₹{goal_amount} by {date} 🤝",
                    "This purchase delays your {goal} goal by {delay} days ⏰",
                ],
                "moderate": [
                    "⚠️ This conflicts with your promise to limit {category} spending",
                    "You committed to staying under ₹{limit} for {category} 📝",
                ],
                "strong": [
                    "🚨 COMMITMENT VIOLATION: You promised to avoid purchases like this!",
                    "⛔ Breaking this commitment costs ₹{penalty} from your goal fund!",
                ],
            },
            "temporal_discounting": {
                "gentle": [
                    "In 6 months, you'll have ₹{future_value} instead of this purchase ⏰",
                    "This ₹{amount} could become ₹{investment_value} by next year 📈",
                ],
                "moderate": [
                    "⏰ Waiting 24 hours could save you from this impulse purchase",
                    "By year-end, this money could grow to ₹{year_end_value} 💰",
                ],
                "strong": [
                    "🚨 This ₹{amount} costs you ₹{opportunity_cost} in future wealth!",
                    "⛔ 10-year impact: You're sacrificing ₹{decade_cost} for this!",
                ],
            },
            "anchoring": {
                "gentle": [
                    "Compared to your usual ₹{anchor_amount} {category} spending... 🎯",
                    "This is {percentage}% higher than your typical {category} purchase 📊",
                ],
                "moderate": [
                    "⚠️ This ₹{amount} is {multiplier}x your average {category} purchase",
                    "Your friends typically spend ₹{friend_anchor} for similar items 👥",
                ],
                "strong": [
                    "🚨 This is {extreme_multiplier}x higher than any {category} purchase you've made!",
                    "⛔ Even luxury buyers spend only ₹{luxury_anchor} for this category!",
                ],
            },
        }

    def analyze_user_psychology(
        self, user_id: str, transaction_history: List[Dict]
    ) -> UserPsychProfile:
        """
        Analyze user's psychological spending profile from transaction history

        Uses behavioral patterns to infer psychological traits:
        - Response to past nudges (loss aversion sensitivity)
        - Spending consistency (commitment tendency)
        - Purchase timing patterns (temporal preference)
        - Category switching (mental accounting)
        """

        # Initialize with neutral baseline
        profile = UserPsychProfile(
            loss_aversion_sensitivity=0.5,
            social_influence_susceptibility=0.5,
            commitment_tendency=0.5,
            temporal_preference=0.5,
            anchoring_susceptibility=0.5,
            risk_tolerance=0.5,
            spending_impulsiveness=0.5,
        )

        if not transaction_history:
            return profile

        # Analyze loss aversion from refund/return patterns
        returns = [t for t in transaction_history if t.get("type") == "refund"]
        if len(transaction_history) > 10:
            return_rate = len(returns) / len(transaction_history)
            profile.loss_aversion_sensitivity = min(
                1.0, return_rate * 3
            )  # Higher returns = higher loss aversion

        # Analyze commitment tendency from budget adherence
        monthly_spending = self._group_by_month(transaction_history)
        consistency_scores = []
        for month_data in monthly_spending.values():
            if len(month_data) > 1:
                amounts = [t["amount"] for t in month_data]
                cv = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 1
                consistency_scores.append(
                    1 - min(1, cv)
                )  # Lower variance = higher consistency

        if consistency_scores:
            profile.commitment_tendency = np.mean(consistency_scores)

        # Analyze temporal preference from purchase timing
        weekend_purchases = len(
            [
                t
                for t in transaction_history
                if datetime.fromisoformat(t.get("timestamp", "2024-01-01")).weekday()
                >= 5
            ]
        )
        if len(transaction_history) > 0:
            weekend_ratio = weekend_purchases / len(transaction_history)
            profile.temporal_preference = (
                weekend_ratio  # More weekend purchases = more present-focused
            )

        # Analyze impulsiveness from purchase frequency spikes
        daily_spending = self._group_by_day(transaction_history)
        daily_counts = [len(day_data) for day_data in daily_spending.values()]
        if daily_counts:
            spike_threshold = np.mean(daily_counts) + 2 * np.std(daily_counts)
            spike_days = len(
                [count for count in daily_counts if count > spike_threshold]
            )
            profile.spending_impulsiveness = min(
                1.0, spike_days / len(daily_counts) * 5
            )

        # Analyze social influence from category diversification
        categories = set(t.get("category", "other") for t in transaction_history)
        if len(transaction_history) > 20:
            diversity_score = len(categories) / min(len(transaction_history), 10)
            profile.social_influence_susceptibility = min(
                1.0, diversity_score
            )  # More categories = more social influence

        # Analyze anchoring from price point consistency within categories
        category_prices = {}
        for t in transaction_history:
            cat = t.get("category", "other")
            if cat not in category_prices:
                category_prices[cat] = []
            category_prices[cat].append(t["amount"])

        anchoring_scores = []
        for cat, prices in category_prices.items():
            if len(prices) > 3:
                cv = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
                anchoring_scores.append(cv)  # Higher variance = less anchoring effect

        if anchoring_scores:
            profile.anchoring_susceptibility = 1 - min(1.0, np.mean(anchoring_scores))

        logger.info(f"Generated psychological profile for user {user_id}: {profile}")
        return profile

    def generate_nudge(
        self,
        user_id: str,
        proposed_transaction: Dict[str, Any],
        user_context: Dict[str, Any],
        psych_profile: Optional[UserPsychProfile] = None,
    ) -> List[NudgeResult]:
        """
        Generate personalized behavioral nudges for a proposed transaction

        Args:
            user_id: User identifier
            proposed_transaction: Transaction details (amount, category, merchant, etc.)
            user_context: User's financial context (budget, goals, history, etc.)
            psych_profile: User's psychological profile (optional, will analyze if not provided)

        Returns:
            List of personalized nudge recommendations ranked by effectiveness
        """

        if not psych_profile:
            transaction_history = user_context.get("transaction_history", [])
            psych_profile = self.analyze_user_psychology(user_id, transaction_history)

        nudges = []
        amount = proposed_transaction.get("amount", 0)
        category = proposed_transaction.get("category", "other")
        merchant = proposed_transaction.get("merchant", "Unknown")

        # Generate Loss Aversion nudges
        if psych_profile.loss_aversion_sensitivity > 0.3:
            loss_nudges = self._generate_loss_aversion_nudges(
                amount, category, user_context, psych_profile
            )
            nudges.extend(loss_nudges)

        # Generate Social Proof nudges
        if psych_profile.social_influence_susceptibility > 0.3:
            social_nudges = self._generate_social_proof_nudges(
                amount, category, user_context, psych_profile
            )
            nudges.extend(social_nudges)

        # Generate Mental Accounting nudges
        mental_nudges = self._generate_mental_accounting_nudges(
            amount, category, user_context, psych_profile
        )
        nudges.extend(mental_nudges)

        # Generate Commitment Device nudges
        if psych_profile.commitment_tendency > 0.2:
            commitment_nudges = self._generate_commitment_nudges(
                amount, category, user_context, psych_profile
            )
            nudges.extend(commitment_nudges)

        # Generate Temporal Discounting nudges
        if psych_profile.temporal_preference > 0.4:  # Present-focused users
            temporal_nudges = self._generate_temporal_discounting_nudges(
                amount, category, user_context, psych_profile
            )
            nudges.extend(temporal_nudges)

        # Generate Anchoring nudges
        if psych_profile.anchoring_susceptibility > 0.3:
            anchoring_nudges = self._generate_anchoring_nudges(
                amount, category, user_context, psych_profile
            )
            nudges.extend(anchoring_nudges)

        # Rank nudges by predicted effectiveness
        ranked_nudges = self._rank_nudges_by_effectiveness(
            nudges, psych_profile, user_context
        )

        logger.info(
            f"Generated {len(ranked_nudges)} nudges for user {user_id}, transaction ₹{amount}"
        )
        return ranked_nudges[:3]  # Return top 3 most effective nudges

    def _generate_loss_aversion_nudges(
        self,
        amount: float,
        category: str,
        user_context: Dict,
        psych_profile: UserPsychProfile,
    ) -> List[NudgeResult]:
        """Generate nudges based on loss aversion psychology"""
        nudges = []

        # Calculate potential losses
        budget = user_context.get("budgets", {}).get(category, 0)
        spent_this_month = user_context.get("monthly_spending", {}).get(category, 0)
        savings_goal = user_context.get("savings_goal", {})

        # Determine intensity based on budget overspend
        overspend = max(0, spent_this_month + amount - budget) if budget > 0 else 0

        if overspend > budget * 0.5:  # >50% overspend
            intensity = NudgeIntensity.STRONG
        elif overspend > budget * 0.2:  # >20% overspend
            intensity = NudgeIntensity.MODERATE
        else:
            intensity = NudgeIntensity.GENTLE

        # Select appropriate message template
        templates = self.nudge_templates["loss_aversion"][intensity.value]

        if savings_goal:
            goal_name = savings_goal.get("name", "savings")
            goal_amount = savings_goal.get("target_amount", 0)
            days_delay = int((amount / goal_amount) * 365) if goal_amount > 0 else 0

            message = templates[0].format(
                amount=amount,
                goal=goal_name,
                overspend=overspend,
                category=category,
                total_impact=amount * 2,  # Psychological amplification
                days_delay=days_delay,
            )
        else:
            message = f"This ₹{amount} purchase reduces your available savings"

        confidence = psych_profile.loss_aversion_sensitivity * 0.9

        nudges.append(
            NudgeResult(
                nudge_type=NudgeType.LOSS_AVERSION,
                intensity=intensity,
                message=message,
                psychological_principle="Loss Aversion (Kahneman & Tversky): Losses feel 2x stronger than equivalent gains",
                confidence=confidence,
                recommended_action=(
                    "Delay purchase for 24 hours"
                    if intensity != NudgeIntensity.STRONG
                    else "Skip this purchase"
                ),
                context_data={
                    "overspend_amount": overspend,
                    "budget_utilization": (
                        (spent_this_month + amount) / budget if budget > 0 else 0
                    ),
                    "savings_impact": (
                        amount / savings_goal.get("target_amount", 1)
                        if savings_goal
                        else 0
                    ),
                },
            )
        )

        return nudges

    def _generate_social_proof_nudges(
        self,
        amount: float,
        category: str,
        user_context: Dict,
        psych_profile: UserPsychProfile,
    ) -> List[NudgeResult]:
        """Generate nudges based on social proof psychology"""
        nudges = []

        # Get social benchmarks (would come from aggregated user data)
        peer_avg = self.social_benchmarks.get(category, {}).get(
            "peer_average", amount * 0.8
        )
        top_savers_avg = self.social_benchmarks.get(category, {}).get(
            "top_10_percent", amount * 0.5
        )

        # Determine intensity based on deviation from social norms
        deviation = (amount - peer_avg) / peer_avg if peer_avg > 0 else 0

        if deviation > 1.0:  # >100% above peer average
            intensity = NudgeIntensity.STRONG
        elif deviation > 0.5:  # >50% above peer average
            intensity = NudgeIntensity.MODERATE
        else:
            intensity = NudgeIntensity.GENTLE

        templates = self.nudge_templates["social_proof"][intensity.value]

        if deviation > 0:
            message = templates[1].format(
                benchmark=peer_avg, category=category, percentage=int(deviation * 100)
            )
        else:
            message = templates[0].format(category=category)

        confidence = psych_profile.social_influence_susceptibility * 0.85

        nudges.append(
            NudgeResult(
                nudge_type=NudgeType.SOCIAL_PROOF,
                intensity=intensity,
                message=message,
                psychological_principle="Social Proof (Cialdini): People follow others' behavior as evidence of correct action",
                confidence=confidence,
                recommended_action="Compare with similar users before purchasing",
                context_data={
                    "peer_average": peer_avg,
                    "user_vs_peer_deviation": deviation,
                    "top_savers_benchmark": top_savers_avg,
                    "social_ranking": (
                        "above_average" if deviation > 0 else "below_average"
                    ),
                },
            )
        )

        return nudges

    def _generate_mental_accounting_nudges(
        self,
        amount: float,
        category: str,
        user_context: Dict,
        psych_profile: UserPsychProfile,
    ) -> List[NudgeResult]:
        """Generate nudges based on mental accounting psychology"""
        nudges = []

        budgets = user_context.get("budgets", {})
        monthly_spending = user_context.get("monthly_spending", {})

        # Check for budget violations
        current_spent = monthly_spending.get(category, 0)
        budget_limit = budgets.get(category, 0)

        if budget_limit > 0:
            utilization = (current_spent + amount) / budget_limit

            if utilization > 1.2:  # >20% over budget
                intensity = NudgeIntensity.STRONG
            elif utilization > 1.0:  # Over budget
                intensity = NudgeIntensity.MODERATE
            else:
                intensity = NudgeIntensity.GENTLE

            templates = self.nudge_templates["mental_accounting"][intensity.value]

            violation = max(0, current_spent + amount - budget_limit)
            message = templates[0].format(
                amount=amount,
                category=category,
                other_category="savings",
                allocated=budget_limit,
                violation=violation,
            )

            confidence = 0.8  # Mental accounting is universal

            nudges.append(
                NudgeResult(
                    nudge_type=NudgeType.MENTAL_ACCOUNTING,
                    intensity=intensity,
                    message=message,
                    psychological_principle="Mental Accounting (Thaler): People categorize money into separate mental buckets",
                    confidence=confidence,
                    recommended_action=(
                        "Review budget allocation"
                        if intensity == NudgeIntensity.GENTLE
                        else "Reallocate from another category"
                    ),
                    context_data={
                        "budget_utilization": utilization,
                        "budget_violation": violation,
                        "category_budget": budget_limit,
                        "month_to_date_spent": current_spent,
                    },
                )
            )

        return nudges

    def _generate_commitment_nudges(
        self,
        amount: float,
        category: str,
        user_context: Dict,
        psych_profile: UserPsychProfile,
    ) -> List[NudgeResult]:
        """Generate nudges based on commitment device psychology"""
        nudges = []

        # Check for active commitments/goals
        goals = user_context.get("financial_goals", [])
        commitments = user_context.get("spending_commitments", [])

        for goal in goals:
            if goal.get("category") == category or goal.get("type") == "savings":
                target_date = goal.get("target_date")
                target_amount = goal.get("target_amount", 0)
                current_progress = goal.get("current_amount", 0)

                # Calculate delay caused by this purchase
                remaining = target_amount - current_progress
                delay_days = int((amount / remaining) * 30) if remaining > 0 else 0

                intensity = (
                    NudgeIntensity.STRONG if delay_days > 7 else NudgeIntensity.MODERATE
                )

                templates = self.nudge_templates["commitment_device"][intensity.value]
                message = templates[0].format(
                    goal_amount=target_amount,
                    date=target_date,
                    goal=goal.get("name", "financial goal"),
                    delay=delay_days,
                    category=category,
                    limit=target_amount,
                    amount=amount,
                    penalty=amount * 0.1,  # 10% penalty for psychological effect
                )

                confidence = psych_profile.commitment_tendency * 0.9

                nudges.append(
                    NudgeResult(
                        nudge_type=NudgeType.COMMITMENT_DEVICE,
                        intensity=intensity,
                        message=message,
                        psychological_principle="Commitment Device (Ariely): Public commitments increase follow-through",
                        confidence=confidence,
                        recommended_action="Review your commitment before proceeding",
                        context_data={
                            "goal_name": goal.get("name"),
                            "progress_percentage": (
                                (current_progress / target_amount) * 100
                                if target_amount > 0
                                else 0
                            ),
                            "delay_days": delay_days,
                            "commitment_strength": psych_profile.commitment_tendency,
                        },
                    )
                )

        return nudges

    def _generate_temporal_discounting_nudges(
        self,
        amount: float,
        category: str,
        user_context: Dict,
        psych_profile: UserPsychProfile,
    ) -> List[NudgeResult]:
        """Generate nudges based on temporal discounting psychology"""
        nudges = []

        # Calculate future value of money (assuming 12% investment return)
        annual_return = 0.12
        future_6m = amount * (1 + annual_return / 2)
        future_1y = amount * (1 + annual_return)
        future_10y = amount * ((1 + annual_return) ** 10)

        # Determine intensity based on user's temporal preference
        if psych_profile.temporal_preference > 0.7:  # Very present-focused
            intensity = NudgeIntensity.STRONG
        elif psych_profile.temporal_preference > 0.5:
            intensity = NudgeIntensity.MODERATE
        else:
            intensity = NudgeIntensity.GENTLE

        templates = self.nudge_templates["temporal_discounting"][intensity.value]

        if intensity == NudgeIntensity.STRONG:
            message = templates[0].format(
                amount=amount,
                opportunity_cost=future_1y - amount,
                decade_cost=future_10y - amount,
            )
        else:
            message = templates[0].format(
                future_value=future_6m,
                amount=amount,
                investment_value=future_1y,
                year_end_value=future_1y,
            )

        confidence = (
            1 - psych_profile.temporal_preference
        ) * 0.8  # More effective for future-focused users

        nudges.append(
            NudgeResult(
                nudge_type=NudgeType.TEMPORAL_DISCOUNTING,
                intensity=intensity,
                message=message,
                psychological_principle="Temporal Discounting: People undervalue future rewards vs immediate gratification",
                confidence=confidence,
                recommended_action="Wait 24 hours to reduce temporal bias",
                context_data={
                    "future_value_6m": future_6m,
                    "future_value_1y": future_1y,
                    "future_value_10y": future_10y,
                    "opportunity_cost": future_1y - amount,
                    "temporal_preference": psych_profile.temporal_preference,
                },
            )
        )

        return nudges

    def _generate_anchoring_nudges(
        self,
        amount: float,
        category: str,
        user_context: Dict,
        psych_profile: UserPsychProfile,
    ) -> List[NudgeResult]:
        """Generate nudges based on anchoring bias psychology"""
        nudges = []

        # Calculate anchoring references
        transaction_history = user_context.get("transaction_history", [])
        category_transactions = [
            t for t in transaction_history if t.get("category") == category
        ]

        if category_transactions:
            category_amounts = [t["amount"] for t in category_transactions]
            avg_amount = np.mean(category_amounts)
            typical_amount = np.median(category_amounts)
            max_amount = max(category_amounts)

            # Calculate deviations
            avg_deviation = (amount - avg_amount) / avg_amount if avg_amount > 0 else 0
            typical_deviation = (
                (amount - typical_amount) / typical_amount if typical_amount > 0 else 0
            )

            # Determine intensity based on deviation from anchors
            if avg_deviation > 2.0:  # >200% of average
                intensity = NudgeIntensity.STRONG
            elif avg_deviation > 0.5:  # >50% of average
                intensity = NudgeIntensity.MODERATE
            else:
                intensity = NudgeIntensity.GENTLE

            templates = self.nudge_templates["anchoring"][intensity.value]

            if intensity == NudgeIntensity.STRONG:
                message = templates[0].format(
                    extreme_multiplier=round(amount / avg_amount, 1),
                    category=category,
                    luxury_anchor=max_amount,
                )
            else:
                message = templates[0].format(
                    anchor_amount=typical_amount,
                    category=category,
                    percentage=int(abs(typical_deviation) * 100),
                    amount=amount,
                    multiplier=round(amount / avg_amount, 1) if avg_amount > 0 else 1,
                )

            confidence = psych_profile.anchoring_susceptibility * 0.7

            nudges.append(
                NudgeResult(
                    nudge_type=NudgeType.ANCHORING,
                    intensity=intensity,
                    message=message,
                    psychological_principle="Anchoring Bias (Tversky & Kahneman): First number influences all subsequent judgments",
                    confidence=confidence,
                    recommended_action="Compare with your typical spending pattern",
                    context_data={
                        "user_average": avg_amount,
                        "user_typical": typical_amount,
                        "user_maximum": max_amount,
                        "deviation_from_average": avg_deviation,
                        "deviation_from_typical": typical_deviation,
                        "anchoring_strength": psych_profile.anchoring_susceptibility,
                    },
                )
            )

        return nudges

    def _rank_nudges_by_effectiveness(
        self,
        nudges: List[NudgeResult],
        psych_profile: UserPsychProfile,
        user_context: Dict,
    ) -> List[NudgeResult]:
        """Rank nudges by predicted effectiveness for the specific user"""

        # Define effectiveness weights based on research
        effectiveness_weights = {
            NudgeType.LOSS_AVERSION: 0.9,  # Highest effectiveness
            NudgeType.SOCIAL_PROOF: 0.8,
            NudgeType.MENTAL_ACCOUNTING: 0.85,
            NudgeType.COMMITMENT_DEVICE: 0.75,
            NudgeType.TEMPORAL_DISCOUNTING: 0.7,
            NudgeType.ANCHORING: 0.65,
        }

        # Calculate effectiveness scores
        for nudge in nudges:
            base_effectiveness = effectiveness_weights.get(nudge.nudge_type, 0.5)

            # Adjust for user's psychological profile
            profile_multiplier = 1.0
            if nudge.nudge_type == NudgeType.LOSS_AVERSION:
                profile_multiplier = psych_profile.loss_aversion_sensitivity
            elif nudge.nudge_type == NudgeType.SOCIAL_PROOF:
                profile_multiplier = psych_profile.social_influence_susceptibility
            elif nudge.nudge_type == NudgeType.COMMITMENT_DEVICE:
                profile_multiplier = psych_profile.commitment_tendency
            elif nudge.nudge_type == NudgeType.TEMPORAL_DISCOUNTING:
                profile_multiplier = (
                    1 - psych_profile.temporal_preference
                )  # Inverse relationship
            elif nudge.nudge_type == NudgeType.ANCHORING:
                profile_multiplier = psych_profile.anchoring_susceptibility

            # Adjust for intensity
            intensity_multiplier = {
                NudgeIntensity.GENTLE: 0.8,
                NudgeIntensity.MODERATE: 1.0,
                NudgeIntensity.STRONG: 1.2,
            }[nudge.intensity]

            # Calculate final effectiveness score
            effectiveness_score = (
                base_effectiveness
                * profile_multiplier
                * intensity_multiplier
                * nudge.confidence
            )

            # Store in context data for debugging
            nudge.context_data["effectiveness_score"] = effectiveness_score

        # Sort by effectiveness score (descending)
        return sorted(
            nudges,
            key=lambda n: n.context_data.get("effectiveness_score", 0),
            reverse=True,
        )

    def _group_by_month(self, transactions: List[Dict]) -> Dict[str, List[Dict]]:
        """Group transactions by month"""
        monthly = {}
        for t in transactions:
            try:
                date = datetime.fromisoformat(t.get("timestamp", "2024-01-01"))
                month_key = f"{date.year}-{date.month:02d}"
                if month_key not in monthly:
                    monthly[month_key] = []
                monthly[month_key].append(t)
            except:
                continue
        return monthly

    def _group_by_day(self, transactions: List[Dict]) -> Dict[str, List[Dict]]:
        """Group transactions by day"""
        daily = {}
        for t in transactions:
            try:
                date = datetime.fromisoformat(t.get("timestamp", "2024-01-01"))
                day_key = f"{date.year}-{date.month:02d}-{date.day:02d}"
                if day_key not in daily:
                    daily[day_key] = []
                daily[day_key].append(t)
            except:
                continue
        return daily

    def track_nudge_effectiveness(
        self,
        user_id: str,
        nudge_result: NudgeResult,
        user_action: str,
        outcome: Dict[str, Any],
    ):
        """
        Track nudge effectiveness for continuous learning

        Args:
            user_id: User identifier
            nudge_result: The nudge that was shown
            user_action: What the user did (purchased, delayed, skipped, etc.)
            outcome: Financial outcome data
        """
        effectiveness_data = {
            "user_id": user_id,
            "nudge_type": nudge_result.nudge_type.value,
            "intensity": nudge_result.intensity.value,
            "predicted_confidence": nudge_result.confidence,
            "user_action": user_action,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
            "psychological_principle": nudge_result.psychological_principle,
        }

        # Store for ML model training
        # In production, this would go to a database or ML pipeline
        logger.info(f"Nudge effectiveness tracked: {effectiveness_data}")

        return effectiveness_data

    def get_nudge_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get analytics on nudge effectiveness for a user"""
        # This would query actual effectiveness data in production
        return {
            "user_id": user_id,
            "period_days": days,
            "total_nudges_shown": 45,
            "nudges_followed": 32,
            "effectiveness_rate": 0.71,
            "money_saved": 8750.50,
            "behavioral_score_improvement": 0.15,
            "most_effective_nudge_type": NudgeType.LOSS_AVERSION.value,
            "psychological_profile_evolution": {
                "loss_aversion_sensitivity": "+0.1",
                "commitment_tendency": "+0.2",
                "spending_impulsiveness": "-0.15",
            },
        }
