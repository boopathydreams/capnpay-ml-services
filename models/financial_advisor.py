"""
GPT-4 Financial Advisor for Cap'n Pay
Intelligent financial advisory system using OpenAI GPT-4
Provides personalized financial advice, goal setting, and insights
"""

import openai
from openai import AsyncOpenAI
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, asdict
import pandas as pd

from core.feature_store import feature_store
from core.model_registry import model_registry

logger = logging.getLogger(__name__)


@dataclass
class UserFinancialProfile:
    """Comprehensive user financial profile"""

    user_id: str
    monthly_income: float
    monthly_expenses: float
    savings_rate: float
    debt_amount: float
    investment_amount: float
    financial_goals: List[Dict[str, Any]]
    risk_tolerance: str  # conservative, moderate, aggressive
    age: int
    dependents: int
    employment_status: str
    spending_categories: Dict[str, float]


@dataclass
class FinancialAdviceRequest:
    """Request for financial advice"""

    user_id: str
    query: str
    context_type: (
        str  # spending_review, goal_setting, investment_advice, debt_management
    )
    transaction_data: Optional[List[Dict[str, Any]]] = None
    time_horizon: Optional[str] = None  # short, medium, long


@dataclass
class FinancialAdviceResponse:
    """Response with financial advice"""

    advice: str
    confidence_score: float
    advice_category: str
    actionable_steps: List[str]
    relevant_insights: List[str]
    follow_up_questions: List[str]
    risk_warnings: List[str]
    supporting_data: Dict[str, Any]


class GPT4FinancialAdvisor:
    """
    GPT-4 powered financial advisor with Indian fintech context
    Provides personalized advice, goal setting, and financial planning
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = AsyncOpenAI(
            api_key=api_key or "your-openai-api-key-here"  # Should be in environment
        )

        # Indian financial context prompts
        self.system_prompts = {
            "base": """You are a highly knowledgeable Indian financial advisor with expertise in:
- Indian banking and investment products (SIP, PPF, ELSS, Fixed Deposits)
- Indian tax laws and Section 80C benefits
- UPI payments and digital finance in India
- Indian mutual funds, stock market (NSE/BSE)
- Personal finance for middle-class Indian families
- Currency in Indian Rupees (₹)

Provide practical, actionable financial advice tailored to Indian context.
Be empathetic, clear, and avoid overly technical jargon.
Always consider the user's risk tolerance and financial goals.""",
            "spending_review": """Analyze the user's spending patterns and provide insights on:
- Budget optimization for Indian lifestyle
- Category-wise spending recommendations
- Suggestions for reducing unnecessary expenses
- Tips for better money management""",
            "goal_setting": """Help users set and achieve financial goals:
- Emergency fund recommendations (6-12 months expenses)
- Investment planning for goals (home, education, retirement)
- SIP calculations and mutual fund suggestions
- Tax-saving instrument recommendations""",
            "investment_advice": """Provide investment guidance:
- Asset allocation based on age and risk tolerance
- Mutual fund recommendations (Large cap, Mid cap, ELSS)
- SIP vs Lump sum investment strategies
- Market timing and long-term investment principles""",
            "debt_management": """Help with debt management:
- Personal loan vs credit card debt prioritization
- EMI optimization strategies
- Debt consolidation options
- Building credit score in India""",
        }

        # GPT-4 model configuration
        self.model_config = {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.3,  # Lower for more consistent advice
            "max_tokens": 1000,
            "top_p": 0.9,
        }

        # Advice templates and examples
        self.advice_templates = {
            "spending_alert": "Based on your ₹{amount} spending on {category}, here's my analysis...",
            "goal_progress": "You're {progress}% towards your {goal_name} goal of ₹{target_amount}...",
            "investment_suggestion": "Consider investing ₹{amount} in {instrument} to achieve {goal}...",
            "budget_optimization": "Your {category} spending is {percentage}% above recommended limits...",
        }

        logger.info("GPT-4 Financial Advisor initialized")

    async def get_financial_advice(
        self, request: FinancialAdviceRequest
    ) -> FinancialAdviceResponse:
        """
        Get personalized financial advice using GPT-4

        Args:
            request: Financial advice request with user context

        Returns:
            Structured financial advice response
        """
        try:
            # Get user context from feature store
            user_context = await self._build_user_context(request.user_id)

            # Build GPT-4 prompt
            prompt = await self._build_advice_prompt(request, user_context)

            # Get GPT-4 response
            gpt_response = await self._query_gpt4(prompt, request.context_type)

            # Parse and structure response
            structured_advice = await self._parse_advice_response(gpt_response, request)

            # Store advice for learning
            await self._store_advice_interaction(request, structured_advice)

            return structured_advice

        except Exception as e:
            logger.error(f"Error getting financial advice: {e}")
            return await self._fallback_advice(request)

    async def _build_user_context(self, user_id: str) -> Dict[str, Any]:
        """Build comprehensive user context from various data sources"""
        try:
            # Get user features from feature store
            payment_features = await asyncio.to_thread(
                feature_store.get_features, user_id, "payment_features"
            )
            behavioral_features = await asyncio.to_thread(
                feature_store.get_features, user_id, "behavioral_features"
            )

            # Mock user profile (in production, would come from user service)
            user_profile = {
                "monthly_income": 75000,  # ₹75k typical IT professional
                "age": 28,
                "employment_status": "employed",
                "dependents": 1,
                "risk_tolerance": "moderate",
                "financial_goals": [
                    {
                        "name": "Emergency Fund",
                        "target": 450000,
                        "current": 120000,
                        "priority": "high",
                    },
                    {
                        "name": "Home Purchase",
                        "target": 2500000,
                        "current": 350000,
                        "priority": "medium",
                    },
                    {
                        "name": "Retirement",
                        "target": 10000000,
                        "current": 85000,
                        "priority": "low",
                    },
                ],
            }

            # Calculate spending patterns
            spending_summary = {
                "monthly_expenses": (
                    payment_features.get("user_total_spent", 45000)
                    if payment_features
                    else 45000
                ),
                "category_breakdown": {
                    "food": 12000,
                    "transport": 8000,
                    "shopping": 15000,
                    "utilities": 5000,
                    "entertainment": 5000,
                },
                "savings_rate": 0.35,  # 35% savings rate
                "transaction_count": (
                    payment_features.get("user_transaction_count", 25)
                    if payment_features
                    else 25
                ),
            }

            return {
                "user_profile": user_profile,
                "spending_summary": spending_summary,
                "payment_features": payment_features or {},
                "behavioral_features": behavioral_features or {},
            }

        except Exception as e:
            logger.error(f"Error building user context: {e}")
            return {"error": str(e)}

    async def _build_advice_prompt(
        self, request: FinancialAdviceRequest, context: Dict[str, Any]
    ) -> str:
        """Build comprehensive prompt for GPT-4"""

        system_prompt = (
            self.system_prompts["base"]
            + "\n\n"
            + self.system_prompts.get(request.context_type, "")
        )

        user_context_str = f"""
USER FINANCIAL PROFILE:
- Monthly Income: ₹{context['user_profile']['monthly_income']:,.0f}
- Monthly Expenses: ₹{context['spending_summary']['monthly_expenses']:,.0f}
- Savings Rate: {context['spending_summary']['savings_rate']:.1%}
- Age: {context['user_profile']['age']} years
- Dependents: {context['user_profile']['dependents']}
- Risk Tolerance: {context['user_profile']['risk_tolerance']}

CURRENT FINANCIAL GOALS:
"""

        for goal in context["user_profile"]["financial_goals"]:
            progress = goal["current"] / goal["target"] * 100
            user_context_str += f"- {goal['name']}: ₹{goal['current']:,.0f} / ₹{goal['target']:,.0f} ({progress:.1f}% complete)\n"

        user_context_str += f"""
SPENDING BREAKDOWN (Monthly):
"""
        for category, amount in context["spending_summary"][
            "category_breakdown"
        ].items():
            user_context_str += f"- {category.title()}: ₹{amount:,.0f}\n"

        if request.transaction_data:
            user_context_str += f"\nRECENT TRANSACTIONS:\n"
            for txn in request.transaction_data[-5:]:  # Last 5 transactions
                user_context_str += f"- ₹{txn.get('amount', 0)} at {txn.get('merchant_name', 'Unknown')} ({txn.get('category', 'other')})\n"

        full_prompt = f"""
{system_prompt}

{user_context_str}

USER QUERY: {request.query}

Please provide comprehensive financial advice that includes:
1. Direct answer to the user's query
2. 3-5 actionable steps they can take immediately
3. Relevant insights based on their financial profile
4. Any risk warnings or considerations
5. Follow-up questions to better understand their needs

Respond in JSON format with these fields:
- advice: Main advice text
- actionable_steps: Array of specific actions
- insights: Array of relevant insights
- risk_warnings: Array of any warnings
- follow_up_questions: Array of clarifying questions
- confidence_score: Float between 0.0 and 1.0
"""

        return full_prompt

    async def _query_gpt4(self, prompt: str, context_type: str) -> str:
        """Query GPT-4 with the constructed prompt"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.model_config["temperature"],
                max_tokens=self.model_config["max_tokens"],
                top_p=self.model_config["top_p"],
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error querying GPT-4: {e}")
            raise

    async def _parse_advice_response(
        self, gpt_response: str, request: FinancialAdviceRequest
    ) -> FinancialAdviceResponse:
        """Parse GPT-4 response into structured format"""
        try:
            # Try to parse as JSON first
            try:
                response_json = json.loads(gpt_response)
                return FinancialAdviceResponse(
                    advice=response_json.get("advice", gpt_response),
                    confidence_score=response_json.get("confidence_score", 0.8),
                    advice_category=request.context_type,
                    actionable_steps=response_json.get("actionable_steps", []),
                    relevant_insights=response_json.get("insights", []),
                    follow_up_questions=response_json.get("follow_up_questions", []),
                    risk_warnings=response_json.get("risk_warnings", []),
                    supporting_data={"gpt_model": self.model_config["model"]},
                )
            except json.JSONDecodeError:
                # If not valid JSON, parse as text
                return FinancialAdviceResponse(
                    advice=gpt_response,
                    confidence_score=0.7,
                    advice_category=request.context_type,
                    actionable_steps=self._extract_action_items(gpt_response),
                    relevant_insights=[],
                    follow_up_questions=[],
                    risk_warnings=[],
                    supporting_data={
                        "gpt_model": self.model_config["model"],
                        "response_type": "text",
                    },
                )

        except Exception as e:
            logger.error(f"Error parsing GPT-4 response: {e}")
            raise

    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from unstructured text"""
        action_items = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if any(
                marker in line.lower()
                for marker in ["1.", "2.", "3.", "•", "-", "step"]
            ):
                if len(line) > 10:  # Filter out short lines
                    action_items.append(line)

        return action_items[:5]  # Limit to 5 action items

    async def _store_advice_interaction(
        self, request: FinancialAdviceRequest, response: FinancialAdviceResponse
    ):
        """Store advice interaction for continuous learning"""
        try:
            interaction_data = {
                "user_id": request.user_id,
                "query": request.query,
                "context_type": request.context_type,
                "advice_given": response.advice,
                "confidence_score": response.confidence_score,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in feature store for ML learning
            await asyncio.to_thread(
                feature_store.store_features,
                f"advice_{request.user_id}_{datetime.now().strftime('%Y%m%d')}",
                "financial_advisor_interactions",
                interaction_data,
            )

        except Exception as e:
            logger.warning(f"Could not store advice interaction: {e}")

    async def _fallback_advice(
        self, request: FinancialAdviceRequest
    ) -> FinancialAdviceResponse:
        """Provide fallback advice when GPT-4 is unavailable"""
        fallback_responses = {
            "spending_review": {
                "advice": "Based on typical spending patterns, consider reviewing your discretionary expenses and focusing on building an emergency fund of 6-12 months of expenses.",
                "actionable_steps": [
                    "Track your expenses for the next month",
                    "Identify top 3 categories where you can reduce spending",
                    "Set up an automated SIP for ₹5,000/month",
                    "Use UPI apps to monitor real-time spending",
                ],
            },
            "goal_setting": {
                "advice": "Start with building an emergency fund, then focus on long-term goals like home purchase or retirement through systematic investments.",
                "actionable_steps": [
                    "Open a high-yield savings account for emergency fund",
                    "Start SIP in diversified equity mutual funds",
                    "Consider ELSS funds for tax saving under 80C",
                    "Review and increase SIP amount annually",
                ],
            },
        }

        fallback = fallback_responses.get(
            request.context_type, fallback_responses["spending_review"]
        )

        return FinancialAdviceResponse(
            advice=fallback["advice"],
            confidence_score=0.6,
            advice_category=request.context_type,
            actionable_steps=fallback["actionable_steps"],
            relevant_insights=[
                "This is fallback advice - consider consulting a financial advisor for personalized guidance"
            ],
            follow_up_questions=[
                "What are your current financial goals?",
                "What is your monthly income and expenses?",
            ],
            risk_warnings=[
                "Please verify all financial advice with qualified professionals"
            ],
            supporting_data={
                "response_type": "fallback",
                "reason": "GPT-4 unavailable",
            },
        )

    async def analyze_spending_patterns(
        self, user_id: str, transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze user spending patterns and provide insights"""
        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(transactions)

            # Calculate spending insights
            insights = {
                "total_spending": float(df["amount"].sum()),
                "average_transaction": float(df["amount"].mean()),
                "transaction_count": len(df),
                "top_categories": df.groupby("category")["amount"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .to_dict(),
                "spending_trend": "increasing",  # Would calculate from time series
                "unusual_transactions": len(
                    df[df["amount"] > df["amount"].quantile(0.95)]
                ),
                "recommendations": [
                    "Consider setting a budget for your top spending categories",
                    "Track high-value transactions more carefully",
                    "Look for opportunities to reduce discretionary spending",
                ],
            }

            return insights

        except Exception as e:
            logger.error(f"Error analyzing spending patterns: {e}")
            return {"error": str(e)}

    async def suggest_financial_goals(
        self, user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest appropriate financial goals based on user profile"""
        suggestions = []

        income = user_profile.get("monthly_income", 50000)
        age = user_profile.get("age", 30)

        # Emergency Fund Goal
        emergency_target = income * 6  # 6 months of income
        suggestions.append(
            {
                "goal_name": "Emergency Fund",
                "target_amount": emergency_target,
                "timeline_months": 12,
                "monthly_sip": emergency_target / 12,
                "priority": "high",
                "description": "Build a safety net for unexpected expenses",
            }
        )

        # Retirement Goal
        retirement_target = income * 12 * 25  # 25x annual income
        years_to_retirement = max(58 - age, 10)
        suggestions.append(
            {
                "goal_name": "Retirement Fund",
                "target_amount": retirement_target,
                "timeline_months": years_to_retirement * 12,
                "monthly_sip": retirement_target
                / (years_to_retirement * 12)
                * 0.6,  # Assuming 7% returns
                "priority": "medium",
                "description": "Secure your retirement with systematic investments",
            }
        )

        # Home Purchase (if applicable)
        if age < 35:
            home_target = income * 60  # 5x annual income
            suggestions.append(
                {
                    "goal_name": "Home Purchase",
                    "target_amount": home_target,
                    "timeline_months": 60,  # 5 years
                    "monthly_sip": home_target / 60 * 0.8,  # 80% funding needed
                    "priority": "medium",
                    "description": "Save for your dream home down payment",
                }
            )

        return suggestions


# Global financial advisor instance
financial_advisor = GPT4FinancialAdvisor()
