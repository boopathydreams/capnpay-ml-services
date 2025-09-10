"""
Sample Data Generator for Cap'n Pay ML Training
Creates realistic Indian fintech transaction data for model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
import uuid


class SampleDataGenerator:
    """Generate realistic transaction data for ML training"""

    def __init__(self):
        # Indian merchant patterns
        self.merchants = {
            "food": [
                "Zomato",
                "Swiggy",
                "Dominos Pizza",
                "McDonalds",
                "KFC",
                "Pizza Hut",
                "Cafe Coffee Day",
                "Starbucks",
                "Haldirams",
                "Sagar Ratna",
                "Barbeque Nation",
                "Local Restaurant",
                "Street Food Vendor",
            ],
            "transport": [
                "Uber",
                "Ola Cabs",
                "Delhi Metro",
                "Mumbai Local",
                "BMTC Bus",
                "Indian Oil Petrol",
                "HP Petrol",
                "Bharat Petroleum",
                "Auto Rickshaw",
            ],
            "shopping": [
                "Amazon India",
                "Flipkart",
                "Big Bazaar",
                "Reliance Fresh",
                "DMart",
                "Shoppers Stop",
                "Lifestyle",
                "Myntra",
                "Ajio",
                "Local Grocery Store",
                "Medical Store",
            ],
            "utilities": [
                "BSES Electricity",
                "Airtel Mobile",
                "Jio Recharge",
                "Vi Vodafone",
                "Tata Sky DTH",
                "Airtel Broadband",
                "Water Bill Payment",
                "Gas Cylinder Booking",
                "Municipal Tax",
            ],
            "entertainment": [
                "BookMyShow",
                "Netflix India",
                "Amazon Prime",
                "Spotify",
                "PVR Cinemas",
                "INOX Movies",
                "Gaming Zone",
                "Amusement Park",
            ],
            "healthcare": [
                "Apollo Hospital",
                "Max Healthcare",
                "Local Pharmacy",
                "Diagnostic Center",
                "Dental Clinic",
                "Eye Care",
            ],
            "education": [
                "School Fees",
                "College Tuition",
                "Coaching Classes",
                "Online Course",
                "Book Store",
                "Stationery Shop",
            ],
            "investment": [
                "SIP Investment",
                "Mutual Fund",
                "LIC Premium",
                "HDFC Bank FD",
                "Stock Purchase",
                "Gold Investment",
            ],
        }

        # Amount patterns by category (in INR)
        self.amount_patterns = {
            "food": (50, 2000, "lognormal"),
            "transport": (20, 500, "exponential"),
            "shopping": (100, 10000, "lognormal"),
            "utilities": (200, 5000, "normal"),
            "entertainment": (100, 3000, "lognormal"),
            "healthcare": (500, 20000, "lognormal"),
            "education": (1000, 50000, "normal"),
            "investment": (1000, 100000, "lognormal"),
        }

        # Time patterns by category
        self.time_patterns = {
            "food": [11, 12, 13, 19, 20, 21],  # Meal times
            "transport": [8, 9, 18, 19, 20],  # Commute times
            "shopping": [10, 11, 15, 16, 17],  # Shopping hours
            "utilities": [10, 11, 12, 14, 15],  # Business hours
            "entertainment": [18, 19, 20, 21, 22],  # Evening
            "healthcare": [9, 10, 11, 14, 15, 16],  # Clinic hours
            "education": [9, 10, 14, 15, 16],  # School/college hours
            "investment": [10, 11, 12, 14, 15],  # Business hours
        }

    def generate_amount(self, category: str) -> float:
        """Generate realistic amount for category"""
        min_amt, max_amt, distribution = self.amount_patterns.get(
            category, (100, 5000, "normal")
        )

        if distribution == "lognormal":
            # Log-normal for skewed distributions (food, shopping)
            amount = np.random.lognormal(np.log(min_amt * 2), 0.8)
        elif distribution == "exponential":
            # Exponential for transport (many small, few large)
            amount = np.random.exponential(min_amt * 2)
        else:
            # Normal for utilities, education
            amount = np.random.normal((min_amt + max_amt) / 2, (max_amt - min_amt) / 6)

        # Clamp to reasonable bounds
        amount = max(min_amt, min(max_amt, amount))

        # Round to realistic precision
        if amount < 100:
            return round(amount)
        elif amount < 1000:
            return round(amount / 10) * 10
        else:
            return round(amount / 50) * 50

    def generate_timestamp(self, category: str, base_date: datetime = None) -> datetime:
        """Generate realistic timestamp for category"""
        if base_date is None:
            base_date = datetime.now() - timedelta(days=random.randint(1, 90))

        # Get preferred hours for category
        preferred_hours = self.time_patterns.get(category, [10, 11, 12, 14, 15, 16])
        hour = random.choice(preferred_hours)
        minute = random.randint(0, 59)

        # Add some randomness to dates
        date_offset = random.randint(-7, 7)  # ±1 week variation
        final_date = base_date + timedelta(days=date_offset, hours=hour, minutes=minute)

        return final_date

    def generate_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Generate user spending profile"""
        # User spending patterns
        profiles = [
            {
                "name": "budget_conscious",
                "spending_categories": ["food", "transport", "utilities"],
                "avg_transaction": 300,
                "frequency": 15,  # transactions per month
                "variance": 0.3,
            },
            {
                "name": "tech_professional",
                "spending_categories": [
                    "food",
                    "transport",
                    "shopping",
                    "entertainment",
                ],
                "avg_transaction": 800,
                "frequency": 25,
                "variance": 0.5,
            },
            {
                "name": "family_person",
                "spending_categories": [
                    "food",
                    "shopping",
                    "utilities",
                    "healthcare",
                    "education",
                ],
                "avg_transaction": 1200,
                "frequency": 30,
                "variance": 0.4,
            },
            {
                "name": "investor",
                "spending_categories": [
                    "investment",
                    "utilities",
                    "food",
                    "entertainment",
                ],
                "avg_transaction": 2000,
                "frequency": 20,
                "variance": 0.8,
            },
        ]

        return random.choice(profiles)

    def generate_transactions(
        self, n_users: int = 100, n_transactions: int = 5000
    ) -> pd.DataFrame:
        """
        Generate realistic transaction dataset

        Args:
            n_users: Number of unique users
            n_transactions: Total number of transactions

        Returns:
            DataFrame with transaction data
        """
        transactions = []
        user_ids = [f"user_{i:04d}" for i in range(n_users)]

        for _ in range(n_transactions):
            user_id = random.choice(user_ids)
            user_profile = self.generate_user_profile(user_id)

            # Select category based on user profile
            category = random.choice(user_profile["spending_categories"])

            # Select merchant
            merchant = random.choice(self.merchants[category])

            # Generate amount with user profile influence
            base_amount = self.generate_amount(category)
            user_multiplier = np.random.normal(1.0, user_profile["variance"])
            amount = max(10, base_amount * user_multiplier)

            # Generate timestamp
            timestamp = self.generate_timestamp(category)

            # Add some noise for realism
            if random.random() < 0.05:  # 5% wrong categories (labeling noise)
                category = random.choice(list(self.merchants.keys()))

            transaction = {
                "transaction_id": str(uuid.uuid4()),
                "user_id": user_id,
                "merchant_name": merchant,
                "amount": round(amount, 2),
                "timestamp": timestamp.isoformat(),
                "category": category,
            }

            transactions.append(transaction)

        df = pd.DataFrame(transactions)

        # Sort by timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"✅ Generated {len(df)} transactions for {n_users} users")
        print(f"📊 Category distribution:")
        print(df["category"].value_counts())
        print(f"💰 Amount statistics:")
        print(df["amount"].describe())

        return df


# Global generator instance
data_generator = SampleDataGenerator()
