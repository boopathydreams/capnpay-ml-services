#!/usr/bin/env python3
"""
Prepare Training Data
Create training CSV from merchant data
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


def create_training_data():
    """Create training data CSV"""
    print("📊 Loading merchant data...")

    # Load merchant data
    merchant_df = pd.read_csv("data/merchant_category.csv")
    print(f"✅ Loaded {len(merchant_df)} merchants")

    # Category mapping
    category_map = {
        "Food & Dining": "food",
        "Transport": "transport",
        "Shopping": "shopping",
        "Bills & Utilities": "utilities",
        "Digital Subscriptions": "entertainment",
        "Education & Coaching": "education",
        "Finance & Insurance": "investment",
        "Healthcare & Pharmacy": "healthcare",
        "Groceries & Kirana": "shopping",
        "Travel & Hospitality": "transport",
        "Fashion & Lifestyle": "shopping",
        "Government & NGOs": "other",
        "Home & Services": "other",
    }

    # Generate transactions
    transactions = []
    users = [f"user_{i:04d}" for i in range(200)]

    print("🔄 Generating transactions...")
    for i in range(5000):
        # Random merchant
        merchant_row = merchant_df.sample(1).iloc[0]
        merchant_name = merchant_row["Merchant Name"]
        csv_category = merchant_row["Category"]
        category = category_map.get(csv_category, "other")

        # Random user
        user_id = random.choice(users)

        # Amount based on category
        if category == "food":
            amount = np.random.lognormal(5.5, 0.8)  # ~₹200-2000
        elif category == "transport":
            amount = np.random.exponential(150)  # ~₹50-500
        elif category == "shopping":
            amount = np.random.lognormal(6.5, 1.0)  # ~₹500-5000
        elif category == "utilities":
            amount = np.random.normal(1500, 500)  # ~₹500-3000
        else:
            amount = np.random.lognormal(6.0, 0.9)  # ~₹300-3000

        amount = max(10, min(50000, round(amount, 2)))

        # Random timestamp (last 3 months)
        base_date = datetime.now() - timedelta(days=90)
        days_offset = random.randint(0, 90)
        hours_offset = random.randint(0, 23)
        timestamp = base_date + timedelta(days=days_offset, hours=hours_offset)

        transactions.append(
            {
                "user_id": user_id,
                "merchant_name": merchant_name,
                "amount": amount,
                "timestamp": timestamp.isoformat(),
                "category": category,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(transactions)

    # Save to CSV
    output_file = "data/training_data.csv"
    df.to_csv(output_file, index=False)

    print(f"✅ Saved {len(df)} transactions to {output_file}")
    print(f"📋 Categories: {df['category'].value_counts().to_dict()}")

    return output_file


if __name__ == "__main__":
    create_training_data()
