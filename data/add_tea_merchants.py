#!/usr/bin/env python3
"""Add more tea merchants to improve training data"""

import pandas as pd
import csv


def add_tea_merchants():
    print("🍵 Adding Tea Merchants to Improve Training Data")
    print("=" * 50)

    # Read existing data
    df = pd.read_csv("merchant_category.csv")
    print(f"Current merchants: {len(df)}")

    # Additional tea merchants to add
    new_tea_merchants = [
        ("Tea Point", "Food & Dining", "Cafe/Dessert"),
        ("Chai Wala", "Food & Dining", "Street Food"),
        ("Tea Junction", "Food & Dining", "Cafe/Dessert"),
        ("Chai Point", "Food & Dining", "Cafe/Dessert"),
        ("Kulhad Chai", "Food & Dining", "Street Food"),
        ("Tea Time Cafe", "Food & Dining", "Cafe/Dessert"),
        ("Masala Chai Corner", "Food & Dining", "Street Food"),
        ("Green Tea House", "Food & Dining", "Cafe/Dessert"),
        ("Tapri Tea", "Food & Dining", "Street Food"),
        ("Cha Bar", "Food & Dining", "Cafe/Dessert"),
        ("Irani Chai", "Food & Dining", "Cafe/Dessert"),
        ("Cutting Chai", "Food & Dining", "Street Food"),
        ("Tea Lounge", "Food & Dining", "Cafe/Dessert"),
        ("Chai Sutta Bar", "Food & Dining", "Cafe/Dessert"),
        ("Tea Villa Cafe", "Food & Dining", "Cafe/Dessert"),
    ]

    # Check for duplicates
    existing_merchants = set(df["Merchant Name"].str.lower())
    new_merchants_to_add = []

    for merchant, category, subcategory in new_tea_merchants:
        if merchant.lower() not in existing_merchants:
            new_merchants_to_add.append(
                {
                    "Merchant Name": merchant,
                    "Category": category,
                    "Subcategory": subcategory,
                }
            )
        else:
            print(f"⚠️ Skipping duplicate: {merchant}")

    if new_merchants_to_add:
        # Add new merchants
        new_df = pd.DataFrame(new_merchants_to_add)
        updated_df = pd.concat([df, new_df], ignore_index=True)

        # Save updated file
        updated_df.to_csv("merchant_category.csv", index=False)
        print(f"✅ Added {len(new_merchants_to_add)} new tea merchants")
        print(f"📊 Total merchants: {len(updated_df)}")

        # Show added merchants
        print("\n🍵 Added Tea Merchants:")
        for merchant in new_merchants_to_add:
            print(f"  {merchant['Merchant Name']} → {merchant['Category']}")
    else:
        print("ℹ️ No new merchants to add (all already exist)")


if __name__ == "__main__":
    add_tea_merchants()
