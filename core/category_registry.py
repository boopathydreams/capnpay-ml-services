"""Category registry for dynamic taxonomy."""
from pathlib import Path
import json
from typing import List

DEFAULT_CATEGORIES = [
    "Food & Dining",
    "Shopping",
    "Transport",
    "Healthcare",
    "Personal",
    "Bills",
    # P2P subtypes under Personal (optional use in UI)
    "P2P: Family Transfer",
    "P2P: Loan Given",
    "P2P: Loan Repayment",
    "P2P: Gift",
    "P2P: Tip",
    "P2P: Reimbursement",
]


def load_categories(path: Path) -> List[str]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return DEFAULT_CATEGORIES
    return DEFAULT_CATEGORIES

