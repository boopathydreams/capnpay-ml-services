"""Community label store with simple JSON persistence.

Key: signature = hash(normalized_merchant + vpa_handle + city_geohash5(optional))
Value: {category_name, confidence, votes, last_seen}
"""
from pathlib import Path
import json
import hashlib
from typing import Optional, Dict, Any

STORE_PATH = Path("ml-services/data/community_labels.json")


def _normalize(text: str) -> str:
    import re
    if not isinstance(text, str):
        return "unknown"
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", s).strip()


def signature(merchant_name: str, vpa_handle: str = "", geohash5: str = "") -> str:
    raw = f"{_normalize(merchant_name)}|{vpa_handle.lower()}|{geohash5}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_store() -> Dict[str, Any]:
    if STORE_PATH.exists():
        try:
            return json.loads(STORE_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_store(store: Dict[str, Any]):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, indent=2))


def get_label(merchant_name: str, vpa_handle: str = "", geohash5: str = "") -> Optional[Dict[str, Any]]:
    store = load_store()
    key = signature(merchant_name, vpa_handle, geohash5)
    return store.get(key)


def set_label(merchant_name: str, category_name: str, confidence: float = 0.9, vpa_handle: str = "", geohash5: str = ""):
    store = load_store()
    key = signature(merchant_name, vpa_handle, geohash5)
    entry = store.get(key, {"votes": 0})
    entry.update({"category_name": category_name, "confidence": confidence})
    entry["votes"] = int(entry.get("votes", 0)) + 1
    store[key] = entry
    save_store(store)

