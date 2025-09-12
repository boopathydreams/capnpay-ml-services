"""
Redis Feature Store for Cap'n Pay AI
High-performance feature caching and serving for real-time ML inference
"""

import redis
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import hashlib
import pickle
from dataclasses import dataclass, asdict
import numpy as np

from config.mlflow_config import FEATURE_STORE_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Feature set metadata"""

    name: str
    version: str
    features: List[str]
    ttl_seconds: int
    created_at: datetime
    updated_at: datetime


class RedisFeatureStore:
    """
    Redis-based feature store for real-time ML inference
    Supports feature caching, versioning, and batch operations
    """

    def __init__(self):
        # Build Redis connection arguments
        redis_args = {
            "host": FEATURE_STORE_CONFIG["redis_host"],
            "port": FEATURE_STORE_CONFIG["redis_port"],
            "db": FEATURE_STORE_CONFIG["redis_db"],
        }

        # Add optional authentication if provided
        if FEATURE_STORE_CONFIG.get("redis_password"):
            redis_args["password"] = FEATURE_STORE_CONFIG["redis_password"]
        if FEATURE_STORE_CONFIG.get("redis_username"):
            redis_args["username"] = FEATURE_STORE_CONFIG["redis_username"]

        self.redis_client = redis.Redis(
            **redis_args,
            decode_responses=False,  # Keep binary for pickle
        )
        self.redis_str_client = redis.Redis(
            **redis_args,
            decode_responses=True,  # For string operations
        )
        self.default_ttl = FEATURE_STORE_CONFIG["feature_ttl"]

        # Feature set registry
        self.feature_sets = {
            "payment_features": [
                "amount",
                "merchant_category",
                "time_of_day",
                "day_of_week",
                "user_spending_last_7d",
                "user_spending_last_30d",
                "merchant_frequency",
                "location_category",
                "payment_method",
                "hour_bin",
                "amount_percentile",
                "velocity_score",
            ],
            "behavioral_features": [
                "avg_transaction_amount",
                "transaction_frequency",
                "weekend_ratio",
                "night_transaction_ratio",
                "category_diversity",
                "spending_trend",
                "seasonal_pattern",
                "impulse_score",
                "budget_adherence",
            ],
            "trust_features": [
                "transaction_history_score",
                "network_centrality",
                "fraud_risk_score",
                "behavioral_consistency",
                "identity_verification",
                "device_trust",
                "location_trust",
                "time_pattern_trust",
                "amount_pattern_trust",
            ],
            "merchant_features": [
                "merchant_trust_score",
                "community_rating",
                "transaction_volume",
                "category_confidence",
                "fraud_reports",
                "verification_status",
                "business_age",
                "location_verified",
                "payment_success_rate",
            ],
        }

        logger.info("Redis Feature Store initialized")

    def _get_feature_key(
        self, entity_id: str, feature_set: str, version: str = "v1"
    ) -> str:
        """Generate Redis key for feature set"""
        return f"features:{feature_set}:{version}:{entity_id}"

    def _get_metadata_key(self, feature_set: str, version: str = "v1") -> str:
        """Generate Redis key for feature set metadata"""
        return f"metadata:{feature_set}:{version}"

    def store_features(
        self,
        entity_id: str,
        feature_set: str,
        features: Dict[str, Any],
        version: str = "v1",
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Store features for an entity (user, merchant, etc.)

        Args:
            entity_id: Unique identifier (user_id, merchant_id, etc.)
            feature_set: Feature set name (payment_features, behavioral_features, etc.)
            features: Dictionary of feature values
            version: Feature set version
            ttl_seconds: Time to live in seconds

        Returns:
            Success boolean
        """
        try:
            key = self._get_feature_key(entity_id, feature_set, version)
            ttl = ttl_seconds or self.default_ttl

            # Validate features
            expected_features = self.feature_sets.get(feature_set, [])
            if expected_features:
                missing_features = set(expected_features) - set(features.keys())
                if missing_features:
                    logger.warning(
                        f"Missing features for {feature_set}: {missing_features}"
                    )

            # Add metadata
            feature_data = {
                "features": features,
                "stored_at": datetime.now().isoformat(),
                "entity_id": entity_id,
                "feature_set": feature_set,
                "version": version,
            }

            # Store as pickle for efficient serialization
            serialized_data = pickle.dumps(feature_data)

            # Store with TTL
            result = self.redis_client.setex(key, ttl, serialized_data)

            # Update access statistics
            stats_key = f"stats:{feature_set}:{version}"
            self.redis_str_client.hincrby(stats_key, "store_count", 1)
            self.redis_str_client.hset(
                stats_key, "last_stored", datetime.now().isoformat()
            )

            logger.debug(f"Stored features for {entity_id} in {feature_set}")
            return result

        except Exception as e:
            logger.error(f"Error storing features for {entity_id}: {e}")
            return False

    def get_features(
        self, entity_id: str, feature_set: str, version: str = "v1"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve features for an entity

        Args:
            entity_id: Entity identifier
            feature_set: Feature set name
            version: Feature set version

        Returns:
            Dictionary of features or None if not found
        """
        try:
            key = self._get_feature_key(entity_id, feature_set, version)

            # Get from Redis
            serialized_data = self.redis_client.get(key)
            if not serialized_data:
                logger.debug(f"Features not found for {entity_id} in {feature_set}")
                return None

            # Deserialize
            feature_data = pickle.loads(serialized_data)

            # Update access statistics
            stats_key = f"stats:{feature_set}:{version}"
            self.redis_str_client.hincrby(stats_key, "get_count", 1)
            self.redis_str_client.hset(
                stats_key, "last_accessed", datetime.now().isoformat()
            )

            return feature_data["features"]

        except Exception as e:
            logger.error(f"Error retrieving features for {entity_id}: {e}")
            return None

    def batch_store_features(
        self, feature_data: List[Dict[str, Any]], feature_set: str, version: str = "v1"
    ) -> int:
        """
        Store features for multiple entities in batch

        Args:
            feature_data: List of dicts with 'entity_id' and 'features' keys
            feature_set: Feature set name
            version: Feature set version

        Returns:
            Number of successfully stored feature sets
        """
        try:
            pipe = self.redis_client.pipeline()
            stored_count = 0

            for data in feature_data:
                entity_id = data["entity_id"]
                features = data["features"]

                key = self._get_feature_key(entity_id, feature_set, version)

                feature_data_obj = {
                    "features": features,
                    "stored_at": datetime.now().isoformat(),
                    "entity_id": entity_id,
                    "feature_set": feature_set,
                    "version": version,
                }

                serialized_data = pickle.dumps(feature_data_obj)
                pipe.setex(key, self.default_ttl, serialized_data)
                stored_count += 1

            # Execute pipeline
            pipe.execute()

            # Update stats
            stats_key = f"stats:{feature_set}:{version}"
            self.redis_str_client.hincrby(stats_key, "batch_store_count", stored_count)

            logger.info(f"Batch stored {stored_count} feature sets for {feature_set}")
            return stored_count

        except Exception as e:
            logger.error(f"Error in batch store: {e}")
            return 0

    def batch_get_features(
        self, entity_ids: List[str], feature_set: str, version: str = "v1"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve features for multiple entities in batch

        Args:
            entity_ids: List of entity identifiers
            feature_set: Feature set name
            version: Feature set version

        Returns:
            Dictionary mapping entity_id to features
        """
        try:
            pipe = self.redis_client.pipeline()
            keys = []

            # Queue all gets
            for entity_id in entity_ids:
                key = self._get_feature_key(entity_id, feature_set, version)
                keys.append((key, entity_id))
                pipe.get(key)

            # Execute pipeline
            results = pipe.execute()

            # Parse results
            features_by_entity = {}
            found_count = 0

            for i, (key, entity_id) in enumerate(keys):
                serialized_data = results[i]
                if serialized_data:
                    try:
                        feature_data = pickle.loads(serialized_data)
                        features_by_entity[entity_id] = feature_data["features"]
                        found_count += 1
                    except Exception as e:
                        logger.warning(
                            f"Error deserializing features for {entity_id}: {e}"
                        )

            # Update stats
            stats_key = f"stats:{feature_set}:{version}"
            self.redis_str_client.hincrby(stats_key, "batch_get_count", len(entity_ids))
            self.redis_str_client.hincrby(stats_key, "batch_hit_count", found_count)

            logger.debug(
                f"Batch retrieved {found_count}/{len(entity_ids)} feature sets"
            )
            return features_by_entity

        except Exception as e:
            logger.error(f"Error in batch get: {e}")
            return {}

    def invalidate_features(
        self, entity_id: str, feature_set: str, version: str = "v1"
    ) -> bool:
        """Invalidate (delete) features for an entity"""
        try:
            key = self._get_feature_key(entity_id, feature_set, version)
            result = self.redis_client.delete(key)

            # Update stats
            stats_key = f"stats:{feature_set}:{version}"
            self.redis_str_client.hincrby(stats_key, "invalidate_count", 1)

            return bool(result)

        except Exception as e:
            logger.error(f"Error invalidating features for {entity_id}: {e}")
            return False

    def get_feature_freshness(
        self, entity_id: str, feature_set: str, version: str = "v1"
    ) -> Optional[Dict[str, Any]]:
        """Get feature freshness information"""
        try:
            key = self._get_feature_key(entity_id, feature_set, version)

            # Get TTL and creation time
            ttl = self.redis_client.ttl(key)
            if ttl == -2:  # Key doesn't exist
                return None

            serialized_data = self.redis_client.get(key)
            if not serialized_data:
                return None

            feature_data = pickle.loads(serialized_data)
            stored_at = datetime.fromisoformat(feature_data["stored_at"])

            return {
                "entity_id": entity_id,
                "feature_set": feature_set,
                "version": version,
                "stored_at": stored_at,
                "age_seconds": (datetime.now() - stored_at).total_seconds(),
                "ttl_seconds": ttl,
                "expires_at": (
                    datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
                ),
            }

        except Exception as e:
            logger.error(f"Error getting freshness for {entity_id}: {e}")
            return None

    def get_feature_stats(
        self, feature_set: str, version: str = "v1"
    ) -> Dict[str, Any]:
        """Get statistics for a feature set"""
        try:
            stats_key = f"stats:{feature_set}:{version}"
            stats = self.redis_str_client.hgetall(stats_key)

            # Convert string numbers to integers
            for key in [
                "store_count",
                "get_count",
                "batch_store_count",
                "batch_get_count",
                "batch_hit_count",
                "invalidate_count",
            ]:
                if key in stats:
                    stats[key] = int(stats[key])

            # Calculate hit rate
            if "get_count" in stats and stats["get_count"] > 0:
                stats["hit_rate"] = stats.get("batch_hit_count", 0) / max(
                    stats.get("batch_get_count", 1), 1
                )

            return stats

        except Exception as e:
            logger.error(f"Error getting stats for {feature_set}: {e}")
            return {}

    def cleanup_expired_features(self) -> int:
        """Clean up expired features (manual cleanup for monitoring)"""
        try:
            cleaned_count = 0

            for feature_set in self.feature_sets.keys():
                pattern = f"features:{feature_set}:*"
                keys = self.redis_str_client.keys(pattern)

                for key in keys:
                    ttl = self.redis_client.ttl(key)
                    if ttl == -2:  # Already expired
                        cleaned_count += 1

            logger.info(f"Cleaned up {cleaned_count} expired features")
            return cleaned_count

        except Exception as e:
            logger.error(f"Error cleaning up features: {e}")
            return 0

    def health_check(self) -> Dict[str, Any]:
        """Feature store health check"""
        try:
            # Test Redis connection
            self.redis_client.ping()

            # Get memory info
            memory_info = self.redis_client.info("memory")

            # Count features by set
            feature_counts = {}
            for feature_set in self.feature_sets.keys():
                pattern = f"features:{feature_set}:*"
                count = len(self.redis_str_client.keys(pattern))
                feature_counts[feature_set] = count

            return {
                "status": "healthy",
                "redis_connected": True,
                "memory_used_mb": memory_info["used_memory"] / (1024 * 1024),
                "feature_counts": feature_counts,
                "total_features": sum(feature_counts.values()),
                "feature_sets": list(self.feature_sets.keys()),
            }

        except Exception as e:
            logger.error(f"Feature store health check failed: {e}")
            return {"status": "unhealthy", "redis_connected": False, "error": str(e)}


# Global feature store instance
feature_store = RedisFeatureStore()
