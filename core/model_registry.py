"""
MLflow Model Registry Setup and Management
Handles all model versioning, experiment tracking, and deployment for Cap'n Pay AI systems
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_REGISTRY_SETTINGS,
    EXPERIMENTS,
    MONITORING_CONFIG,
)

logger = logging.getLogger(__name__)


class MLflowModelRegistry:
    """Central MLflow registry for all Cap'n Pay AI models"""

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        self._setup_experiments()
        self._register_models()

    def _setup_experiments(self):
        """Create MLflow experiments for each AI component"""
        try:
            # Main experiment
            experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            if experiment is None:
                mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
                logger.info(f"Created main experiment: {MLFLOW_EXPERIMENT_NAME}")

            # Component-specific experiments
            for component, config in EXPERIMENTS.items():
                exp_name = f"{MLFLOW_EXPERIMENT_NAME}-{component}"
                experiment = mlflow.get_experiment_by_name(exp_name)
                if experiment is None:
                    mlflow.create_experiment(
                        exp_name,
                        tags={
                            "component": component,
                            "baseline_model": config["baseline"],
                            "champion_model": config["champion"],
                            "models": json.dumps(config["models"]),
                        },
                    )
                    logger.info(f"Created experiment: {exp_name}")

        except Exception as e:
            logger.error(f"Error setting up experiments: {e}")
            raise

    def _register_models(self):
        """Register all model names in MLflow Model Registry"""
        try:
            for component, settings in MODEL_REGISTRY_SETTINGS.items():
                model_name = settings["name"]

                # Check if model already exists
                try:
                    self.client.get_registered_model(model_name)
                    logger.info(f"Model {model_name} already registered")
                except mlflow.exceptions.MlflowException:
                    # Model doesn't exist, create it
                    self.client.create_registered_model(
                        model_name,
                        description=settings["description"],
                        tags={
                            "component": component,
                            "target_accuracy": str(
                                settings.get("target_accuracy", "N/A")
                            ),
                            "target_latency_ms": str(
                                settings.get("target_latency_ms", "N/A")
                            ),
                            "target_satisfaction": str(
                                settings.get("target_satisfaction", "N/A")
                            ),
                            "target_engagement": str(
                                settings.get("target_engagement", "N/A")
                            ),
                            "target_consensus": str(
                                settings.get("target_consensus", "N/A")
                            ),
                        },
                    )
                    logger.info(f"Registered model: {model_name}")

        except Exception as e:
            logger.error(f"Error registering models: {e}")
            raise

    def log_model(
        self,
        component: str,
        model: Any,
        model_type: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        artifacts: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Log a model to MLflow with proper experiment tracking

        Args:
            component: AI component name (auto_tagging, behavioral_nudges, etc.)
            model: The trained model object
            model_type: Type of model (xgboost, pytorch, sklearn, etc.)
            metrics: Performance metrics
            params: Model parameters
            artifacts: Additional artifacts to log
            tags: Custom tags

        Returns:
            run_id: MLflow run ID
        """
        exp_name = f"{MLFLOW_EXPERIMENT_NAME}-{component}"
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(tags=tags or {}) as run:
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model based on type
            if model_type == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            elif model_type == "sklearn":
                mlflow.sklearn.log_model(model, "model")
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(model, "model")
            else:
                # Generic pickle logging
                mlflow.sklearn.log_model(model, "model")

            # Log additional artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)

            # Log system info
            mlflow.log_param("timestamp", datetime.now().isoformat())
            mlflow.log_param("component", component)
            mlflow.log_param("model_type", model_type)

            logger.info(f"Logged {component} model with run_id: {run.info.run_id}")
            return run.info.run_id

    def promote_to_staging(self, model_name: str, run_id: str) -> str:
        """Promote model version to Staging"""
        model_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)

        self.client.transition_model_version_stage(
            name=model_name, version=model_version.version, stage="Staging"
        )

        logger.info(f"Promoted {model_name} v{model_version.version} to Staging")
        return model_version.version

    def promote_to_production(self, model_name: str, version: str) -> None:
        """Promote model version to Production (Champion)"""
        self.client.transition_model_version_stage(
            name=model_name, version=version, stage="Production"
        )

        logger.info(f"Promoted {model_name} v{version} to Production")

    def get_model(self, component: str, stage: str = "Production") -> Any:
        """Load model from registry"""
        model_name = MODEL_REGISTRY_SETTINGS[component]["name"]
        model_uri = f"models:/{model_name}/{stage}"

        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded {component} model from {stage}")
            return model
        except Exception as e:
            logger.error(f"Error loading {component} model: {e}")
            raise

    def get_model_metrics(
        self, component: str, stage: str = "Production"
    ) -> Dict[str, float]:
        """Get performance metrics for a model"""
        model_name = MODEL_REGISTRY_SETTINGS[component]["name"]

        try:
            model_version = self.client.get_latest_versions(model_name, stages=[stage])[
                0
            ]

            run = self.client.get_run(model_version.run_id)
            return run.data.metrics

        except Exception as e:
            logger.error(f"Error getting metrics for {component}: {e}")
            return {}

    def compare_models(self, component: str, metric: str) -> Dict[str, Any]:
        """Compare Staging vs Production models"""
        try:
            staging_metrics = self.get_model_metrics(component, "Staging")
            production_metrics = self.get_model_metrics(component, "Production")

            comparison = {
                "component": component,
                "metric": metric,
                "staging": staging_metrics.get(metric, 0.0),
                "production": production_metrics.get(metric, 0.0),
                "improvement": staging_metrics.get(metric, 0.0)
                - production_metrics.get(metric, 0.0),
                "should_promote": False,
            }

            # Check if staging model meets promotion criteria
            threshold = MONITORING_CONFIG["accuracy_threshold"].get(component, 0.85)
            if comparison["staging"] >= threshold and comparison["improvement"] > 0.01:
                comparison["should_promote"] = True

            return comparison

        except Exception as e:
            logger.error(f"Error comparing models for {component}: {e}")
            return {}

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with metadata"""
        experiments = []

        for component in EXPERIMENTS.keys():
            exp_name = f"{MLFLOW_EXPERIMENT_NAME}-{component}"
            try:
                experiment = mlflow.get_experiment_by_name(exp_name)
                if experiment:
                    experiments.append(
                        {
                            "component": component,
                            "experiment_id": experiment.experiment_id,
                            "name": exp_name,
                            "lifecycle_stage": experiment.lifecycle_stage,
                            "tags": experiment.tags,
                        }
                    )
            except Exception as e:
                logger.warning(f"Could not get experiment {exp_name}: {e}")

        return experiments

    def cleanup_old_runs(self, max_runs_per_experiment: int = 100) -> None:
        """Clean up old experiment runs to save storage"""
        for component in EXPERIMENTS.keys():
            exp_name = f"{MLFLOW_EXPERIMENT_NAME}-{component}"
            try:
                experiment = mlflow.get_experiment_by_name(exp_name)
                if experiment:
                    runs = self.client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"],
                    )

                    if len(runs) > max_runs_per_experiment:
                        old_runs = runs[max_runs_per_experiment:]
                        for run in old_runs:
                            self.client.delete_run(run.info.run_id)

                        logger.info(
                            f"Cleaned up {len(old_runs)} old runs for {component}"
                        )

            except Exception as e:
                logger.warning(f"Error cleaning up {component} runs: {e}")


# Global registry instance
model_registry = MLflowModelRegistry()
