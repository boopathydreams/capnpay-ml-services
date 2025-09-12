"""CLI to train Champion model from a CSV using the parity pipeline.

Usage:
  cd ml-services
  python -m training.train_champion --csv data/your_training_data.csv --version v1
"""
import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.auto_tagging import XGBoostAutoTagger


def main():
    parser = argparse.ArgumentParser(description="Train Champion auto-tagging model")
    parser.add_argument("--csv", required=True, help="Path to labeled training CSV")
    parser.add_argument("--version", default="v1", help="Model version label for MLflow")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)

    print(f"📥 Loading data: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"user_id", "merchant_name", "amount", "timestamp", "category"}
    missing = required - set(df.columns)
    if missing:
        print(f"❌ Missing required columns: {missing}")
        sys.exit(1)

    print(f"🚀 Training Champion on {len(df)} rows...")
    tagger = XGBoostAutoTagger()
    metrics = tagger.train(df)
    print("✅ Training complete")

    # Optionally log to MLflow and persist classic artifacts
    try:
        run_id = tagger.save_model(args.version)
        print(f"📦 Saved to MLflow (run_id={run_id})")
    except Exception as e:
        print(f"⚠️ Skipped MLflow save: {e}")

    print("\n=== Champion Summary ===")
    print(f"Accuracy: {metrics.get('accuracy'):.4f}")
    print(f"CV mean: {metrics.get('cv_mean_accuracy'):.4f} ± {metrics.get('cv_std_accuracy'):.4f}")
    print(f"Samples: {metrics.get('n_samples')}")
    print("Artifacts: training/model_artifacts/champion/")


if __name__ == "__main__":
    main()
