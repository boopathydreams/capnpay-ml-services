# Cap'n Pay ML Services

This directory contains the FastAPI-based machine learning services for Cap'n Pay AI features.

## Services

- **Enhanced Tagging Service**: XGBoost-based payment categorization
- **Behavioral Analysis Service**: Psychology-based spending insights
- **Voice Intelligence Service**: Audio processing and NLP analysis
- **Trust Scoring Service**: Multi-factor contact risk assessment
- **Merchant Intelligence Service**: Community-driven merchant categorization

## Architecture

```
ml-services/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── models/              # ML model definitions
│   ├── services/            # Business logic services
│   ├── api/                 # API route handlers
│   └── core/                # Configuration and utilities
├── models/                  # Trained model files
├── data/                    # Training data and features
├── requirements.txt         # Python dependencies
└── Dockerfile              # Container deployment
```

## Setup

```bash
cd ml-services
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /predict/categorize` - Payment categorization
- `POST /analyze/behavior` - Behavioral insights
- `POST /process/voice` - Voice analysis
- `POST /calculate/trust` - Trust scoring
- `GET /health` - Health check
