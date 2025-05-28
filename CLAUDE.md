# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a machine learning pipeline for fraudulent transactions detection built with Kedro framework and MLflow for experiment tracking. The project simulates credit card transaction data and implements ML models to detect fraudulent activities.

## Development Commands

### Docker Environment
```bash
# Start all services (MLflow server on port 5001, FastAPI on port 80, Kedro on ports 4142 & 8889)
docker-compose up

# Stop and remove containers
docker-compose down --rmi all

# Execute Kedro commands in Docker container
docker exec -it kedro-fraud-detection /bin/bash
cd projects/fraud-detection
kedro <command>
```

### Local Development
```bash
# Setup environment
conda create --name fraud_detection python=3.10 -y
conda activate fraud_detection
cd kedro/fraud_detection
pip install -r src/requirements.txt

# Run Kedro commands
kedro run                     # Run full pipeline
kedro viz --host=0.0.0.0     # Visualize pipeline (opens on localhost:4142)
kedro test                    # Run test suite
kedro lint                    # Run flake8, isort, and black
```

### Available Kedro Pipelines
- `kedro run --pipeline=etl` - ETL pipeline only
- `kedro run --pipeline=ml` - ML pipeline only
- `kedro run --pipeline=etl_generation` - Data generation only
- `kedro run --pipeline=etl_transformation` - Data transformation only
- `kedro run --pipeline=etl_preprocessing` - Data preprocessing only
- `kedro run --pipeline=etl_exploration` - Data exploration only
- `kedro run --pipeline=ml_training` - Model training and prediction
- `kedro run --pipeline=ml_inference` - Model evaluation and inference

## Architecture

### Project Structure
The Kedro project follows the standard data science pipeline architecture:

1. **ETL Pipeline** (`fraud_detection/pipelines/etl_app/`):
   - Data generation: Simulates customers, terminals, and transactions
   - Data transformation: Feature engineering and aggregation
   - Data preprocessing: Cleaning and preparation for ML
   - Data exploration: Analysis and visualization

2. **ML Pipeline** (`fraud_detection/pipelines/ml_app/`):
   - Model training: Trains fraud detection models
   - Model evaluation: Performance metrics and validation
   - Model inference: Predictions on new data

### Data Layer Structure
The project uses Kedro's data catalog with organized data layers:
- `01_raw/` - Raw transaction data
- `02_intermediate/` - Processed intermediate data
- `03_primary/` - Primary datasets
- `04_feature/` - Feature engineered data
- `05_model_input/` - ML-ready datasets
- `06_models/` - Trained models
- `07_model_output/` - Model predictions
- `08_reporting/` - Reports and visualizations

### Key Components
- **Pipeline Registry** (`src/fraud_detection/pipeline_registry.py`): Defines and registers all available pipelines with specific tags for modular execution
- **Node Structure**: Each pipeline contains nodes organized in `/nodes/` directories for specific data processing tasks
- **MLflow Integration**: Experiment tracking and model versioning through kedro-mlflow plugin

### Fraud Detection Approach
The system implements four fraud scenarios:
1. Amount-based fraud (transactions > 220)
2. Terminal compromise (compromised terminals over 28 days)
3. Customer credential theft (modified transaction amounts over 14 days)
4. Velocity-based fraud (10+ transactions within 2-hour window, simulating account takeover)

## Configuration
- Kedro configuration files in `conf/base/` and `conf/local/`
- MLflow tracking configuration in `conf/local/mlflow.yml`
- Environment variables managed through `.env` file for Docker setup