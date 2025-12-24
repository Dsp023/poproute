# MLOps Basics - Beginner Guide

## Table of Contents
1. [What is MLOps?](#what-is-mlops)
2. [The Problem MLOps Solves](#the-problem-mlops-solves)
3. [The ML Lifecycle](#the-ml-lifecycle)
4. [Core MLOps Principles](#core-mlops-principles)
5. [Essential Components](#essential-components)
6. [MLOps Maturity Levels](#mlops-maturity-levels)
7. [Getting Started](#getting-started)
8. [Tools Ecosystem](#tools-ecosystem)
9. [Best Practices](#best-practices)
10. [Common Pitfalls](#common-pitfalls)

---

## What is MLOps?

**MLOps** = **Machine Learning** + **Operations** + **DevOps practices**

**Formal Definition**: Set of practices, tools, and cultural philosophies for deploying, monitoring, and maintaining machine learning models in production environments reliably, efficiently, and at scale.

**Simple Analogy**: 
- **DevOps**: Automating software deployment (code → production)
- **MLOps**: Automating ML deployment (data + model → production)

**Key Difference from Traditional Software**:
| Aspect | Traditional Software | ML Systems |
|--------|---------------------|------------|
| **Code** | Primary asset | One of three assets |
| **Data** | Input/output | Critical asset (changes over time) |
| **Model** | N/A | Third critical asset |
| **Testing** | Deterministic | Probabilistic |
| **Degradation** | Breaks | Silently degrades |

---

## The Problem MLOps Solves

### The ML Production Gap

**Research Reality**:
- Jupyter notebooks
- Experimental code
- Clean academic datasets
- Single-machine training
- "It works on my machine!"

**Production Requirements**:
- Scalable, maintainable code
- Messy real-world data
- Distributed systems
- 24/7 reliability
- Performance monitoring

**The Gap**: 87% of ML projects never make it to production (VentureBeat, 2019)

### Why Models Fail in Production

**1. Technical Debt**:
- Quick experiments become production code
- Copy-pasted notebooks
- Hardcoded values
- No testing

**2. data Issues**:
- Training/serving skew (different data in production)
- Data drift (distributions change over time)
- Missing values
- Schema changes

**3. Model Decay**:
- World changes, model doesn't
- Performance degrades silently
- No alerts until business impact

**4. Lack of Monitoring**:
- Model running ≠ model working
- Can't debug black box
- No visibility into predictions

**MLOps addresses all of these!**

---

## The ML Lifecycle

### Traditional View (Incomplete)
```
Data → Train Model → Deploy → Done ❌
```

### MLOps View (Reality)
```
┌─────────────────────────────────────────────┐
│                                             │
↓                                             │
Data Collection ────→ Data Validation         │
       ↓                                      │
Feature Engineering                           │
       ↓                                      │
Model Training ──→ Model Evaluation           │
       ↓                                      │
Model Validation (testing)                    │
       ↓                                      │
Model Registry (versioning)                   │
       ↓                                      │
Deployment (serving)                          │
       ↓                                      │
Monitoring (performance, data drift)          │
       ↓                                      │
   Retrain (when performance degrades) ───────┘
```

### Detailed Lifecycle Stages

#### 1. Data Management
**Activities**:
- Data collection (APIs, databases, streams)
- Data validation (schema, quality checks)
- Data versioning (DVC, lakeFS)
- Feature engineering
- Feature store (centralized features)

**Tools**: DVC, lakeFS, Feast, Tecton

#### 2. Model Development  
**Activities**:
- Experiment tracking
- Hyperparameter tuning
- Model selection
- Version control (code,models, data)

**Tools**: MLflow, W&B, Neptune, Git

#### 3. Model Validation
**Activities**:
- Offline evaluation (test set)
- Model comparison
- Business metric validation
- Model documentation

**Output**: Model ready for staging

#### 4. Deployment
**Activities**:
- Model packaging (containers)
- Infrastructure provisioning
- API creation
- Load balancing
- Blue-green/canary deployment

**Tools**: Docker, Kubernetes, FastAPI, BentoML

#### 5. Monitoring & Maintenance
**Activities**:
- Performance monitoring
- Data drift detection
- Logging and alerting  
- Model retraining triggers
- Incident response

**Tools**: Prometheus, Grafana, Evidently AI, Arize

---

## Core MLOps Principles

### 1. Version Everything

**Code**: Git (obviously)

**Data**: 
- Raw data snapshots
- Processed data versions
- Data lineage tracking

**Models**:
- Model files (.pkl, .h5, .pt)
- Training code commit hash
- Hyperparameters
- Metrics

**Why**: Reproducibility, debugging, rollback

### 2. Automate Everything

**Manual processes = errors + slow**

**Automate**:
- Data validation
- Model training (on schedule or trigger)
- Model testing
- Deployment
- Monitoring

**Benefits**: Speed, consistency, reliability

### 3. Test Thoroughly

**Unit Tests**: Individual functions

**Integration Tests**: Components together

**Model Tests**:
- Performance benchmarks
- Bias/fairness checks
- Inference time
- Resource usage

**Data Tests**:
- Schema validation
- Distribution checks
- Missing value limits

### 4. Monitor Continuously

**System Metrics**:
- Latency (p50, p95, p99)
- Throughput (requests/sec)
- Error rates
- Resource utilization (CPU, memory, GPU)

**Model Metrics**:
- Prediction accuracy (on live data)
- Confidence scores distribution
- Feature importance drift

**Data Metrics**:
- Input distribution changes
- Missing value rates
- Out-of-range values

### 5. Fail Gracefully

**Handle errors**:
- Invalid inputs → meaningful error messages
- Model failures → fallback to simpler model or cached predictions
- High latency → timeouts with default responses

**Circuit breakers**: Stop calling failing dependencies

---

## Essential Components

### 1. Experiment Tracking

**Purpose**: Record every experiment for reproducibility

**Track**:
- Hyperparameters (learning rate, batch size, etc.)
- Metrics (accuracy, loss, F1, etc.)
- Artifacts (model files, plots, logs)
- Environment (library versions, hardware)
- Code version (Git commit)

**Example with MLflow**:
```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 10)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

**Tools**:
- **MLflow**: Open-source, comprehensive
- **Weights & Biases**: Collaborative, great UI
- **Neptune**: Enterprise features
- **TensorBoard**: TensorFlow/PyTorch visualization

**Benefits**:
- Compare experiments easily
- Reproduce results months later
- Share with team
- Track progress over time

### 2. Model Registry

**Purpose**: Central repository for productionmodels

**Capabilities**:
- Store model files
- Version models (v1, v2, v3...)
- Stage models (dev, staging, production)
- Track lineage (training data, code, experiments)
- Access control

**Workflow**:
```
Experiment → Register Model → Staging → Validation → Production
```

**Example with MLflow**:
```python
# Register model from experiment
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="customer_churn_predictor"
)

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="customer_churn_predictor",
    version=3,
    stage="Production"
)
```

**Tools**: MLflow Model Registry, Azure ML Model Registry, SageMaker Model Registry

### 3. Feature Store

**Purpose**: Centralized feature management

**Problems it Solves**:
- Training/serving skew (different feature calculations)
- Feature reuse across teams
- Feature discovery
- Consistency

**Features**:
- Offline store (historical features for training)
- Online store (low-latency for serving)
- Feature versioning
- Point-in-time correctness (historical feature values)

**Example**:
```python
# Define features
@feast.feature_definition
def customer_features():
    return FeatureView(
        name="customer_stats",
        entities=["customer_id"],
        features=[
            Feature("total_purchases", ValueType.INT64),
            Feature("avg_order_value", ValueType.FLOAT),
        ]
    )

# Get features for training
features = feast.get_historical_features(
    entity_df=customer_df,
    features=["customer_stats:total_purchases"]
)

# Get features for serving
online_features = feast.get_online_features(
    features=["customer_stats:total_purchases"],
    entity_rows=[{"customer_id": 123}]
)
```

**Tools**: Feast, Tecton, AWS SageMaker Feature Store

### 4. Model Serving Infrastructure

**Purpose**: Make model accessible for predictions

**Patterns**:

**Batch Serving**:
- Process large dataset at once
- Schedule (hourly, daily)
- Example: Email recommendations computed nightly

**Online Serving**:
- Real-time predictions via API
- Low latency required (<100ms)
- Example: Fraud detection during transaction

**Streaming**:
- Process events as they arrive
- Example: Real-time personalization

**Tools**:
- **FastAPI/Flask**: Simple Python APIs
- **TensorFlow Serving**: TensorFlow models, highly optimized
- **TorchServe**: PyTorch official serving
- **BentoML**: Framework-agnostic, feature-rich
- **Seldon Core**: Kubernetes-native
- **KServe**: Kubernetes standard for ML serving

### 5. CI/CD Pipelines

**Continuous Integration**:
```
Code Push → Run Tests → Build Container → Push to Registry
```

**Continuous Deployment**:
```
New Model → Validation Tests → Deploy to Staging 
→ Integration Tests → Deploy to Production (gradually)
```

**GitHub Actions Example**:
```yaml
name: ML CI/CD

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: pytest tests/
      - name: Run model tests
        run: python test_model.py
      
  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: docker build -t mymodel:latest .
      - name: Push to registry
        run: docker push mymodel:latest
      - name: Deploy to production
        run: kubectl apply -f deployment.yaml
```

### 6. Monitoring & Observability

**Three Pillars**:

**Logs**: Detailed event records
```
2024-12-24 10:15:32 - Prediction request received
2024-12-24 10:15:33 - Input: {...}
2024-12-24 10:15:34 - Prediction: 0.85
```

**Metrics**: Numerical measurements over time
- Latency: p50, p95, p99
- Throughput: requests/sec
- Error rate: %
- Model accuracy: recent predictions vs actuals

**Traces**: Request flow through system
```
API Gateway (10ms) → Load Balancer (5ms) → Model Service (45ms) 
→ Database (30ms) → Response (2ms)
Total: 92ms
```

**Dashboards**: Visualize metrics (Grafana, Kibana)

**Alerts**: Notify on anomalies
- Latency > 500ms
- Error rate > 1%
- Accuracy drop > 5%
- Data drift detected

---

## MLOps Maturity Levels

### Level 0: Manual Process
- Jupyter notebooks in production
- Manual testing
- No versioning
- No monitoring
- **One-off experiments**

### Level 1: ML Pipeline Automation
- Automated training pipeline
- Experiment tracking
- Model versioning
- Basic monitoring
- **Continuous training**

### Level 2: CI/CD Pipeline Automation
- Automated testing (data, model, integration)
- Automated deployment
- Rollback capability
- Comprehensive monitoring
- **Continuous delivery**

### Level 3: Full MLOps Automation
- Automated feature engineering
- Automated model selection
- Automated retraining triggers
- Advanced monitoring (drift detection)
- Self-healing systems
- **Autonomous ML systems**

**Most organizations**: Level 0-1  
**Goal**: Level 2 minimum for production

---

## Getting Started

### Step 1: Learn Version Control (Git)

**Essential commands**:
```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

**Branch strategy**: GitFlow or GitHub Flow

**Resources**: GitHub Learning Lab, Atlassian Git Tutorial

### Step 2: Containerize Your Model (Docker)

**Why Docker?**
- Packages code + dependencies
- "Works on my machine" → "Works everywhere"
- Easy deployment

**Simple Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl .
COPY app.py .

CMD ["python", "app.py"]
```

**Build & Run**:
```bash
docker build -t my-model .
docker run -p 5000:5000 my-model
```

**Resources**: Docker docs, Play with Docker

### Step 3: Create Simple API (FastAPI)

**Why FastAPI?**
- Fast (high performance)
- Easy to use
- Auto-generated docs
- Type hints

**Example**:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.post("/predict")
def predict(request: PredictionRequest):
    features = [[request.feature1, request.feature2, request.feature3]]
    prediction = model.predict(features)[0]
    return {"prediction": float(prediction)}
```

**Run**:
```bash
uvicorn app:app --reload
```

**Test**: Visit `http://localhost:8000/docs` for interactive API docs

### Step 4: Track Experiments (MLflow)

**Install**:
```bash
pip install mlflow
```

**Track experiment**:
```python
import mlflow

mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    # ... train model ...
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

**View UI**:
```bash
mlflow ui
```

Visit `http://localhost:5000`

### Step 5: Deploy to Cloud

**Options**:

**Simplest**: Hugging Face Spaces, Streamlit Cloud (free, easy)

**More control**: AWS, GCP, Azure
- AWS: SageMaker, Lambda + API Gateway
- GCP: Vertex AI, Cloud Functions
- Azure: Azure ML, Functions

**Start simple, scale as needed**

---

## Tools Ecosystem

### By Category

**Experiment Tracking**:
- MLflow (open source, comprehensive)
- Weights & Biases (team collaboration)
- Neptune.ai (enterprise)
- TensorBoard (TensorFlow/PyTorch)

**Model Registry**:
- MLflow Model Registry
- W&B Artifacts
- Azure ML Model Registry

**Data Versioning**:
- DVC (Git for data)
- lakeFS (data lake versioning)
- Pachyderm

**Feature Store**:
- Feast (open source)
- Tecton (managed)
- Hopsworks

**Orchestration**:
- Airflow (general workflow)
- Kubeflow (ML pipelines on Kubernetes)
- Prefect (modern, Python-first)
- Metaflow(Netflix)

**Model Serving**:
- FastAPI (simple, Python)
- BentoML (production-ready)
- TensorFlow Serving (TF optimization)
- TorchServe (PyTorch official)
- Seldon Core / KServe (Kubernetes)

**Monitoring**:
- Prometheus + Grafana (metrics)
- Evidently AI (ML monitoring)
- WhyLabs (data quality)
- Arize (model performance)
- Fiddler

**Cloud Platforms**:
- AWS SageMaker (end-to-end)
- Google Vertex AI (GCP)
- Azure Machine Learning (Azure)

### Tool Selection Guide

**Starting out**: MLflow + FastAPI + Docker
**Small team**: MLflow + W&B + GitHub Actions
**Enterprise**: Cloud platform (SageMaker / Vertex AI / Azure ML)

---

## Best Practices

### 1. Start Simple, Iterate

❌ Don't: Build complex MLOps infrastructure before first model
✅ Do: Deploy first model simply, add MLOps incrementally

### 2. Automate Gradually

**Priority order**:
1. Version control (Git) - Day 1
2. Experiment tracking - Week 1
3. Containerization - Month 1
4. CI/CD - Month 2-3
5. Advanced monitoring - As needed

### 3. Document Everything

**Model Cards**: Document model purpose, training data, limitations

**Example**:
```markdown
# Customer Churn Predictor v2.1

**Purpose**: Predict probability of customer churn

**Training Data**: 100K customers, 2020-2023

**Features**: demographic, usage patterns, support tickets

**Performance**: AUC=0.85 on test set

**Limitations**:
- Lower accuracy for customers <3 months
- Biased towards high-value customers

**Intended Use**: Monthly batch predictions

**Not for**: Real-time decisions
```

### 4. Test at Multiple Levels

**Data Tests**:
```python
def test_data_schema():
    assert set(df.columns) == expected_columns

def test_no_missing_values():
    assert df.isna().sum().sum() == 0

def test_value_ranges():
    assert df['age'].between(18, 100).all()
```

**Model Tests**:
```python
def test_model_accuracy():
    assert model.score(X_test, y_test) > 0.8

def test_inference_time():
    start = time.time()
    model.predict(X_sample)
    assert time.time() - start < 0.1  # < 100ms
```

### 5. Monitor What Matters

**System**: latency, throughput, errors
**Model**: accuracy on recent data
**Data**: distribution shifts
**Business**: revenue impact, user engagement

**Set Alerts**: on degradation, not just failures

### 6. Version Everything

**Git**: Code
**DVC**: Data
**MLflow**: Models
**Docker tags**: Containers

**Benefits**: Reproducibility, rollback, debugging

### 7. Security & Compliance

- **Authentication**: API keys, OAuth
- **Encryption**: Data at rest and in transit
- **Access control**: Who can deploy models?
- **Audit logs**: Who did what when?
- **Compliance**: GDPR, HIPAA, data privacy

---

## Common Pitfalls

### 1. Notebook Code in Production

❌ **Problem**: Jupyter notebooks directly in production
- Not modular
- Hard to test
- Not version-controlled properly

✅ **Solution**: Refactor to Python modules, use notebooks for exploration only

### 2.  Model Without Monitoring

❌ **Problem**: Deploy and forget
- Model degrades silently
- Discover issues when business complains

✅ **Solution**: Monitor from day 1, even basic metrics

### 3. Training/Serving Skew

❌ **Problem**: Different feature logic in training vs serving
```python
# Training
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 65, 100])

# Serving (different logic!)
if age < 18:
    age_group = 'young'
elif age < 65:
    age_group = 'adult'
```

✅ **Solution**: Share feature engineering code or use feature store

### 4. No Model Versioning

❌ **Problem**: Can't reproduce or rollback

✅ **Solution**: Tag every model with version and metadata

### 5. Over-Engineering Early

❌ **Problem**: Building Kubernetes cluster before first model

✅ **Solution**: Start simple (FastAPI on single server), scale when needed

### 6. Ignoring Data Quality

❌ **Problem**: Assume data is clean

✅ **Solution**: Validate data at ingestion, monitor distributions

### 7. Manual Deployment

❌ **Problem**: SSH into server, copy files, restart service

✅ **Solution**: Automated deployment via CI/CD

---

## Key Takeaways

✅ **MLOps** automates ML lifecycle from development to production to monitoring

✅ **Core principles**: Version everything, automate everything, test thoroughly, monitor continuously

✅ **Essential components**: Experiment tracking, model registry, feature store, serving, monitoring  

✅ **Maturity levels**: Manual (0) → Automated pipelines (1) → CI/CD (2) → Full automation (3)

✅ **Tools**: MLflow, Docker, FastAPI, cloud platforms - start simple

✅ **Best practices**: Start simple, iterate, document, test multiple levels

✅ **Common pitfalls**: Notebooks in prod, no monitoring, training/serving skew

✅ **Goal**: Reliable, scalable, reproducible ML systems in production

---

## What's Next?

Ready to dive deeper? Continue to:

- **[Intermediate Deployment Strategies](02-intermediate-deployment-strategies.md)** - Model serving, APIs, CI/CD
- **[Advanced Production Systems](03-advanced-production-systems.md)** - Scaling, monitoring, A/B testing
- **Practice**: Deploy a simple model end-to-end with MLflow + Docker + FastAPI

---

## Additional Resources

**Books**:
- "Designing Machine Learning Systems" - Chip Huyen
- "Machine Learning Engineering" - Andriy Burkov
- "Building Machine Learning Powered Applications" - Emmanuel Ameisen

**Courses**:
- "Machine Learning Engineering for Production" (DeepLearning.AI)
- "Full Stack Deep Learning"  
- "Made With ML" (MLOps course)

**Blogs**:
- Chip Huyen's blog
- Eugene Yan's blog
- Netflix Tech Blog (ML Platform)

---

*Next: [Intermediate Deployment Strategies](02-intermediate-deployment-strategies.md)*

