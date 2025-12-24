# Intermediate Deployment Strategies - Deep Dive

## Table of Contents
1. [Model Serving Patterns](#model-serving-patterns)
2. [REST API Design and Implementation](#rest-api-design-and-implementation)
3. [Containerization with Docker](#containerization-with-docker)
4. [CI/CD Pipelines for ML](#cicd-pipelines-for-ml)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Deployment Strategies](#deployment-strategies)
7. [Performance Optimization](#performance-optimization)
8. [Security Best Practices](#security-best-practices)

---

## Model Serving Patterns

### Pattern 1: Batch Prediction

**When to Use**:
- Non-time-critical predictions
- Large volumes of data
- Cost-sensitive (can use cheaper infrastructure off-hours)

**Architecture**:
```
Database → Extract Data → Batch Prediction Job → Write Results Back
```

**Implementation Example** (Python):
```python
import pandas as pd
import pickle
from datetime import datetime

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Batch prediction function
def batch_predict(batch_size=1000):
    # Query data
    df = pd.read_sql("SELECT * FROM customers WHERE predicted_at IS NULL LIMIT batch_size", conn)
    
    # Prepare features
    X = df[feature_columns]
    
    # Predict
    predictions = model.predict_proba(X)[:, 1]
    
    # Write back
    df['churn_probability'] = predictions
    df['predicted_at'] = datetime.now()
    df[['id', 'churn_probability', 'predicted_at']].to_sql('predictions', conn, if_exists='append')

# Schedule with cron or Airflow
if __name__ == "__main__":
    batch_predict()
```

**Scheduling**:
- **Cron**: Simple, works for single-machine
- **Airflow**: Complex workflows, dependencies, retries
- **Prefect**: Modern alternative to Airflow
- **Cloud schedulers**: AWS EventBridge, GCP Cloud Scheduler

**Pros**:
✅ Simple to implement  
✅ Cost-effective  
✅ Easy to monitor (single job)

**Cons**:
❌ High latency (hours/days)  
❌ Not suitable for real-time needs

### Pattern 2: Online (Real-Time) Serving

**When to Use**:
- User-facing features requiring immediate responses
- Fraud detection
- Recommendations during browsing

**Architecture**:
```
User Request → API Gateway → Load Balancer → Model Service → Response
```

**Latency Requirements**:
- User-facing: < 100ms
- Internal services: < 500ms
- Background tasks: < 5s

### Pattern 3: Streaming Predictions

**When to Use**:
- Continuous data streams (IoT sensors, clickstream)
- Real-time analytics
- Event-driven architectures

**Architecture**:
```
Event Stream (Kafka/Kinesis) → Stream Processor → Model Service → Output Stream
```

**Technologies**:
- **Kafka Streams**: Java/Scala processing
- **Flink**: Complex event processing
- **Spark Streaming**: Micro-batches
- **AWS Kinesis**: Managed streaming

**Example with Kafka + Python**:
```python
from kafka import KafkaConsumer, KafkaProducer
import json

consumer = KafkaConsumer('input_events', value_deserializer=lambda m: json.loads(m.decode('utf-8')))
producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'))

for message in consumer:
    event = message.value
    
    # Extract features
    features = extract_features(event)
    
    # Predict
    prediction = model.predict([features])[0]
    
    # Produce output
    producer.send('predictions', {
        'event_id': event['id'],
        'prediction': float(prediction),
        'timestamp': event['timestamp']
    })
```

---

## REST API Design and Implementation

### API Structure

**Well-designed ML API endpoints**:

```
POST /api/v1/predict              # Single prediction
POST /api/v1/batch-predict        # Batch predictions
GET  /api/v1/models               # List available models
GET  /api/v1/models/{id}          # Get model info
GET  /health                      # Health check
GET  /ready                       # Readiness check
GET  /metrics                     # Prometheus metrics
```

### FastAPI Implementation (Production-Ready)

**Complete Example**:

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, validator
from typing import List, Optional
import pickle
import numpy as np
import logging
from prometheus_client import Counter, Histogram, generate_latest
import time

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Prediction duration')
ERRORS = Counter('prediction_errors_total', 'Total errors')

# App
app = FastAPI(title="ML Model API", version="1.0.0")

# Load model (do this once, not per request!)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Request/Response models
class PredictionRequest(BaseModel):
    features: List[float]
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 10:  # Expected feature count
            raise ValueError('Must provide exactly 10 features')
        if any(np.isnan(v)):
            raise ValueError('Features cannot contain NaN')
        return v

class PredictionResponse(BaseModel):
    prediction: float
    probability: Optional[float] = None
    model_version: str = "1.0.0"

class BatchPredictionRequest(BaseModel):
    instances: List[List[float]]

# Authentication (simplified - use proper auth in production)
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":  # Use environment variable
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# Endpoints
@app.post("/api/v1/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(request: PredictionRequest):
    try:
        start_time = time.time()
        
        # Prepare input
        features = np.array([request.features])
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        PREDICTION_COUNT.inc()
        PREDICTION_DURATION.observe(time.time() - start_time)
        
        logger.info(f"Prediction: {prediction}, took {time.time()-start_time:.3f}s")
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=float(probability) if probability else None
        )
    
    except Exception as e:
        ERRORS.inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch-predict")
async def batch_predict(request: BatchPredictionRequest):
    predictions = model.predict(np.array(request.instances))
    return {"predictions": predictions.tolist()}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.get("/metrics")
async def metrics():
    return generate_latest()

# Startup/shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
```

**Run**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Input Validation

**Why Critical**:
- Prevent crashes from invalid inputs
- Security (injection attacks)
- Clear error messages

**Pydantic Validators**:
```python
from pydantic import BaseModel, validator, Field

class InputData(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age in years")
    income: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)
    
    @validator('age')
    def validate_age(cls, v):
        if v < 18:
            raise ValueError('Must be 18 or older')
        return v
```

### Error Handling

**Proper HTTP status codes**:
- `200`: Success
- `400`: Bad request (invalid input)
- `401`: Unauthorized
- `429`: Rate limit exceeded
- `500`: Server error
- `503`: Service unavailable

**Example**:
```python
from fastapi import HTTPException

try:
    prediction = model.predict(features)
except ValueError as e:
    raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise HTTPException(status_code=500, detail="Internal server error")
```

---

## Containerization with Docker

### Why Docker for ML?

**Reproducibility**: Same environment everywhere  
**Isolation**: Dependencies don't conflict  
**Portability**: Run anywhere (laptop, cloud, edge)  
**Scalability**: Easy to replicate

### Production-Ready Dockerfile

```dockerfile
#Multi-stage build for smaller image
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.9-slim

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Make sure scripts are executable and in PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key practices**:
✅ Multi-stage builds (smaller image)  
✅ Non-root user (security)  
✅ Health checks  
✅ Minimal base image (slim)  
✅ `.dockerignore` file

**.dockerignore**:
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.log
.git
.pytest_cache
.venv
venv/
notebooks/
tests/
```

### Docker Commands

**Build**:
```bash
docker build -t mymodel:v1.0.0 .
```

**Run locally**:
```bash
docker run -p 8000:8000 \
  -e API_KEY=secret \
  --name mymodel \
  mymodel:v1.0.0
```

**Push to registry**:
```bash
# Tag for registry
docker tag mymodel:v1.0.0 myregistry.azurecr.io/mymodel:v1.0.0

# Push
docker push myregistry.azurecr.io/mymodel:v1.0.0
```

### Docker Compose (Local Development)

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  model-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/model.pkl
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/models
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

**Run**:
```bash
docker-compose up
```

---

## CI/CD Pipelines for ML

### CI: Continuous Integration

**Goals**:
- Automated testing on every commit
- Code quality checks
- Fast feedback

**GitHub Actions Example** (`.github/workflows/ci.yml`):

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov flake8
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Run unit tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    -name: Test model performance
      run: |
        python tests/test_model_performance.py
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build-image:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ secrets.REGISTRY_URL }}
        username: ${{ secrets.REGISTRY_USERNAME }}
        password: ${{ secrets.REGISTRY_PASSWORD }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ${{ secrets.REGISTRY_URL }}/mymodel:latest
          ${{ secrets.REGISTRY_URL }}/mymodel:${{ github.sha }}
        cache-from: type=registry,ref=${{ secrets.REGISTRY_URL }}/mymodel:latest
        cache-to: type=inline
```

### CD: Continuous Deployment

**Deployment Pipeline** (`.github/workflows/deploy.yml`):

```yaml
name: Deploy to Production

on:
  workflow_run:
    workflows: ["CI Pipeline"]
    types:
      - completed
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Run model validation tests
      run: |
        python tests/validate_model.py
    
    - name: Deploy to staging
      run: |
        aws ecs update-service \
          --cluster ml-cluster \
          --service model-staging \
          --force-new-deployment
    
    - name: Wait for staging deployment
      run: |
        aws ecs wait services-stable \
          --cluster ml-cluster \
          --services model-staging
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ --staging
    
    - name: Deploy to production (canary)
      run: |
        # Deploy to 10% of traffic
        ./scripts/canary_deploy.sh --percentage 10
    
    - name: Monitor canary
      run: |
        # Monitor for 30 minutes
        ./scripts/monitor_canary.sh --duration 30m
    
    - name: Promote canary to full production
      if: success()
      run: |
        ./scripts/promote_canary.sh
    
    - name: Rollback on failure
      if: failure()
      run: |
        ./scripts/rollback.sh
```

### Model-Specific CI Tests

**tests/test_model_performance.py**:
```python
import pytest
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# Load test data
test_data = pd.read_csv('data/test.csv')
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def test_model_accuracy():
    """Ensure model meets minimum accuracy threshold"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.80, f"Model accuracy {accuracy:.3f} below threshold 0.80"

def test_model_auc():
    """Ensure AUC meets minimum threshold"""
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    assert auc > 0.85, f"Model AUC {auc:.3f} below threshold 0.85"

def test_inference_time():
    """Ensure predictions are fast enough"""
    import time
    start = time.time()
    model.predict(X_test[:1000])
    duration = time.time() - start
    avg_time = duration / 1000
    assert avg_time < 0.01, f"Average prediction time {avg_time:.4f}s exceeds 0.01s"

def test_no_nan_predictions():
    """Ensure no NaN in predictions"""
    predictions = model.predict(X_test)
    assert not pd.isna(predictions).any(), "Model produced NaN predictions"

def test_prediction_distribution():
    """Ensure predictions are reasonable"""
    predictions = model.predict(X_test)
    assert predictions.min() >= 0, "Negative predictions found"
    assert predictions.max() <= 1, "Predictions exceed 1.0"
```

---

## Monitoring and Logging

### Three Pillars of Observability

#### 1. Metrics (Prometheus + Grafana)

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'model-api'
    static_configs:
      - targets: ['model-api:8000']
```

**Custom metrics in code**:
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
PREDICTIONS = Counter('model_predictions_total', 'Total predictions', ['model_version'])
LAT ENCY = Histogram('model_prediction_duration_seconds', 'Prediction latency')
FEATURE_VALUES = Histogram('feature_value', 'Feature values', ['feature_name'])
MODEL_SCORE = Gauge('model_confidence_score', 'Average model confidence')

# Use in code
@LATENCY.time()
def predict(features):
    prediction = model.predict([features])
    PREDICTIONS.labels(model_version='1.0.0').inc()
    return prediction
```

**Grafana Dashboard**: Create visual dashboards for metrics

#### 2. Logging (Structured Logging)

**Best practice**: Structured logs (JSON) not plain text

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
        }
        # Add extra fields if present
        if hasattr(record, 'prediction'):
            log_data['prediction'] = record.prediction
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        
        return json.dumps(log_data)

# Configure
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Use
logger.info("Prediction made", extra={'prediction': 0.85, 'user_id': 123})
```

**Log aggregation**: ELK Stack (Elasticsearch, Logstash, Kibana) or Cloud solutions (CloudWatch, Stackdriver)

#### 3. Tracing (Distributed Tracing)

**For microservices**: Track request through multiple services

**OpenTelemetry example**:
```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

tracer = trace.get_tracer(__name__)

@app.post("/predict")
async def predict(request: PredictionRequest):
    with tracer.start_as_current_span("prediction"):
        with tracer.start_as_current_span("feature_engineering"):
            features = engineer_features(request)
        
        with tracer.start_as_current_span("model_inference"):
            prediction = model.predict(features)
        
        return prediction
```

### Alerting

**Define SLOs** (Service Level Objectives):
- Latency: p99 < 200ms
- Availability: 99.9% uptime
- Error rate: < 0.1%

**Prometheus Alerts** (`alerts.yml`):
```yaml
groups:
  - name: model_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, model_prediction_duration_seconds) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency"
      
      - alert: HighErrorRate
        expr: rate(prediction_errors_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
      
      - alert: ModelAccuracyDrop
        expr: model_accuracy < 0.75
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy dropped below threshold"
```

---

## Deployment Strategies

### Blue-Green Deployment

**Concept**: Two identical environments (Blue=current, Green=new)

**Process**:
1. Deploy new version to Green
2. Test Green thoroughly
3. Switch traffic from Blue to Green
4. Keep Blue as rollback option

**Benefits**: Zero downtime, instant rollback

**Kubernetes example**:
```yaml
# Service (points to either blue or green)
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model
    version: blue  # Change to 'green' to switch
  ports:
    - port: 80
      targetPort: 8000
```

### Canary Deployment

**Concept**: Gradually increase traffic to new version

**Process**:
1. Deploy new version
2. Route 5% traffic → monitor
3. Route 25% traffic → monitor
4. Route 50% traffic → monitor
5. Route 100% traffic (or rollback)

**Implementation** (AWS ALB):
```json
{
  "TargetGroups": [
    {"TargetGroupArn": "blue-tg", "Weight": 90},
    {"TargetGroupArn": "green-tg", "Weight": 10}
  ]
}
```

### A/B Testing

**Concept**: Run two model versions simultaneously to compare

**Route based on**:
- User ID hash (consistent experience per user)
- Random (true A/B)
- Geography, segment, etc.

**Track metrics** for each version, choose winner

---

## Performance Optimization

### Model Optimization

**1. Quantization**: Reduce precision (float32 → int8)
**2. Pruning**: Remove unimportant weights  
**3. Distillation**: Train smaller model from larger  
**4. ONNX**:Convert to optimized format

###Serving Optimization

**1. Batching**: Process multiple requests together
```python
# Dynamic batching with queue
from collections import deque
import asyncio

request_queue = deque()

async def batch_predict_worker():
    while True:
        if len(request_queue) >= BATCH_SIZE or time_since_last_batch() > MAX_WAIT:
            batch = [request_queue.popleft() for _ in range(min(len(request_queue), BATCH_SIZE))]
            predictions = model.predict(batch)
            # Return predictions
```

**2. Caching**: Cache frequent predictions
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def predict_cached(features_tuple):
    return model.predict([list(features_tuple)])[0]
```

**3. Model serving frameworks**: TensorFlow Serving, TorchServe (optimized)

---

## Security Best Practices

✅ **API Authentication**: API keys, OAuth 2.0, JWT  
✅ **Rate limiting**: Prevent abuse  
✅ **Input validation**: Prevent injection attacks  
✅ **HTTPS**: Encrypt data in transit  
✅ **Secrets management**: Never hardcode, use vaults  
✅ **Container scanning**: Check for vulnerabilities  
✅ **Least privilege**: Minimal permissions  
✅ **Audit logging**: Track all access

---

## Key Takeaways

✅ **Serving patterns**: Batch, online, streaming - choose based on latency needs

✅ **API design**: Versioning, validation, error handling, authentication

✅ **Docker**: Multi-stage builds, non-root user, health checks

✅ **CI/CD**: Automated testing (code, model performance, integration)

✅ **Monitoring**: Metrics (Prometheus), logs (structured), traces (OpenTelemetry)

✅ **Deployment**: Blue-green (zero downtime), canary (gradual), A/B (compare)

✅ **Optimization**: Quantization, batching, caching for performance

✅ **Security**: Authentication, rate limiting, encryption, secrets management

---

*Previous: [MLOps Basics](01-beginner-mlops-basics.md) | Next: [Advanced Production Systems](03-advanced-production-systems.md)*
