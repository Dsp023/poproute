# Advanced Production Systems - Comprehensive Guide

## Table of Contents
1. [Scaling ML Systems](#scaling-ml-systems)
2. [Advanced Model Performance Monitoring](#advanced-model-performance-monitoring)
3. [A/B Testing Framework](#ab-testing-framework)
4. [Feature Stores in Production](#feature-stores-in-production)
5. [ML Platform Architecture](#ml-platform-architecture)
6. [Cost Optimization](#cost-optimization)
7. [Incident Response](#incident-response)
8. [Production Best Practices](#production-best-practices)

---

## Scaling ML Systems

### Horizontal Scaling (Multiple Replicas)

**Kubernetes Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-deployment
spec:
  replicas: 5  # Start with 5 replicas
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: model-server
        image: myregistry.io/model:v1.2.0
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Auto-Scaling

**Horizontal Pod Autoscaler (HPA)**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 min cooldown
      policies:
      - type: Percent
        value: 50  # Scale down max 50% at a time
        periodSeconds: 60
```

### Load Balancing

**Application Load Balancer** (AWS):
- Distributes requests across replicas
- Health checks
- Sticky sessions (if needed)
- SSL termination

**Service Mesh** (Istio):
- Advanced routing (canary, A/B)
- Circuit breaking
- Retry logic
- Distributed tracing

---

## Advanced Model Performance Monitoring

### Data Drift Detection

**Statistical Tests**:

```python
from scipy import stats
import numpy as np

class DriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold
    
    def detect_drift(self, current_data, feature_name):
        """Kolmogorov-Smirnov test for distribution shift"""
        statistic, p_value = stats.ks_2samp(
            self.reference_data[feature_name],
            current_data[feature_name]
        )
        
        is_drift = p_value < self.threshold
        
        return {
            'feature': feature_name,
            'p_value': p_value,
            'statistic': statistic,
            'drift_detected': is_drift
        }
    
    def detect_all_features(self, current_data):
        results = []
        for feature in self.reference_data.columns:
            result = self.detect_drift(current_data, feature)
            results.append(result)
        
        return results

# Usage
detector = DriftDetector(train_data)
drift_results = detector.detect_all_features(production_data)

# Alert if drift detected
for result in drift_results:
    if result['drift_detected']:
        alert(f"Drift detected in {result['feature']}")
```

**Population Stability Index (PSI)**:
```python
def calculate_psi(expected, actual, buckets=10):
    """
    Calculate PSI (Population Stability Index)
    PSI < 0.1: No significant change
    0.1 < PSI < 0.2: Small change
    PSI > 0.2: Significant change
    """
    # Create bins based on expected distribution
    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    
    # Calculate percentages in each bucket
    expected_perc = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_perc = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)
    
    # Calculate PSI
    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    
    return psi

# Monitor weekly
psi_score = calculate_psi(train_income, this_week_income)
if psi_score > 0.2:
    alert("Significant distribution shift detected")
```

### Concept Drift Monitoring

**Track model performance over time**:

```python
from collections import deque
import pandas as pd

class PerformanceMonitor:
    def __init__(self, window_size=1000, alert_threshold=0.05):
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.baseline_accuracy = None
        self.alert_threshold = alert_threshold
    
    def log_prediction(self, prediction, actual=None):
        self.predictions.append(prediction)
        if actual is not None:
            self.actuals.append(actual)
    
    def calculate_current_accuracy(self):
        if len(self.actuals) < 100:  # Need minimum samples
            return None
        
        correct = sum([p == a for p, a in zip(self.predictions, self.actuals)])
        return correct / len(self.actuals)
    
    def check_degradation(self):
        current_acc = self.calculate_current_accuracy()
        if current_acc is None or self.baseline_accuracy is None:
            return False
        
        drop = self.baseline_accuracy - current_acc
        if drop > self.alert_threshold:
            alert(f"Model accuracy dropped by {drop:.2%}")
            return True
        return False
    
    def set_baseline(self):
        self.baseline_accuracy = self.calculate_current_accuracy()

# Usage
monitor = PerformanceMonitor()
monitor.set_baseline()  # After initial deployment

# In production
prediction = model.predict(features)
monitor.log_prediction(prediction)

# When ground truth arrives (async)
monitor.log_prediction(prediction, actual=ground_truth)
monitor.check_degradation()
```

### Feature Importance Drift

**Track if model relies on different features over time**:

```python
import shap

class FeatureImportanceMonitor:
    def __init__(self, model, baseline_data):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.baseline_importance = self._calculate_ importance(baseline_data)
    
    def _calculate_importance(self, data):
        shap_values = self.explainer.shap_values(data)
        # Average absolute SHAP values
        return np.abs(shap_values).mean(axis=0)
    
   def detect_importance_shift(self, current_data):
        current_importance = self._calculate_importance(current_data)
        
        # Correlation between baseline and current
        correlation = np.corrcoef(self.baseline_importance, current_importance)[0, 1]
        
        if correlation < 0.8:  # Threshold
            alert("Feature importance pattern has shifted significantly")
        
        return {
            'correlation': correlation,
            'baseline': self.baseline_importance,
            'current': current_importance
        }
```

---

## A/B Testing Framework

### Experiment Design

**Key Decisions**:
1. **Metric**: What to optimize (accuracy, latency, revenue)?
2. **Sample size**: How much data needed for statistical significance?
3. **Duration**: How long to run?
4. **Allocation**: What % to each variant?

**Sample Size Calculation**:
```python
from statsmodels.stats.power import zt_ind_solve_power

def calculate_sample_size(baseline_rate, mde, power=0.8, alpha=0.05):
    """
    baseline_rate: Current conversion/accuracy rate
    mde: Minimum detectable effect (e.g., 0.02 for 2% improvement)
    power: Statistical power (1 - false negative rate)
    alpha: Significance level (false positive rate)
    """
    effect_size = mde / baseline_rate
    
    n = zt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,  # Equal allocation
        alternative='two-sided'
    )
    
    return int(np.ceil(n))

# Example: Need 2% improvement over 75% baseline
sample_size = calculate_sample_size(baseline_rate=0.75, mde=0.02)
print(f"Need {sample_size} samples per variant")
```

### Traffic Routing

**Hash-based routing** (consistent per user):

```python
import hashlib

def route_to_variant(user_id, variants=['control', 'treatment'], weights=[0.5, 0.5]):
    """
    Consistent hash-based routing
    Same user always gets same variant
    """
    # Hash user ID
    hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
    
    # Normalize to [0, 1]
    normalized = (hash_val % 10000) / 10000
    
    # Route based on weights
    cumulative = 0
    for variant, weight in zip(variants, weights):
        cumulative += weight
        if normalized < cumulative:
            return variant
    
    return variants[-1]

# Usage
user_variant = route_to_variant(user_id=12345)
if user_variant == 'control':
    prediction = model_v1.predict(features)
else:
    prediction = model_v2.predict(features)
```

### Statistical Analysis

**Sequential Testing** (monitor continuously):

```python
from scipy import stats

class ABTest:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.control_successes = []
        self.treatment_successes = []
    
    def add_observation(self, variant, success):
        if variant == 'control':
            self.control_successes.append(success)
        else:
            self.treatment_successes.append(success)
    
    def calculate_result(self):
        n_control = len(self.control_successes)
        n_treatment = len(self.treatment_successes)
        
        success_control = sum(self.control_successes)
        success_treatment = sum(self.treatment_successes)
        
        rate_control = success_control / n_control
        rate_treatment = success_treatment / n_treatment
        
        # Chi-squared test
        contingency_table = [
            [success_control, n_control - success_control],
            [success_treatment, n_treatment - success_treatment]
        ]
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Calculate relative improvement
        relative_improvement = (rate_treatment - rate_control) / rate_control
        
        return {
            'control_rate': rate_control,
            'treatment_rate': rate_treatment,
            'relative_improvement': relative_improvement,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'n_control': n_control,
            'n_treatment': n_treatment
        }
    
    def get_recommendation(self):
        result = self.calculate_result()
        
        if not result['significant']:
            return "CONTINUE", "Not enough data for statistical significance"
        
        if result['relative_improvement'] > 0:
            return "DEPLOY TREATMENT", f"{result['relative_improvement']:.1%} improvement"
        else:
            return "KEEP CONTROL", f"Treatment {result['relative_improvement']:.1%} worse"

# Usage
test = ABTest()
# Log observations
test.add_observation('control', success=True)
test.add_observation('treatment', success=True)
# ... more observations ...

# Check results
result = test.calculate_result()
recommendation, reason = test.get_recommendation()
```

---

## Feature Stores in Production

### Feast Implementation

**Feature definitions**:

```python
from feast import FeatureView, Entity, Field, FeatureStore
from feast.types import Float32, Int64, String
from datetime import timedelta

# Define entity
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="Customer entity"
)

# Define feature view
customer_features = FeatureView(
    name="customer_statistics",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="total_purchases", dtype=Int64),
        Field(name="avg_order_value", dtype=Float32),
        Field(name="days_since_last_purchase", dtype=Int64),
        Field(name="favorite_category", dtype=String),
    ],
    source=BigQuerySource(
        timestamp_field="event_timestamp",
        table="project.dataset.customer_stats"
    )
)
```

**Online serving** (low latency):

```python
from feast import FeatureStore

fs = FeatureStore(repo_path=".")

# Get features for real-time prediction
features = fs.get_online_features(
    features=[
        "customer_statistics:total_purchases",
        "customer_statistics:avg_order_value",
        "customer_statistics:days_since_last_purchase",
    ],
    entity_rows=[
        {"customer_id": 1001},
        {"customer_id": 1002},
    ]
).to_dict()

# Use in model
predictions = model.predict(features)
```

**Training data retrieval** (point-in-time correct):

```python
# Get historical features for training
training_data = fs.get_historical_features(
    entity_df=entity_df,  # DataFrame with customer_id and timestamps
    features=[
        "customer_statistics:total_purchases",
        "customer_statistics:avg_order_value",
    ]
).to_df()

# Train model
X = training_data.drop('target', axis=1)
y = training_data['target']
model.fit(X, y)
```

### Training/Serving Skew Prevention

**Shared feature logic**:

```python
# feature_engineering.py (used in both training and serving)

def engineer_features(raw_data):
    """
    Shared feature engineering logic
    Used in both training pipeline and online serving
    """
    features = {}
    
    # Age group
    features['age_group'] = pd.cut(
        raw_data['age'],
        bins=[0, 18, 35, 50, 65, 100],
        labels=['<18', '18-35', '35-50', '50-65', '65+']
    )
    
    # Income bracket
    features['income_bracket'] = pd.qcut(
        raw_data['income'],
        q=5,
        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    )
    
    # Interaction features
    features['income_per_age'] = raw_data['income'] / raw_data['age']
    
    return features

# Training pipeline
train_features = engineer_features(train_raw_data)

# Serving
@app.post("/predict")
def predict(request):
    features = engineer_features(request.data)
    return model.predict(features)
```

---

## ML Platform Architecture

### Platform Components

```
┌─────────────────────────────────────────────────────────┐
│                   ML Platform                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Data Layer  │  │ Compute Layer│  │ Serving Layer│ │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤ │
│  │ Data Lake    │  │ Training     │  │ Online       │ │
│  │ Feature Store│  │ Jobs         │  │ Serving      │ │
│  │ Metadata DB  │  │ HPO          │  │ Batch        │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Orchestration Layer                     │  │
│  │  (Airflow / Kubeflow / Metaflow)                 │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │        Monitoring & Governance Layer              │  │
│  │  (Metrics, Logging, Alerting, Model Registry)    │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Self-Service ML Platform

**Goals**:
- Data scientists focus on modeling, not infrastructure
- Standardized workflows
- Governance and compliance built-in

**Example Platform API**:

```python
from ml_platform import Platform

platform = Platform()

# Register dataset
dataset = platform.create_dataset(
    name="customer_churn",
    source="s3://data/churn.parquet",
    schema=schema
)

# Create experiment
experiment = platform.create_experiment(
    name="churn_prediction_xgboost",
    dataset=dataset,
    model_type="classification"
)

# Train model
model = experiment.train(
    algorithm="xgboost",
    hyperparameters={
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100
    },
    compute="gpu-large"  # Platform handles provisioning
)

# Evaluate
metrics = model.evaluate(test_dataset)
print(f"AUC: {metrics['auc']}")

# Deploy to staging
deployment = model.deploy(
    environment="staging",
    replicas=2,
    autoscaling=True
)

# After validation, promote to production
deployment.promote_to_production()
```

---

## Cost Optimization

### Model Optimization

**Quantization** (reduce model size):
```python
import tensorflow as tf

# Post-training quantization
converter = tf.lite.TFLiteConverter.from_saved_model('model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 4x smaller, faster inference
```

**Distillation** (train smaller model from larger):
```python
# Teacher model (large, accurate)
teacher_predictions = large_model.predict(X_train)

# Student model (small,  fast)
student_model = SmallModel()
student_model.fit(
    X_train,
    teacher_predictions,  # Learn from teacher
    epochs=50
)

# Deploy student (4x faster, 10x smaller)
```

### Infrastructure Cost Optimization

**Right-sizing**:
- Profile memory/CPU usage
- Use smallest instance that meets requirements
- Spot instances for training (70% cheaper)

**Auto-scaling**:
- Scale to zero during off-hours
- Scale up during peak

**Batch optimization**:
- Larger batches = fewer API calls
- Schedule batch jobs during off-peak (cheaper compute)

**Caching**:
- Cache frequent predictions
- Reduces compute cost

**Resource Example**:
```
Original: 5x c5.2xlarge instances 24/7
- Cost: ~$1,500/month

Optimized: 
- 2x c5.large instances (right-sized)
- Auto-scale 2-10 based on traffic
- Spot instances for training
- Cost: ~$400/month
- Savings: 73%
```

---

## Incident Response

### Runbook Example

**Model Serving Incident**:

```markdown
# Runbook: Model API High Error Rate

## Alert Trigger
Error rate > 1% for 5 minutes

## Severity
Critical (impacting users)

## Investigation Steps

1. Check Grafana dashboard: link-to-dashboard
   - What's the error rate trend?
   - Which endpoints affected?
   - Geographic distribution?

2. Check recent deployments
   ```
   kubectl rollout history deployment/model-api
   ```
   - Was there a recent deploy?

3. Check logs
   ```
   kubectl logs -l app=model-api --tail=100
   ```
   - What kind of errors?
   - Input validation? Model errors?

4. Check upstream services
   - Database healthy?
   - Feature store responding?

## Mitigation Steps

### If recent deployment:
```bash
# Rollback immediately
kubectl rollout undo deployment/model-api

# Verify error rate drops
```

### If data issue:
- Enable fallback to cached predictions
- Alert data team

### If infrastructure:
- Check pod health: `kubectl get pods`
- Restart unhealthy pods
- Check resource limits

## Communication
- Post in #ml-incidents Slack channel
- Update status page if user-facing
- Create post-mortem doc

## Post-Incident
- Root cause analysis
- Update monitoring/alerts
-Update this runbook
```

### Automated Remediation

**Self-healing with Kubernetes**:

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: model
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          failureThreshold: 3
          periodSeconds: 10
        # Kubernetes automatically restarts on failure
        
        resources:
          limits:
            memory: "4Gi"
          # Kubernetes kills pod if exceeded (OOM)
```

**Circuit breaker** (prevent cascade failures):

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_external_service():
    # If fails 5 times, circuit opens
    # Requests fail fast for 60s
    # Then tries again (half-open state)
    response = requests.get("external-api")
    return response
```

---

## Production Best Practices

### 1. Model Versioning Strategy

**Semantic versioning for models**:
- `v1.0.0`: Major (breaking changes, retraining)
- `v1.1.0`: Minor (new features, compatible)
- `v1.1.1`: Patch (bug fixes)

**Track**:
- Model file
- Training code commit
- Training data version
- Hyperparameters
- Dependencies

### 2. Gradual Rollouts

**Never deploy 100% immediately**:
1. Deploy to canary (5%)
2. Monitor for 1 hour
3. Increase to 25%
4. Monitor for 4 hours
5. Increase to 50%
6. Monitor overnight
7. Deploy 100% (or rollback)

### 3. Feature Flags

**Control features dynamically**:

```python
from feature_flags import is_enabled

def predict(user_id, features):
    if is_enabled("use_new_model", user_id=user_id):
        return model_v2.predict(features)
    else:
        return model_v1.predict(features)
```

**Benefits**:
- Instant rollback (flip flag)
- A/B test easily
- Gradual rollout by user segment

### 4. Comprehensive Testing

**Test Pyramid**:
```
         /\
        / E2E \        Few (slow, expensive)
       /________\
      /Integration\    Some (medium speed)
     /______________\
    / Unit Tests     \  Many (fast, cheap)
   /__________________\
```

**ML-specific tests**:
- Model accuracy > threshold
- Inference time < SLA
- No NaN in predictions
- Bias/fairness checks
- Schema validation

### 5. Documentation

**Model Card** (mandatory for production models):
- Model purpose
- Training data
- Performance metrics
- Limitations
- Ethical considerations
- Intended use cases
- NOT intended for...

### 6. Security

**Least Privilege**:
- Model service can only read model files
- Cannot write to production database
- Limited network access

**Secrets Management**:
- Never hardcode API keys
- Use AWS Secrets Manager / HashiCorp Vault
- Rotate secrets regularly

**Input Sanitization**:
- Validate all inputs
- Prevent SQL injection / code execution
- Rate limiting

### 7. Disaster Recovery

**Backup strategy**:
- Model files: S3 with versioning
- Feature store: Regular snapshots
- Metadata: Database backups

**RTO/RPO**:
- Recovery Time Objective: < 1 hour
- Recovery Point Objective: Last good model version

**Practice**:
- Quarterly disaster recovery drills
- Document restore procedures

---

## Key Takeaways

✅ **Scaling**: Horizontal scaling with Kubernetes, auto-scaling based on metrics, load balancing

✅ **Monitoring**: Data drift (KS test, PSI), concept drift (performance tracking), feature importance shifts

✅ **A/B Testing**: Proper experiment design, hash-based routing, statistical significance testing

✅ **Feature Stores**: Feast for online/offline serving, shared feature logic prevents skew

✅ **ML Platform**: Self-service platform abstracts infrastructure, standardizes workflows

✅ **Cost Optimization**: Model compression, right-sizing instances, spot instances, caching

✅ **Incident Response**: Runbooks, automated remediation, circuit breakers, rollback procedures

✅ **Best Practices**: Semantic versioning, gradual rollouts, feature flags, comprehensive testing, documentation

---

*Previous: [Intermediate Deployment Strategies](02-intermediate-deployment-strategies.md)*

