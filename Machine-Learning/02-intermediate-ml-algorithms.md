# Intermediate ML Algorithms and Techniques

## Table of Contents
1. [Advanced Supervised Learning Algorithms](#advanced-supervised-learning-algorithms)
2. [Ensemble Methods](#ensemble-methods)
3. [Feature Engineering](#feature-engineering)
4. [Model Evaluation and Selection](#model-evaluation-and-selection)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Handling Imbalanced Data](#handling-imbalanced-data)
7. [Regularization Techniques](#regularization-techniques)
8. [Unsupervised Learning Deep Dive](#unsupervised-learning-deep-dive)
9. [Best Practices](#best-practices)

---

## Advanced Supervised Learning Algorithms

### Support Vector Machines (SVM)

**Idea**: Find the hyperplane that best separates classes with maximum margin.

**Key Concepts**:
- **Hyperplane**: Decision boundary separating classes
- **Support Vectors**: Data points closest to decision boundary
- **Margin**: Distance from hyperplane to nearest point
- **Kernel Trick**: Transform data to higher dimensions for non-linear separation

**Kernels**:
1. **Linear**: `K(x,y) = x·y` - For linearly separable data
2. **Polynomial**: `K(x,y) = (x·y + c)^d` - For polynomial boundaries
3. **RBF (Radial Basis Function)**: `K(x,y) = exp(-γ||x-y||²)` - Most common, handles non-linear data
4. **Sigmoid**: Similar to neural network activation

**Hyperparameters**:
- **C**: Regularization (higher = less regularization, risk overfitting)
- **Gamma** (RBF kernel): Influence of single training example (higher = closer fit)

**Pros**:
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors)
- Versatile (different kernels)

**Cons**:
- Slow on large datasets
- Doesn't provide probability estimates directly
- Sensitive to feature scaling

**Use Cases**: Text classification, image recognition, bioinformatics

### Random Forest

**Idea**: Ensemble of decision trees, each trained on random subset of data and features.

**Algorithm**:
1. Create N decision trees
2. For each tree:
   - Bootstrap sample (random sample with replacement)
   - At each split, consider random subset of features
3. Predictions: Majority vote (classification) or average (regression)

**Hyperparameters**:
- **n_estimators**: Number of trees (more = better, but slower)
- **max_depth**: Maximum tree depth
- **max_features**: Number of features to consider at each split
- **min_samples_split**: Minimum samples to split node

**Pros**:
- Handles non-linear relationships
- Reduces overfitting compared to single decision tree
- Feature importance built-in
- Works well without much tuning

**Cons**:
- Less interpretable than single tree
- Can be slow on large datasets
- May overfit on noisy data

**Feature Importance**: Measures how much each feature reduces impurity

**Use Cases**: Fraud detection, recommendation systems, medical diagnosis

### Gradient Boosting (GBM, XGBoost, LightGBM, CatBoost)

**Idea**: Build trees sequentially, each correcting errors of previous trees.

**Algorithm**:
1. Start with simple model (e.g., mean prediction)
2. Calculate residuals (errors)
3. Train new tree to predict residuals
4. Add tree to model with weight (learning rate)
5. Repeat

**Popular Implementations**:
1. **XGBoost**: Fast, regularization, handles missing values
2. **LightGBM**: Very fast, efficient for large datasets
3. **CatBoost**: Handles categorical features well

**Hyperparameters**:
- **n_estimators**: Number of boosting rounds
- **learning_rate**: Step size shrinkage (smaller = more conservative)
- **max_depth**: Tree depth
- **subsample**: Fraction of samples for each tree

**Pros**: 
- State-of-the-art performance on tabular data
- Handles missing values
- Built-in feature importance

**Cons**:
- Prone to overfitting if not tuned
- Sensitive to hyperparameters
- Longer training time

**Use Cases**: Kaggle competitions winner, click-through rate prediction, ranking

---

## Ensemble Methods

**Idea**: Combine multiple models for better performance than individual models.

### Types of Ensembles

#### 1. Bagging (Bootstrap Aggregating)

**Process**:
- Train multiple models on bootstrapped samples
- Average predictions (regression) or vote (classification)

**Reduces**: Variance (overfitting)

**Example**: Random Forest

#### 2. Boosting

**Process**:
- Train models sequentially
- Each model focuses on mistakes of previous models

**Reduces**: Bias (underfitting)

**Examples**: AdaBoost, Gradient Boosting, XGBoost

#### 3. Stacking

**Process**:
- Train multiple diverse models (base models)
- Train meta-model on predictions of base models

**Example**:
```
Base Models: Random Forest, SVM, Logistic Regression
Meta-Model: Logistic Regression on base model predictions
```

**Pros**: Can achieve best performance
**Cons**: Complex, risk of overfitting

### Voting Classifier

**Hard Voting**: Majority class vote  
**Soft Voting**: Average predicted probabilities (usually better)

---

## Feature Engineering

**Definition**: Creating new features or transforming existing ones to improve model performance.

### Feature Creation

#### 1. Domain-Specific Features

**Example - Time Series**:
- From timestamp: hour, day, month, year, day_of_week, is_weekend
- Lag features: value from previous time steps
- Rolling statistics: moving average, moving std

**Example - Text**:
- Text length, word count
- Presence of specific keywords
- Sentiment score

**Example - E-commerce**:
- Total spending = sum of purchases
- Average order value
- Days since last purchase

#### 2. Polynomial Features

Transform features to capture non-linear relationships.

**Example**: `x` → `x, x², x³, x*y, ...`

**Use**: With linear models to capture non-linearity

#### 3. Interaction Features

Combine features to capture relationships.

**Example**: `income * family_size` for spending prediction

### Feature Transformation

#### 1. Scaling

**Normalization** (Min-Max Scaling):
```
x_scaled = (x - x_min) / (x_max - x_min)
```
Output: [0, 1]

**Standardization** (Z-score):
```
x_scaled = (x - mean) / std
```
Output: Mean=0, Std=1

**When**: Required for SVM, KNN, Neural Networks; Not needed for tree-based models

#### 2. Log Transformation

**When**: Skewed distributions

**Effect**: Makes distribution more normal, reduces effect of outliers

**Example**: Income, population data

#### 3. Binning (Discretization)

Convert continuous to categorical.

**Example**: Age → [0-18, 19-30, 31-50, 51+]

**Benefit**: Captures non-linear relationships with linear models

### Feature Selection

**Goal**: Select most relevant features, remove redundant/irrelevant ones.

**Benefits**:
- Reduce overfitting
- Improve training speed
- Improve model interpretability

**Methods**:

#### 1. Filter Methods

Based on statistical tests, independent of ML model.

- **Correlation**: Remove highly correlated features
- **Chi-square test**: For categorical features
- **ANOVA F-test**: For numerical features
- **Mutual Information**: Measures dependency

#### 2. Wrapper Methods

Use ML model to evaluate feature subsets.

- **Recursive Feature Elimination (RFE)**: Iteratively remove least important features
- **Forward Selection**: Start with no features, add one at a time
- **Backward Elimination**: Start with all features, remove one at a time

**Pros**: Account for feature interactions  
**Cons**: Computationally expensive

#### 3. Embedded Methods

Feature selection during model training.

- **L1 Regularization (Lasso)**: Drives coefficients to zero
- **Tree-based feature importance**: Random Forest, Gradient Boosting

---

## Model Evaluation and Selection

### Cross-Validation Strategies

#### 1. K-Fold Cross-Validation

Split data into K folds, train K times.

**Standard K-Fold**: For general use (K=5 or 10)

#### 2. Stratified K-Fold

Preserve class distribution in each fold.

**When**: Imbalanced classification problems

#### 3. Time Series Split

Respect temporal order.

**Process**:
```
Fold 1: Train [1:100], Test [101:150]
Fold 2: Train [1:150], Test [151:200]
Fold 3: Train [1:200], Test [201:250]
```

**When**: Time series data, prevents data leakage

#### 4. Leave-One-Out (LOO)

Train on all data except one sample, test on that sample.

**When**: Very small datasets  
**Con**: Computationally expensive

### Model Selection Strategies

#### 1. Validation Curve

Plot model performance vs single hyperparameter.

**Use**: Visualize underfitting/overfitting

#### 2. Learning Curve

Plot model performance vs training set size.

**Diagnose**:
- **High bias (underfitting)**: Both train and val scores low
- **High variance (overfitting)**: Large gap between train and val scores
- **Good fit**: Both scores high and close

**Solution for low performance**:
- More data (if high variance)
- More complex model (if high bias)

---

## Hyperparameter Tuning

**Hyperparameters**: Model settings not learned from data (e.g., learning rate, tree depth).

### Methods

#### 1. Grid Search

**Process**: Try all combinations of specified hyperparameter values.

**Example**:
```python
params = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}
# Tests: 3 × 3 = 9 combinations
```

**Pros**: Exhaustive, guaranteed to find best combination in grid  
**Cons**: Exponentially slow as parameters increase

#### 2. Random Search

**Process**: Sample random combinations.

**Pros**:
- Faster than grid search
- Often finds good solutions
- Better for high-dimensional spaces

**Cons**: Not exhaustive

#### 3. Bayesian Optimization

**Process**: Use previous evaluations to choose next hyperparameters.

**Libraries**: Optuna, Hyperopt, Scikit-Optimize

**Pros**: More efficient, guided search  
**Cons**: More complex to set up

#### 4. Automated ML (AutoML)

Automates entire ML pipeline including hyperparameter tuning.

**Tools**: Auto-sklearn, AutoGluon, H2O.ai, Google AutoML

**Pros**: Easy to use, good baseline  
**Cons**: Less control, can be slow

---

## Handling Imbalanced Data

**Problem**: One class has far more examples than others.

**Example**: Fraud detection (99.9% legitimate, 0.1% fraud)

**Issue**: Model learns to predict majority class always.

### Techniques

#### 1. Resampling

**Oversampling** (Increase minority class):
- **Random Oversampling**: Duplicate minority samples
- **SMOTE** (Synthetic Minority Over-sampling): Create synthetic samples
  - Interpolate between minority samples and neighbors

**Undersampling** (Decrease majority class):
- **Random Undersampling**: Remove majority samples
- **Tomek Links**: Remove majority samples near decision boundary

**Combined**: SMOTEENN, SMOTETomek

#### 2. Class Weights

Assign higher weight to minority class during training.

**In scikit-learn**: `class_weight='balanced'`

**Effect**: Penalizes mistakes on minority class more

#### 3. Anomaly Detection Algorithms

Treat minority class as anomaly.

**Algorithms**: Isolation Forest, One-Class SVM

#### 4. Evaluation Metrics

Don't use accuracy! Use:
- **Precision, Recall, F1-Score**
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve (better for very imbalanced data)

---

## Regularization Techniques

**Goal**: Prevent overfitting by penalizing model complexity.

### L1 Regularization (Lasso)

**Penalty**: Sum of absolute values of coefficients

**Effect**: 
- Drives some coefficients to exactly zero
- Performs feature selection
- Sparse models

**Use**: When you want automatic feature selection

### L2 Regularization (Ridge)

**Penalty**: Sum of squared coefficients

**Effect**:
- Shrinks coefficients toward zero (but not exactly zero)
- Distributes importance across features
- Handles multicollinearity

**Use**: General regularization

### Elastic Net

**Combination** of L1 and L2.

**Parameter**: α controls L1 vs L2 ratio

**Use**: Best of both worlds

### Dropout (Neural Networks)

Randomly drop neurons during training.

**Effect**: Prevents co-adaptation of neurons

---

## Unsupervised Learning Deep Dive

### Advanced Clustering

#### DBSCAN (Density-Based Spatial Clustering)

**Idea**: Clusters as dense regions separated by sparse regions.

**Parameters**:
- **eps**: Maximum distance between points in cluster
- **min_samples**: Minimum points to form dense region

**Pros**:
- Finds arbitrary shaped clusters
- Detects outliers
- No need to specify number of clusters

**Cons**:
- Sensitive to parameters
- Struggles with varying densities

#### Hierarchical Clustering

**Types**:
1. **Agglomerative** (Bottom-up): Start with individual points, merge
2. **Divisive** (Top-down): Start with all points, split

**Linkage Methods**:
- **Single**: Minimum distance between clusters
- **Complete**: Maximum distance
- **Average**: Average distance
- **Ward**: Minimize within-cluster variance

**Output**: Dendrogram (tree diagram)

**Use**: Taxonomies, gene sequence analysis

### Dimensionality Reduction

#### Principal Component Analysis (PCA)

**Goal**: Find directions (principal components) of maximum variance.

**Process**:
1. Standardize data
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. Project data onto top K components

**Output**: Uncorrelated features in order of importance

**Use Cases**:
- Data visualization (reduce to 2D/3D)
- Noise reduction
- Speed up training
- Multicollinearity reduction

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Goal**: Visualize high-dimensional data in 2D/3D.

**Process**: Preserve local structure (nearby points stay nearby)

**Pros**: Beautiful visualizations, preserves local structure  
**Cons**: 
- Non-deterministic
- Computationally expensive
- Only for visualization (can't transform new data)

**Hyperparameter**: Perplexity (5-50, affects local vs global structure)

#### UMAP (Uniform Manifold Approximation and Projection)

**Newer alternative to t-SNE**.

**Pros**:
- Faster than t-SNE
- Preserves both local and global structure
- Can transform new data
- Better theoretical foundation

---

## Best Practices

### 1. Start Simple

Begin with simple models (Logistic Regression, Decision Tree) before complex ones.

**Benefits**:
- Faster iteration
- Baseline performance
- Better understanding

### 2. Feature Engineering > Algorithm Selection

Often, better features outperform better algorithms.

### 3. Domain Knowledge

Understanding the problem domain leads to better features and model choices.

### 4. Monitor Multiple Metrics

Don't rely on single metric; understand trade-offs.

### 5. Validate on Holdout Data

Never touch test set until final evaluation.

### 6. Document Everything

Track experiments: data, features, hyperparameters, results.

**Tools**: MLflow, Weights & Biases, Neptune.ai

### 7. Error Analysis

Analyze mistakes to understand model weaknesses.

**Questions**:
- Which types of examples does the model fail on?
- Are there patterns in errors?
- Can we create features to address these errors?

### 8. Pipeline Development

Use scikit-learn pipelines to prevent data leakage.

**Example**:
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier())
])
```

**Benefits**:
- Ensures preprocessing applied consistently
- Prevents data leakage
- Easier deployment

---

## Key Takeaways

✅ **Advanced algorithms** (SVM, Random Forest, Gradient Boosting) offer superior performance

✅ **Ensemble methods** combine models for better predictions

✅ **Feature engineering** is crucial for model performance

✅ **Cross-validation** provides reliable performance estimates

✅ **Hyperparameter tuning** (Grid Search, Random Search, Bayesian) optimizes models

✅ **Imbalanced data** requires special techniques (SMOTE, class weights, appropriate metrics)

✅ **Regularization** (L1, L2) prevents overfitting

✅ **Advanced clustering** (DBSCAN, Hierarchical) handles complex data structures

✅ **Best practices** include starting simple, focusing on features, and thorough validation

---

## What's Next?

- **Deep Learning**: [Advanced ML - Deep Learning](03-advanced-ml-deep-learning.md)
- **Specialized Domains**: [NLP](../Natural-Language-Processing/01-beginner-nlp-basics.md), [Computer Vision](../Computer-Vision/01-beginner-cv-basics.md)
- **Deployment**: [MLOps Basics](../MLOps-Deployment/01-beginner-mlops-basics.md)

---

*Previous: [ML Fundamentals](01-beginner-ml-fundamentals.md) | Next: [Deep Learning](03-advanced-ml-deep-learning.md)*
