# 🧠 Artificial Intelligence & Machine Learning

Welcome to the AI/ML section! This comprehensive guide will take you from fundamentals to advanced concepts.

## 📚 Table of Contents

1. [Fundamentals](#fundamentals)
2. [Core Algorithms](#core-algorithms)
3. [Deep Learning](#deep-learning)
4. [Practical Projects](#practical-projects)
5. [Resources](#resources)

## Fundamentals

### What is Machine Learning?

Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.

**Three main types:**
- **Supervised Learning**: Learning from labeled data
- **Unsupervised Learning**: Finding patterns in unlabeled data
- **Reinforcement Learning**: Learning through rewards and penalties

### Key Concepts

#### 1. **Features & Labels**
```python
# Example: House price prediction
features = ['square_feet', 'bedrooms', 'location']  # Input
label = 'price'  # Output we want to predict
```

#### 2. **Training vs Testing**
- **Training Data** (80%): Used to teach the model
- **Testing Data** (20%): Used to evaluate performance

#### 3. **Overfitting & Underfitting**
- **Overfitting**: Model memorizes training data, poor on new data
- **Underfitting**: Model too simple, poor on all data
- **Just Right**: Generalizes well to new data

## Core Algorithms

### 1. Linear Regression

Predicts continuous values using a linear relationship.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
prediction = model.predict([[6]])
print(f"Prediction for X=6: {prediction[0]}")  # Output: 12
```

**When to use:**
- Predicting house prices
- Sales forecasting
- Trend analysis

### 2. Logistic Regression

For binary classification problems (yes/no, true/false).

```python
from sklearn.linear_model import LogisticRegression

# Email spam classification example
X = [[1, 2], [2, 3], [3, 3], [4, 5]]  # Features: word frequency
y = [0, 0, 1, 1]  # Labels: 0=not spam, 1=spam

model = LogisticRegression()
model.fit(X, y)

# Predict new email
new_email = [[2.5, 3]]
is_spam = model.predict(new_email)
probability = model.predict_proba(new_email)
```

### 3. Decision Trees

Makes decisions through a series of questions.

```python
from sklearn.tree import DecisionTreeClassifier

# Customer purchase prediction
X = [[25, 50000], [45, 60000], [30, 80000], [35, 70000]]
y = [0, 1, 1, 1]  # 0=no purchase, 1=purchase

model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Visualize (optional)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=['Age', 'Income'])
plt.show()
```

### 4. Random Forest

An ensemble of decision trees for better accuracy.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
print("Feature Importances:", importances)
```

### 5. K-Nearest Neighbors (KNN)

Classifies based on similarity to nearby data points.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

### 6. Support Vector Machines (SVM)

Finds the optimal boundary between classes.

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
```

## Deep Learning

### Neural Networks Basics

A neural network consists of layers of interconnected nodes (neurons).

```python
import tensorflow as tf
from tensorflow import keras

# Simple neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

### Common Architectures

#### 1. **Convolutional Neural Networks (CNN)**
For image processing and computer vision.

```python
# CNN for image classification
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

#### 2. **Recurrent Neural Networks (RNN)**
For sequential data like text and time series.

```python
# LSTM for text processing
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 128),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation='sigmoid')
])
```

## Practical Projects

### Beginner Projects

1. **Iris Flower Classification**
   - Dataset: Built-in sklearn dataset
   - Goal: Classify iris species
   - Algorithms: Logistic Regression, Decision Trees

2. **House Price Prediction**
   - Dataset: Boston Housing
   - Goal: Predict house prices
   - Algorithm: Linear Regression, Random Forest

3. **Digit Recognition**
   - Dataset: MNIST
   - Goal: Recognize handwritten digits
   - Algorithm: Simple Neural Network

### Intermediate Projects

1. **Sentiment Analysis**
   - Dataset: IMDB reviews
   - Goal: Classify positive/negative reviews
   - Algorithm: LSTM, BERT

2. **Image Classification**
   - Dataset: CIFAR-10
   - Goal: Classify objects in images
   - Algorithm: CNN, Transfer Learning

3. **Recommendation System**
   - Dataset: MovieLens
   - Goal: Recommend movies
   - Algorithm: Collaborative Filtering

## Best Practices

### 1. Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
```

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

## Model Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Detailed report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# MSE
mse = mean_squared_error(y_test, y_pred)

# RMSE
rmse = np.sqrt(mse)

# R² Score
r2 = r2_score(y_test, y_pred)

# MAE
mae = mean_absolute_error(y_test, y_pred)
```

## Resources

### 📖 Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning" by Ian Goodfellow
- "Pattern Recognition and Machine Learning" by Christopher Bishop

### 🎓 Courses
- Andrew Ng's Machine Learning (Coursera)
- Fast.ai Practical Deep Learning
- MIT Introduction to Deep Learning

### 🛠️ Tools & Libraries
- **Scikit-learn**: Classical ML algorithms
- **TensorFlow/Keras**: Deep learning framework
- **PyTorch**: Research-friendly DL framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization

### 📝 Papers
- "Deep Residual Learning for Image Recognition" (ResNet)
- "Attention Is All You Need" (Transformer)
- "ImageNet Classification with Deep CNNs" (AlexNet)

---

**Next Steps:**
- Explore the [LLMs section](../llms/README.md)
- Check out [RAG systems](../rag/README.md)
- Try hands-on [projects](../projects/README.md)
