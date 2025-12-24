# Code Examples and Implementations

Practical code snippets for common AI/ML/LLM tasks. All examples use Python.

---

## Setup and Installation

```bash
# Create virtual environment
python -m venv mlenv
source mlenv/bin/activate  # On Windows: mlenv\Scripts\activate

# Essential libraries
pip install numpy pandas matplotlib scikit-learn

# Deep learning
pip install torch torchvision  # PyTorch
pip install tensorflow  # TensorFlow

# NLP
pip install transformers datasets  # Hugging Face
pip install spacy nltk

# LLM development
pip install langchain openai
pip install chromadb  # Vector database

# Data science
pip install jupyter notebook
```

---

## Machine Learning Examples

### 1. Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")
print(f"Coefficient: {model.coef_[0]}")
```

### 2. Classification with Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Feature importance
importances = rf.feature_importances_
for name, importance in zip(iris.feature_names, importances):
    print(f"{name}: {importance:.3f}")
```

### 3. K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Cluster
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, c='red', linewidths=3)
plt.title("K-Means Clustering")
plt.show()
```

###4. Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Model
model = RandomForestClassifier()

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## Deep Learning Examples

### 1. Simple Neural Network (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize
model = SimpleNN(input_size=10, hidden_size=20, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (pseudocode)
for epoch in range(100):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
```

### 2. Image Classification with CNN (TensorFlow/Keras)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')
```

### 3. Transfer Learning with Pre-trained Model

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Load pre-trained model (without top layer)
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Add custom classifier
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile and train on your dataset
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# model.fit(your_data, your_labels, ...)
```

---

## NLP Examples

### 1. Text Classification with spaCy

```python
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample data
texts = ["I love this product", "Terrible experience", "Great quality", "Waste of money"]
labels = [1, 0, 1, 0]  # 1: Positive, 0: Negative

# Preprocess
def preprocess(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

processed_texts = [preprocess(text) for text in texts]

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_texts)

# Train classifier
classifier = LogisticRegression()
classifier.fit(X, labels)

# Predict
new_text = "Amazing product, highly recommend"
new_text_processed = preprocess(new_text)
new_text_vec = vectorizer.transform([new_text_processed])
prediction = classifier.predict(new_text_vec)[0]
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

### 2. Named Entity Recognition with Transformers

```python
from transformers import pipeline

# Load NER pipeline
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# Example text
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

# Extract entities
entities = ner(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} (confidence: {entity['score']:.2f})")
```

### 3. Text Generation with GPT-2

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer.encode(prompt, return_tensors="pt")

outputs = model.generate(
    inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

---

## LLM and RAG Examples

### 1. Using OpenAI API

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# Simple completion
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

### 2. Simple RAG with LangChain and Chroma

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load documents
loader = TextLoader("your_document.txt")
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key="your-key")
vectorstore = Chroma.from_documents(texts, embeddings)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key="your-key"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query
query = "What is the main topic of the document?"
result = qa_chain.run(query)
print(result)
```

### 3. Local LLM with Ollama

```python
import ollama

# Generate text (requires Ollama installed)
response = ollama.generate(
    model='llama2',
    prompt='Why is the sky blue?'
)

print(response['response'])

# Chat interface
messages = [
    {'role': 'user', 'content': 'Why is the sky blue?'}
]

response = ollama.chat(model='llama2', messages=messages)
print(response['message']['content'])
```

### 4. Embeddings and Similarity Search

```python
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Documents
documents = [
    "The cat sat on the mat",
    "Dogs are great pets",
    "I love machine learning",
    "The feline rested on the rug"
]

# Create embeddings
doc_embeddings = model.encode(documents)

# Query
query = "cat on mat"
query_embedding = model.encode(query)

# Calculate similarities
similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

# Find most similar
most_similar_idx = similarities.argmax()
print(f"Most similar document: {documents[most_similar_idx]}")
print(f"Similarity score: {similarities[most_similar_idx]:.3f}")
```

---

## MLOps Examples

### 1. Experiment Tracking with MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameters
    n_estimators = 100
    max_depth = 5
    
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Train
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
print(f"Accuracy: {accuracy:.3f}")
```

### 2. Simple FastAPI Model Serving

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class PredictionInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    # Convert to numpy array
    features = np.array([input_data.features])
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()
    
    return PredictionOutput(
        prediction=int(prediction),
        probability=float(probability)
    )

# Run with: uvicorn main:app --reload
```

---

## Utility Functions

### Data Preprocessing

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df, numerical_cols, categorical_cols, target_col):
    """Preprocess dataset"""
    df = df.copy()
    
    # Handle missing values
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Encode categorical features
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y, scaler
```

### Model Evaluation

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_classifier(y_true, y_pred, class_names=None):
    """Comprehensive classifier evaluation"""
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
```

---

## Tips for Running Examples

1. **Install dependencies** before running examples
2. **Replace API keys** with your own
3. **Adjust paths** to your files
4. **Start small** - test with small datasets first
5. **Check documentation** for latest API changes

---

**These examples provide starting points. Customize for your specific use cases!**

*For more examples, check the topic-specific guides in this repository.*
