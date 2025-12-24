# 🛠️ Tech Resources & Tools

Essential tools, frameworks, and best practices for AI/ML development and deployment.

## 🐍 Python for AI/ML

### Essential Libraries

#### Data Science Stack
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Quick data analysis
df = pd.read_csv('data.csv')
print(df.describe())
print(df.info())

# Visualization
sns.pairplot(df)
plt.show()
```

#### Deep Learning Frameworks

**PyTorch**
```bash
pip install torch torchvision torchaudio
```

**TensorFlow**
```bash
pip install tensorflow
```

**Hugging Face**
```bash
pip install transformers datasets accelerate
```

### Development Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
# or
venv\Scripts\activate  # Windows

# Install Jupyter
pip install jupyter notebook jupyterlab

# Install CUDA (for GPU support)
# Visit: https://pytorch.org/get-started/locally/
```

## 🚀 Popular Frameworks

### LangChain
Application framework for LLMs.

```bash
pip install langchain langchain-community langchain-openai
```

**Quick Start:**
```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["product"],
    template="Write a tagline for {product}"
)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("AI-powered task manager"))
```

### LlamaIndex
Data framework for connecting LLMs to data.

```bash
pip install llama-index
```

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader('data').load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is this about?")
print(response)
```

### Streamlit
Build AI web apps quickly.

```bash
pip install streamlit
```

```python
# app.py
import streamlit as st
import openai

st.title("AI Assistant")

prompt = st.text_input("Ask me anything:")

if prompt:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    st.write(response.choices[0].message.content)
```

Run: `streamlit run app.py`

## 📊 Vector Databases Comparison

| Database | Type | Best For | Pricing |
|----------|------|----------|---------|
| **Chroma** | Local | Development, Small projects | Free |
| **FAISS** | Local | High performance, Research | Free |
| **Pinecone** | Cloud | Production, Scale | Paid (Free tier) |
| **Weaviate** | Cloud/Self-hosted | Feature-rich apps | Free/Paid |
| **Qdrant** | Cloud/Self-hosted | Production, Privacy | Free/Paid |
| **Milvus** | Self-hosted | Large scale, Enterprise | Free |

### Installation & Setup

**Chroma**
```bash
pip install chromadb
```

**FAISS**
```bash
pip install faiss-cpu  # or faiss-gpu
```

**Pinecone**
```bash
pip install pinecone-client
```

**Weaviate**
```bash
docker run -p 8080:8080 semitechnologies/weaviate:latest
```

## ☁️ Cloud Platforms

### AWS (Amazon Web Services)

**SageMaker** - ML platform
```python
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()
sess = sagemaker.Session()

# Train model
estimator = sagemaker.estimator.Estimator(
    image_uri='...',
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

estimator.fit({'training': s3_train_data})
```

**Bedrock** - Managed LLM service
```python
import boto3

bedrock = boto3.client('bedrock-runtime')

response = bedrock.invoke_model(
    modelId='anthropic.claude-v2',
    body=json.dumps({"prompt": "Hello"})
)
```

### Google Cloud Platform

**Vertex AI**
```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

model = aiplatform.Model.upload(
    display_name='my-model',
    artifact_uri='gs://my-bucket/model'
)

endpoint = model.deploy(machine_type='n1-standard-4')
```

### Azure

**Azure OpenAI**
```python
import openai

openai.api_type = "azure"
openai.api_base = "https://YOUR_RESOURCE.openai.azure.com/"
openai.api_key = "YOUR_KEY"

response = openai.ChatCompletion.create(
    engine="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## 🔧 MLOps Tools

### Weights & Biases
Experiment tracking and visualization.

```bash
pip install wandb
```

```python
import wandb

# Initialize
wandb.init(project="my-project")

# Log metrics
wandb.log({"loss": 0.5, "accuracy": 0.9})

# Log model
wandb.log_model(model, "model")
```

### MLflow
ML lifecycle management.

```bash
pip install mlflow
```

```python
import mlflow

mlflow.start_run()

# Log parameters
mlflow.log_param("learning_rate", 0.01)

# Log metrics
mlflow.log_metric("accuracy", 0.95)

# Log model
mlflow.sklearn.log_model(model, "model")

mlflow.end_run()
```

### DVC (Data Version Control)
```bash
pip install dvc

# Initialize
dvc init

# Track data
dvc add data/large_dataset.csv
git add data/large_dataset.csv.dvc .gitignore
git commit -m "Add dataset"

# Push to remote storage
dvc remote add -d myremote s3://mybucket/dvc
dvc push
```

## 🐳 Docker for AI/ML

### Basic Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run
CMD ["python", "app.py"]
```

### With GPU Support

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "app.py"]
```

### Docker Compose for RAG App

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - vectordb
      
  vectordb:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
```

## 📦 Project Structure

### Standard ML Project

```
my-ml-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py
├── models/
│   └── trained_models/
├── tests/
│   └── test_model.py
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

### RAG Application

```
rag-app/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── routes.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── rag.py
│   │   ├── embeddings.py
│   │   └── vectorstore.py
│   └── utils/
│       └── helpers.py
├── data/
│   └── documents/
├── tests/
│   └── test_rag.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🔐 Best Practices

### API Key Management

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Never hardcode keys!
# ❌ api_key = "sk-..."
# ✅ api_key = os.getenv("OPENAI_API_KEY")
```

**.env file:**
```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
DATABASE_URL=postgresql://...
```

**.gitignore:**
```
.env
*.env
.env.local
```

### Error Handling

```python
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_llm(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        raise
```

### Testing

```python
import pytest
from src.models import RAGSystem

def test_embedding_generation():
    rag = RAGSystem()
    embedding = rag.get_embedding("test text")
    assert len(embedding) == 1536

def test_retrieval():
    rag = RAGSystem()
    rag.add_documents(["doc1", "doc2"])
    results = rag.retrieve("query")
    assert len(results) > 0

@pytest.mark.integration
def test_end_to_end():
    rag = RAGSystem()
    answer = rag.query("What is AI?")
    assert isinstance(answer, str)
    assert len(answer) > 0
```

## 📚 Resources

### Documentation
- [PyTorch Docs](https://pytorch.org/docs)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Hugging Face Docs](https://huggingface.co/docs)
- [LangChain Docs](https://python.langchain.com/docs)

### Communities
- r/MachineLearning
- r/LocalLLaMA
- Hugging Face Forums
- Discord: EleutherAI, Weights & Biases

---

**Next:** Check out [deployment guides](./deployment.md) for production tips
