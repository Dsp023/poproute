# 🚀 Quick Start Guide

Get up and running with AI/ML development in minutes!

## For Absolute Beginners

### Step 1: Setup Python Environment

```bash
# Install Python (if not already installed)
# Download from: https://www.python.org/downloads/

# Verify installation
python --version

# Create project directory
mkdir my-ai-project
cd my-ai-project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Essential Libraries

```bash
# Install core packages
pip install numpy pandas matplotlib scikit-learn jupyter

# Launch Jupyter
jupyter notebook
```

### Step 3: Your First ML Code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load sample dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

## For LLM Development

### Step 1: Get API Key

```bash
# Get OpenAI API key from: https://platform.openai.com/api-keys
# Store in .env file (NEVER commit this!)

echo "OPENAI_API_KEY=your-key-here" > .env
```

### Step 2: Install LLM Libraries

```bash
pip install openai langchain python-dotenv
```

### Step 3: Your First LLM Call

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain AI in simple terms."}
    ]
)

print(response.choices[0].message.content)
```

## For RAG Development

### Step 1: Install RAG Libraries

```bash
pip install langchain openai chromadb python-dotenv
```

### Step 2: Simple RAG System

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Your documents
documents = [
    "Paris is the capital of France.",
    "Python is a programming language.",
    "Machine learning is a subset of AI."
]

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100)
texts = text_splitter.create_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever()
)

# Ask questions
answer = qa_chain.run("What is Paris?")
print(answer)
```

## Common Commands Cheat Sheet

### Python/pip
```bash
python --version              # Check Python version
pip install package          # Install package
pip install -r requirements.txt  # Install from file
pip freeze > requirements.txt    # Save installed packages
```

### Virtual Environments
```bash
python -m venv venv          # Create venv
venv\Scripts\activate        # Activate (Windows)
source venv/bin/activate     # Activate (Mac/Linux)
deactivate                   # Deactivate
```

### Jupyter
```bash
jupyter notebook             # Start Jupyter
jupyter lab                  # Start JupyterLab
```

### Git
```bash
git init                     # Initialize repo
git add .                    # Stage all files
git commit -m "message"      # Commit changes
git remote add origin URL    # Add remote
git push -u origin main      # Push to GitHub
```

## Troubleshooting

### Issue: Module not found
```bash
# Solution: Install the module
pip install module-name
```

### Issue: API key error
```bash
# Solution: Check .env file exists and has correct key
# Make sure to load_dotenv() in your code
```

### Issue: GPU not detected (PyTorch)
```bash
# Check CUDA version
nvidia-smi

# Install correct PyTorch version
# Visit: https://pytorch.org/get-started/locally/
```

## Next Steps

1. **Beginners**: Start with [AI/ML Fundamentals](./ai-ml/README.md)
2. **LLM Developers**: Check [LLM Guide](./llms/README.md)
3. **RAG Builders**: Read [RAG Documentation](./rag/README.md)
4. **Hands-on**: Try [Projects](./projects/README.md)

---

**Need help?** Open an issue or check the [resources](./resources/README.md)!
