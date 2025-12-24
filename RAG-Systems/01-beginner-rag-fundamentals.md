# RAG Fundamentals - Comprehensive Beginner Guide

## Table of Contents
1. [What is RAG?](#what-is-rag)
2. [Why RAG is Needed](#why-rag-is-needed)
3. [RAG Architecture Overview](#rag-architecture-overview)
4. [Embeddings Deep Dive](#embeddings-deep-dive)
5. [Vector Databases](#vector-databases)
6. [Complete RAG Workflow](#complete-rag-workflow)
7. [RAG vs Fine-Tuning](#rag-vs-fine-tuning)
8. [Hands-On Implementation](#hands-on-implementation)
9. [Common Challenges](#common-challenges)
10. [Key Takeaways](#key-takeaways)

---

## What is RAG?

**RAG** = **Retrieval Augmented Generation**

### Simple Definition

**RAG enhances Large Language Model responses by retrieving relevant information from external knowledge sources before generating answers.**

### The Core Problem RAG Solves

**Without RAG**:
```
User: "What were our Q3 2024 sales numbers?"
LLM: "I don't have access to your specific sales data."
❌ LLM only knows what it was trained on
```

**With RAG**:
```
User: "What were our Q3 2024 sales numbers?"
RAG System:
  1. Searches company database for Q3 2024 sales reports
  2. Retrieves: "Q3 2024 Sales Report: Total revenue $2.5M, up 15% from Q2"
  3. Provides context to LLM
LLM: "According to the Q3 2024 sales report, your total revenue was $2.5 million, 
     representing a 15% increase from Q2."
✅ Accurate, grounded, verifiable answer
```

### Real-World Analogy

**Think of RAG like an open-book exam**:
- **Closed-book exam** (Normal LLM): Student relies only on memorized information
- **Open-book exam** (RAG): Student can reference textbooks and notes before answering

### Key Components

```
┌─────────────────────────────────────────────┐
│              RAG System                     │
├─────────────────────────────────────────────┤
│  1. Knowledge Base (Your Documents)         │
│  2. Embedding Model (Text → Vectors)        │
│  3. Vector Database (Fast Search)           │
│  4. Retriever (Find Relevant Docs)          │
│  5. LLM (Generate Answer)                   │
└─────────────────────────────────────────────┘
```

---

## Why RAG is Needed

### Problem 1: Knowledge Cutoff

**Issue**: LLMs have a training data cutoff date

**Example**:
- GPT-4 (trained through September 2021)
- Doesn't know about events after training
- Can't answer "What happened in 2023?"

**RAG Solution**:
```python
# Query current Wikipedia or news sources
# Retrieve most recent information
# Provide as context to LLM
→ LLM can answer questions about recent events
```

**Business Impact**:
- Customer support needs current product documentation
- Legal research requires latest regulations
- News summarization needs today's articles

### Problem 2: Hallucinations

**Issue**: LLMs sometimes generate plausible but false information

**Example Without RAG**:
```
User: "What is the capital of made-up country Atlantia?"
LLM: "The capital of Atlantia is Oceania." 
❌ Sounds confident but completely fabricated
```

**Example With RAG**:
```
User: "What is the capital of made-up country Atlantia?"
RAG: Searches knowledge base → No results found
LLM: "I couldn't find any information about a country called Atlantia 
     in the available sources."
✅ Honest, grounded response
```

**Statistics**:
- RAG can reduce hallucinations by 50-80%
- Particularly effective for factual questions
- Provides source attribution

### Problem 3: Domain-Specific Knowledge

**Issue**: General LLMs lack specialized knowledge

**Examples**:
- Your company's internal documentation
- Technical manuals for specific equipment
- Medical protocols for a specific hospital
- Legal precedents in a particular jurisdiction

**RAG Solution**:
```
Company Knowledge Base:
├── Product Documentation
├── Internal Policies
├── Technical Specifications
├── Customer FAQs
└── Sales Playbooks

→ RAG retrieves from these sources
→ LLM provides company-specific answers
```

**Use Cases**:
- **Internal chatbot**: Employee questions about policies
- **Customer support**: Product-specific troubleshooting
- **Technical documentation**: API usage examples
- **Research assistant**: Domain-specific papers

### Problem 4: Updating Knowledge is Expensive

**Traditional Approach** (Fine-tuning):
```
New information available
→ Prepare training data
→ Fine-tune model (hours/days, $$$$)
→ Deploy updated model
→ Repeat for each update
```

**RAG Approach**:
```
New information available
→ Add documents to knowledge base
→ Index (minutes, $)
→ Immediately available
→ No model retraining needed
```

**Cost Comparison**:
- Fine-tuning GPT-3.5: $100-$1,000+ per update
- RAG update: $0.01-$1 (just indexing)

### Problem 5: Source Attribution & Verification

**Without RAG**:
```
LLM: "The treatment for condition X is Y."
User: "How do you know that?"
LLM: [Cannot cite sources]
```

**With RAG**:
```
LLM: "According to the 2024 Medical Guidelines (source 1) and 
     the Johns Hopkins study (source 2), the treatment for 
     condition X is Y."
User: Can verify by checking source 1 and source 2
```

**Critical For**:
- Medical and legal advice
- Financial recommendations
- Scientific research
- Compliance and regulatory

---

## RAG Architecture Overview

### High-Level Architecture

```
        USER QUERY
            ↓
    ┌───────────────┐
    │ Query Encoder │ (Convert to embedding)
    └───────┬───────┘
            ↓
    ┌───────────────────┐
    │  Vector Database │ (Similarity search)
    │   (Knowledge Base)│
    └───────┬───────────┘
            ↓
   Retrieve Top-K Docs
            ↓
    ┌───────────────┐
    │ Prompt Builder│ (Query + Retrieved Docs)
    └───────┬───────┘
            ↓
    ┌───────────────┐
    │      LLM      │ (Generate answer)
    └───────┬───────┘
            ↓
      FINAL ANSWER
```

### Detailed Component Breakdown

#### 1. Knowledge Base (Offline Stage)

**Your Data Sources**:
- PDFs, Word documents
- Web pages, wikis
- Databases (SQL, NoSQL)
- APIs (REST, GraphQL)
- Code repositories
- Spreadsheets

**Preprocessing**:
```python
# 1. Load documents
documents = load_folder("./company_docs")

# 2. Clean and normalize
cleaned_docs = [clean_text(doc) for doc in documents]

# 3. Split into chunks
chunks = []
for doc in cleaned_docs:
    chunks.extend(split_into_chunks(doc, chunk_size=500))

# 4. Store metadata
for i, chunk in enumerate(chunks):
    chunk.metadata = {
        "source": chunk.file_name,
        "page": chunk.page_number,
        "chunk_id": i,
        "date_added": datetime.now()
    }
```

#### 2. Embedding Model

**Converts text to numerical vectors**:
```
"Artificial Intelligence" → [0.234, -0.123, 0.891, ..., 0.456]
                            (768 or 1536 dimensions)
```

**Properties**:
- Similar meaning → Similar vectors
- Captures semantic relationships
- Language-aware (can handle different phrasings)

**Popular Models**:
```python
# OpenAI (Paid, High Quality)
from openai import OpenAI
client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-small",  # or text-embedding-3-large
    input="Your text here"
)
embedding = response.data[0].embedding  # 1536 dimensions

# Sentence Transformers (Free, Open Source)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
embedding = model.encode("Your text here")

# Cohere (Paid)
import cohere
co = cohere.Client('your-api-key')
response = co.embed(
    texts=["Your text here"],
    model="embed-english-v3.0"
)
embedding = response.embeddings[0]
```

#### 3. Vector Database

**Specialized database for vector similarity search**:

**Core Operations**:
```python
# Index (add vectors)
vector_db.add(
    vectors=embeddings,
    ids=chunk_ids,
    metadata=chunk_metadata
)

# Query (find similar)
results = vector_db.search(
    query_vector=query_embedding,
    top_k=5,  # Return top 5 most similar
    filter={"category": "product_docs"}  # Optional filtering
)
```

**Under the Hood**:
- Approximate Nearest Neighbor (ANN) algorithms
- HNSW, IVF, PQ for speed
- Trade-off: Speed vs Accuracy

#### 4. Retrieval System

**Finds relevant documents for query**:

**Basic Similarity Search**:
```python
def retrieve(query, top_k=5):
    # 1. Embed query
    query_vector = embedding_model.encode(query)
    
    # 2. Search vector DB
    results = vector_db.similarity_search(
        query_vector,
        k=top_k
    )
    
    # 3. Return documents
    return [r.document for r in results]
```

**Similarity Metrics**:
- **Cosine Similarity**: Angle between vectors (most common)
- **Dot Product**: Magnitude and direction
- **Euclidean Distance**: Straight-line distance

#### 5. LLM Generator

**Generates answer using retrieved context**:

**Prompt Template**:
```
You are a helpful assistant. Answer the question based on the provided context.

Context:
{retrieved_document_1}
{retrieved_document_2}
{retrieved_document_3}

Question: {user_question}

Instructions:
- Answer based only on the provided context
- If the answer is not in the context, say "I don't have enough information"
- Cite which document your answer comes from

Answer:
```

---

## Embeddings Deep Dive

### What Makes Good Embeddings?

**Key Properties**:

1. **Semantic Similarity**:
```
"dog" and "puppy" → Close in vector space
"dog" and "banana" → Far apart
```

2. **Context Awareness**:
```
"bank" (financial) vs "bank" (river)
→ Different embeddings based on surrounding context
```

3. **Dimensionality**:
```
- Lower dimensions (384): Faster, less storage, slightly less accurate
- Higher dimensions (1536): Slower, more storage, more accurate
```

### How Embedding Models are Trained

**Training Process**:
```
1. Contrastive Learning:
   - Similar texts should have similar embeddings
   - Dissimilar texts should have different embeddings

2. Training Data:
   - Pairs of related texts
   - Question-answer pairs
   - Paraphrase pairs

3. Objectives:
   - Maximize similarity for positive pairs
   - Minimize similarity for negative pairs
```

**Example Training Pair**:
```
Positive Pair:
- "How do I reset my password?"
- "Password reset instructions"
→ Should have high similarity

Negative Pair:
- "How do I reset my password?"
- "Shipping policy"
→ Should have low similarity
```

### Choosing an Embedding Model

**Factors to Consider**:

1. **Task Type**:
   - General purpose: OpenAI, Sentence Transformers
   - Domain-specific: Bio-medical, legal (specialized models)
   - Multilingual: Use multilingual models

2. **Cost**:
   - Free Open Source: Sentence Transformers, Hugging Face models
   - Paid API: OpenAI ($0.0001 per 1K tokens), Cohere

3. **Performance Needs**:
   - Real-time: Smaller models (384d)
   - Batch processing: Larger models (1536d)

4. **Quality**:
```
MTEB Benchmark (Massive Text Embedding Benchmark):
- text-embedding-3-large: Score ~65
- all-MiniLM-L6-v2: Score ~58
- Higher = better
```

### Practical Embedding Example

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample texts
texts = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "The cat sat on the mat",
    "Python is popular for data science"
]

# Create embeddings
embeddings = model.encode(texts)

# Calculate similarity matrix
similarity_matrix = cosine_similarity(embeddings)

print("Similarity between texts:")
for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        print(f"{i} - {j}: {similarity_matrix[i][j]:.3f}")
        print(f"  '{texts[i]}'")
        print(f"  '{texts[j]}'")
        print()

# Output:
# 0 - 1: 0.612 (Both about programming, moderate similarity)
# 0 - 2: 0.234 (Unrelated, low similarity)
# 0 - 3: 0.892 (Both about Python, high similarity!)
# 1 - 2: 0.198 (Unrelated, low similarity)
# 1 - 3: 0.524 (Both programming, some similarity)
# 2 - 3: 0.187 (Unrelated, low similarity)
```

---

## Vector Databases

### Why Specialized Vector Databases?

**Traditional Database**:
```sql
SELECT * FROM documents
WHERE title LIKE '%AI%'
→ Keyword matching, exact or fuzzy
```

**Vector Database**:
```python
results = db.similarity_search(
    query_vector=query_embedding,
    top_k=10
)
→ Semantic matching, finds conceptually similar content
```

**Performance**:
- Searching 1M vectors with traditional search: seconds to minutes
- With specialized vector DB (HNSW): milliseconds

### Popular Vector Databases Comparison

#### Pinecone

**Type**: Fully managed cloud service

**Pros**:
- Zero infrastructure management
- Auto-scaling
- High availability
- Simple API

**Cons**:
- Paid only (no free tier for production)
- Vendor lock-in

**Pricing**: $70/month for 100K vectors, then $0.70 per additional 100K

**Use When**: You want simplicity and don't mind cost

**Example**:
```python
import pinecone

pinecone.init(api_key="your-key", environment="us-west1-gcp")
index = pinecone.Index("my-index")

# Upsert vectors
index.upsert(vectors=[
    ("id1", [0.1, 0.2, 0.3, ...], {"text": "doc1"}),
    ("id2", [0.4, 0.5, 0.6, ...], {"text": "doc2"})
])

# Query
results = index.query(
    vector=[0.15, 0.25, 0.35, ...],
    top_k=5,
    include_metadata=True
)
```

#### Chroma

**Type**: Open source, embedded database

**Pros**:
- Free and open source
- Very easy to use
- Great for prototyping
- Python-first API
- Embedded (no separate server needed)

**Cons**:
- Less scalable than enterprise solutions
- Fewer advanced features

**Use When**: Local development, small-scale projects, learning

**Example**:
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_collection")

# Add documents
collection.add(
    documents=["This is document 1", "This is document 2"],
    metadatas=[{"source": "web"}, {"source": "pdf"}],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_texts=["search query"],
    n_results=5
)
```

#### Weaviate

**Type**: Open source with cloud option

**Pros**:
- Feature-rich
- Hybrid search (vector + keyword)
- GraphQL API
- Self-hosted or cloud
- Good documentation

**Cons**:
- More complex setup
- Steeper learning curve

**Use When**: Production scale, need hybrid search, want flexibility

#### Qdrant

**Type**: Open source, Rust-based

**Pros**:
- Very fast (Rust performance)
- Rich filtering capabilities
- Good for large scale
- Active development
- Docker-friendly

**Use When**: High performance needs, large datasets

#### FAISS (Facebook AI Similarity Search)

**Type**: Library (not a full database)

**Pros**:
- Extremely fast
- CPU and GPU support
- Battle-tested (Meta uses it)
- Many index types

**Cons**:
- Not a full database (no persistence by default)
- Requires more code
- No built-in metadata storage

**Use When**: Maximum performance, willing to handle persistence yourself

### Vector DB Operations

**Create Index**:
```python
# Different indexing strategies
- Flat: Exact search, slow but accurate
- HNSW: Fast approximate search
- IVF: Partition space, good  for large datasets
```

**Add Vectors**:
```python
db.add(
    vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    ids=["doc1", "doc2"],
    metadata=[{"type": "pdf"}, {"type": "web"}]
)
```

**Search**:
```python
results = db.search(
    query_vector=[0.15, 0.25, ...],
    top_k=10,
    filter={"type": "pdf"}  # Metadata filtering
)
```

**Update**:
```python
db.update(
    id="doc1",
    vector=[0.11, 0.21, ...],
    metadata={"type": "pdf", "updated": "2024-01-01"}
)
```

**Delete**:
```python
db.delete(ids=["doc1", "doc2"])
```

---

## Complete RAG Workflow

### End-to-End Example: Customer Support Bot

Let me walk through building a complete RAG system for a customer support chatbot.

#### Step 1: Prepare Knowledge Base

**Source Documents**:
```
product_docs/
├── getting_started.pdf
├── api_reference.md
├── troubleshooting.md
├── faq.md
└── release_notes.md
```

**Load Documents**:
```python
from langchain.document_loaders import DirectoryLoader, PDFLoader, TextLoader

# Load all documents
loader = DirectoryLoader(
    "product_docs/",
    glob="**/*.*",
    loader_cls=TextLoader  # or PDFLoader for PDFs
)

documents = loader.load()
print(f"Loaded {len(documents)} documents")
```

#### Step 2: Chunk Documents

**Why Chunk?**
- Embedding models have token limits (512-8192 tokens)
- Smaller chunks = more precise retrieval
- Larger chunks = more context

**Chunking Strategy**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Characters per chunk
    chunk_overlap=50,  # Overlap to preserve context
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Split priority
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Example chunk
print(chunks[0].page_content)
print(chunks[0].metadata)
```

**Output**:
```
Created 245 chunks from 15 documents

Chunk example:
"To get started with our API, you first need to obtain an API key. 
Visit the developer portal and create a new application. Once created, 
you'll receive your API key which you should store securely..."

Metadata: {'source': 'getting_started.pdf', 'page': 1}
```

#### Step 3: Create Embeddings and Index

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Initialize embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # Or use Sentence Transformers for free
)

# Create vector store and index chunks
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Save to disk
)

print("Indexed all chunks into vector database")
```

#### Step 4: Set Up Retriever

```python
# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" for diversity
    search_kwargs={"k": 3}  # Return top 3 chunks
)

# Test retrieval
query = "How do I reset my API key?"
retrieved_docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(retrieved_docs):
    print(f"\nRetrieved Doc {i+1}:")
    print(doc.page_content[:200])
    print(f"Source: {doc.metadata['source']}")
```

#### Step 5: Build RAG Chain

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Custom prompt template
prompt_template = """Use the following pieces of context to answer the question. 
If you don't know the answer, say that you don't know, don't make up an answer.
Always cite which document you got the information from.

Context:
{context}

Question: {question}

Helpful Answer with Source Citation:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",  # "stuff", "map_reduce", "refine", or "map_rerank"
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
```

#### Step 6: Query the System

```python
# User query
question = "How do I reset my API key?"

# Get answer
result = qa_chain({"query": question})

print("Answer:", result['result'])
print("\nSources:")
for doc in result['source_documents']:
    print(f"- {doc.metadata['source']}")
```

**Output**:
```
Answer: To reset your API key, go to the developer portal, navigate to your 
application settings, and click "Regenerate API Key". You'll need to update 
your application with the new key within 24 hours. (Source: getting_started.pdf)

Sources:
- getting_started.pdf
- troubleshooting.md
- faq.md
```

### Complete Working Code

```python
# complete_rag_example.py
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

def build_rag_system(docs_folder, persist_dir="./chroma_db"):
    """Build complete RAG system"""
    
    # 1. Load documents
    print("Loading documents...")
    loader = DirectoryLoader(docs_folder, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    # 2. Chunk documents
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # 3. Create embeddings and vector store
    print("Creating embeddings and indexing...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    # 4. Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 5. Create RAG chain
    prompt_template = """Answer based on context. Cite sources.

Context: {context}
Question: {question}
Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print("RAG system ready!")
    return qa_chain

def query_rag(qa_chain, question):
    """Query the RAG system"""
    result = qa_chain({"query": question})
    print(f"\nQ: {question}")
    print(f"A: {result['result']}")
    print("\nSources:")
    for doc in result['source_documents']:
        print(f"  - {doc.metadata.get('source', 'Unknown')}")
    return result

# Usage
if __name__ == "__main__":
    # Build system
    qa_system = build_rag_system("./product_docs")
    
    # Ask questions
    query_rag(qa_system, "How do I get started with the API?")
    query_rag(qa_system, "What are the rate limits?")
    query_rag(qa_system, "How do I handle errors?")
```

---

## RAG vs Fine-Tuning

### When to Use RAG

✅ **Dynamic, frequently updated information**:
- News articles
- Product documentation
- Company policies
- Real-time data

✅ **Large, diverse knowledge base**:
- Wikipedia-scale content
- Enterprise document libraries
- Research paper collections

✅ **Need source attribution**:
- Legal compliance
- Medical advice
- Financial recommendations
- Academic research

✅ **Quick deployment**:
- Days instead of weeks
- No GPU training required
- Easier to iterate

✅ **Cost-effective updates**:
- Add new docs: minutes, $0.01
- Fine-tuning: hours, $100+

### When to Use Fine-Tuning

✅ **Specific behavior or style**:
- Writing tone (formal, casual, brand voice)
- Response format
- Specific task optimization

✅ **Small, stable knowledge**:
- Doesn't change frequently
- Fits in model parameters

✅ **Latency-sensitive**:
- No retrieval overhead
- Faster response times
- Real-time applications

✅ **Specialized tasks**:
- Classification
- Named entity recognition
- Structured output generation

### Hybrid Approach (Best of Both Worlds)

```
Fine-tuned Model    +    RAG
     ↓                    ↓
  Task/Style         Knowledge
  Optimization       Retrieval
     ↓                    ↓
        Combined = Optimal System
```

**Example**:
- Fine-tune for: Medical terminology, professional tone
- RAG for: Latest treatment guidelines, research papers
- Result: Professional medical assistant with current knowledge

---

## Hands-On Implementation

### Beginner Project: Personal Knowledge Base

Let's build a RAG system for your personal notes and documents!

**Step 1: Install Dependencies**
```bash
pip install langchain openai chromadb tiktoken sentence-transformers
# or use requirements.txt
```

**Step 2: Prepare Your Documents**
```
my_knowledge_base/
├── notes/
│   ├── AI_notes.md
│   └── Python_notes.md
├── books/
│   └── summary.pdf
└── articles/
    └── saved_articles.txt
```

**Step 3: Simple RAG Script**
```python
# simple_rag.py
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  # Free!
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Configuration
DOCS_FOLDER = "./my_knowledge_base"
CHROMA_DB = "./my_chroma_db"

# Load documents
loader = DirectoryLoader(DOCS_FOLDER, glob="**/*.md")
docs = loader.load()

# Chunk
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Use free embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Index
db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB)

# RAG chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    retriever=db.as_retriever()
)

# Query
while True:
    question = input("\nAsk a question (or 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    
    answer = qa.run(question)
    print(f"\nAnswer: {answer}")
```

**Run It**:
```bash
python simple_rag.py
# Ask: "What are my notes about neural networks?"
```

---

## Common Challenges

### Challenge 1: Chunking Too Large or Too Small

**Problem**: 
- Too large → Embedding loses precision
- Too small → Missing context

**Solution**:
```python
# Test different chunk sizes
for chunk_size in [200, 500, 1000]:
    chunks = split_documents(docs, chunk_size)
    # Evaluate retrieval quality
    # Choose optimal size
```

**Rule of Thumb**: 500-1000 characters with 50-100 overlap

### Challenge 2: Poor Retrieval Quality

**Problem**: Retrieved documents not relevant

**Solutions**:
1. **Better embeddings**: Try different models
2. **Hybrid search**: Combine semantic + keyword
3. **Re-ranking**: Two-stage retrieval
4. **Query expansion**: Rephrase query multiple ways

### Challenge 3: Context Window Limits

**Problem**: Too many retrieved docs exceed LLM context

**Solutions**:
```python
# 1. Retrieve more, return less (re-rank)
retriever.search_kwargs = {"k": 10}  # Retrieve 10
# Re-rank, keep top 3

# 2. Summarize retrieved docs
for doc in retrieved:
    doc.content = llm.summarize(doc.content)

# 3. Use map-reduce
chain_type = "map_reduce"  # instead of "stuff"
```

### Challenge 4: Expensive Embedding Costs

**Problem**: Paying per API call adds up

**Solutions**:
1. **Use open-source models**: sentence-transformers (free)
2. **Cache embeddings**: Don't recompute
3. **Batch processing**: Reduce API overhead

```python
# Use free model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache
import joblib
cache_file = "embeddings_cache.pkl"
if os.path.exists(cache_file):
    embeddings = joblib.load(cache_file)
else:
    embeddings = model.encode(texts)
    joblib.dump(embeddings, cache_file)
```

---

## Key Takeaways

✅ **RAG** = Retrieval Augmented Generation - enhances LLMs with external, up-to-date knowledge

✅ **Solves 5 key problems**: Knowledge cutoff, hallucinations, domain expertise, expensive updates, source verification

✅ **Architecture**: Documents → Chunks → Embeddings → Vector DB → Retrieval → LLM Generation

✅ **Embeddings** convert text to vectors enabling semantic search

✅ **Vector databases** (Pinecone, Chroma, Weaviate, Qdrant) enable fast similarity search

✅ **Complete workflow**: Load → Chunk → Embed → Index → Query → Retrieve → Generate

✅ **RAG vs Fine-tuning**: RAG for knowledge, fine-tuning for behavior/style

✅ **Tools**: LangChain and LlamaIndex make RAG implementation straightforward

✅ **Production considerations**: Chunking strategy, embedding model choice, retrieval quality, cost optimization

---

## What's Next?

Ready to dive deeper?

📚 **[Intermediate RAG Implementation](02-intermediate-rag-implementation.md)**  
Advanced chunking, hybrid search, re-ranking, production architecture

📚 **[Advanced RAG Optimization](03-advanced-rag-optimization.md)**  
Query optimization, agentic RAG, multi-modal, evaluation frameworks

📚 **[Prompt Engineering](../Prompt-Engineering/01-beginner-prompt-basics.md)**  
Optimize RAG prompts for better answers

📚 **[Vector Database Deep Dive](../Resources/references.md)**  
Technical details on vector search algorithms

---

## Practice Exercises

1. **Build Your First RAG**: Use the simple_rag.py script with your own documents
2. **Compare Embeddings**: Test OpenAI vs Sentence Transformers on same data
3. **Chunk Size Experiment**: Try 200, 500, 1000 character chunks, compare results
4. **Test Retrieval**: Query your RAG system, verify retrieved documents are relevant
5. **Add Metadata Filtering**: Filter by document type, date, or custom tags

---

**Congratulations!** 🎉 You now understand RAG fundamentals and can build knowledge-augmented AI systems!

---

*Next: [Intermediate RAG Implementation](02-intermediate-rag-implementation.md)*
