# 🔍 Retrieval-Augmented Generation (RAG)

Master the art of building intelligent systems that combine the power of LLMs with your own data.

## 📚 Table of Contents

1. [What is RAG?](#what-is-rag)
2. [Core Concepts](#core-concepts)
3. [Building a RAG System](#building-a-rag-system)
4. [Vector Databases](#vector-databases)
5. [Advanced Patterns](#advanced-patterns)
6. [Production Best Practices](#best-practices)
7. [Resources](#resources)

## What is RAG?

**Retrieval-Augmented Generation** enhances LLMs by retrieving relevant information from external knowledge bases before generating responses.

### Why RAG?

**Problems with vanilla LLMs:**
- ❌ Knowledge cutoff dates
- ❌ No access to private/proprietary data
- ❌ Hallucinations on unfamiliar topics
- ❌ Can't cite sources

**RAG Solutions:**
- ✅ Up-to-date information
- ✅ Access to your custom data
- ✅ Grounded, factual responses
- ✅ Source attribution

### How RAG Works

```
1. User asks a question
2. Question is converted to vector embedding
3. Similar documents are retrieved from vector DB
4. Retrieved docs + question are sent to LLM
5. LLM generates answer based on context
```

## Core Concepts

### 1. Embeddings

Converting text into numerical vectors that capture semantic meaning.

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    """Convert text to vector embedding"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Example
text = "Machine learning is a subset of artificial intelligence"
embedding = get_embedding(text)
print(f"Embedding dimension: {len(embedding)}")  # 1536
```

### 2. Similarity Search

Finding similar vectors using cosine similarity.

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Compare documents
query_embedding = get_embedding("What is AI?")
doc1_embedding = get_embedding("AI is the simulation of human intelligence")
doc2_embedding = get_embedding("Pizza is a popular Italian food")

sim1 = cosine_similarity(query_embedding, doc1_embedding)
sim2 = cosine_similarity(query_embedding, doc2_embedding)

print(f"Query-Doc1 similarity: {sim1:.4f}")  # Higher
print(f"Query-Doc2 similarity: {sim2:.4f}")  # Lower
```

### 3. Chunking Strategies

Breaking documents into optimal sizes for retrieval.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Recursive character splitter (recommended)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

text = """[Your long document here...]"""
chunks = splitter.split_text(text)

print(f"Split into {len(chunks)} chunks")
```

**Other chunking strategies:**

```python
# Sentence-based
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)

# Semantic chunking
from langchain.text_splitter import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings

splitter = SemanticChunker(OpenAIEmbeddings())

# Token-based
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
```

## Building a RAG System

### Basic RAG Pipeline

```python
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Step 1: Load documents
loader = DirectoryLoader('./docs', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Step 3: Create embeddings
embeddings = OpenAIEmbeddings()

# Step 4: Create vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Step 5: Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Step 6: Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Step 7: Query
query = "What are the key features of the product?"
result = qa_chain({"query": query})

print(f"Answer: {result['result']}")
print(f"\nSources: {[doc.metadata for doc in result['source_documents']]}")
```

### Custom RAG Implementation

```python
from openai import OpenAI
import chromadb

client = OpenAI()
chroma_client = chromadb.Client()

class SimpleRAG:
    def __init__(self, collection_name="documents"):
        self.collection = chroma_client.create_collection(collection_name)
        
    def add_documents(self, documents, metadatas=None):
        """Add documents to vector store"""
        # Generate embeddings
        embeddings = [
            client.embeddings.create(input=doc, model="text-embedding-3-small").data[0].embedding
            for doc in documents
        ]
        
        # Store in Chroma
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in documents],
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
    
    def retrieve(self, query, k=3):
        """Retrieve relevant documents"""
        query_embedding = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        return results['documents'][0]
    
    def query(self, question, k=3):
        """RAG query: retrieve + generate"""
        # Retrieve relevant docs
        relevant_docs = self.retrieve(question, k=k)
        
        # Create context
        context = "\n\n".join(relevant_docs)
        
        # Generate answer
        messages = [
            {"role": "system", "content": "Answer based on the provided context. If you can't find the answer in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        )
        
        return response.choices[0].message.content

# Usage
rag = SimpleRAG()

# Add documents
docs = [
    "Python is a high-level programming language.",
    "Machine learning is a type of AI that learns from data.",
    "RAG combines retrieval with generation for better answers."
]

rag.add_documents(docs)

# Query
answer = rag.query("What is machine learning?")
print(answer)
```

## Vector Databases

### 1. Chroma (Local, Simple)

```python
import chromadb
from chromadb.config import Settings

# Initialize
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./my_chroma_db"
))

collection = client.create_collection("my_docs")

# Add documents
collection.add(
    documents=["doc1 text", "doc2 text"],
    metadatas=[{"source": "web"}, {"source": "pdf"}],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_texts=["search query"],
    n_results=2
)
```

### 2. Pinecone (Cloud, Scalable)

```python
import pinecone

# Initialize
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# Create index
index_name = "my-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine"
    )

index = pinecone.Index(index_name)

# Upsert vectors
index.upsert(vectors=[
    ("id1", embedding1, {"text": "doc1"}),
    ("id2", embedding2, {"text": "doc2"})
])

# Query
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)
```

### 3. Weaviate (Feature-rich)

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Define schema
schema = {
    "classes": [{
        "class": "Document",
        "vectorizer": "text2vec-openai",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "source", "dataType": ["string"]}
        ]
    }]
}

client.schema.create(schema)

# Add data
client.data_object.create(
    class_name="Document",
    data_object={
        "content": "Machine learning is amazing",
        "source": "blog"
    }
)

# Query
result = client.query.get("Document", ["content", "source"]) \
    .with_near_text({"concepts": ["AI and ML"]}) \
    .with_limit(5) \
    .do()
```

### 4. FAISS (High Performance)

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Create FAISS index
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    texts=["doc1", "doc2", "doc3"],
    embedding=embeddings
)

# Save locally
vectorstore.save_local("faiss_index")

# Load
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Search
docs = vectorstore.similarity_search("query", k=3)
```

## Advanced Patterns

### 1. Multi-Query RAG

Generate multiple query variations for better retrieval.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=OpenAI(temperature=0)
)

# Automatically generates query variations
docs = retriever.get_relevant_documents("What is machine learning?")
```

### 2. Contextual Compression

Compress retrieved documents to only relevant parts.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

docs = compression_retriever.get_relevant_documents(query)
```

### 3. Hybrid Search

Combine semantic search with keyword search.

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Keyword-based retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 2

# Semantic retriever
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

docs = ensemble_retriever.get_relevant_documents(query)
```

### 4. Parent Document Retriever

Retrieve small chunks but return larger parent documents.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Storage for parent documents
store = InMemoryStore()

# Small chunks for search
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Larger chunks to return
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

retriever.add_documents(documents)
```

### 5. Self-Query Retriever

Convert natural language to metadata filters.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source of the document",
        type="string",
    ),
    AttributeInfo(
        name="date",
        description="The date the document was created",
        type="string",
    ),
]

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Technical documentation",
    metadata_field_info=metadata_field_info
)

# Query: "Documents from 2023 about Python"
# Automatically converts to metadata filter
docs = retriever.get_relevant_documents(query)
```

## Best Practices

### 1. Optimal Chunk Size

```python
def evaluate_chunk_sizes(documents, query, chunk_sizes=[500, 1000, 1500]):
    """Test different chunk sizes"""
    results = {}
    
    for size in chunk_sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=size // 5
        )
        chunks = splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        qa = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
        
        answer = qa.run(query)
        results[size] = answer
    
    return results
```

### 2. Metadata Management

```python
# Rich metadata for better filtering
documents = [
    Document(
        page_content="content here",
        metadata={
            "source": "report.pdf",
            "page": 5,
            "date": "2024-01-15",
            "author": "John Doe",
            "section": "Introduction",
            "topic": "Machine Learning"
        }
    )
]

# Query with metadata filters
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {
            "date": {"$gte": "2024-01-01"},
            "topic": "Machine Learning"
        }
    }
)
```

### 3. Evaluation

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Prepare evaluation dataset
eval_dataset = {
    "question": ["What is ML?", "..."],
    "answer": ["ML is...", "..."],
    "contexts": [[retrieved_docs1], [retrieved_docs2]],
    "ground_truth": ["Expected answer", "..."]
}

# Evaluate
results = evaluate(
    eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)

print(results)
```

### 4. Response Citations

```python
def rag_with_citations(query):
    """RAG that includes source citations"""
    docs = retriever.get_relevant_documents(query)
    
    context = "\n\n".join([
        f"[{i+1}] {doc.page_content} (Source: {doc.metadata.get('source', 'Unknown')})"
        for i, doc in enumerate(docs)
    ])
    
    prompt = f"""
    Answer the question based on the context below. 
    Include citation numbers [1], [2], etc. in your answer.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    response = llm(prompt)
    
    return {
        "answer": response,
        "sources": [doc.metadata for doc in docs]
    }
```

## Production Deployment

### Scaling Considerations

```python
# Connection pooling for vector DB
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    db_url,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)

# Caching for frequent queries
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_retrieval(query_hash):
    return retriever.get_relevant_documents(query)

# Async processing
import asyncio

async def async_rag(query):
    docs = await async_retriever.aget_relevant_documents(query)
    response = await async_llm.agenerate(docs)
    return response
```

## Resources

### 📖 Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- "In-Context Retrieval-Augmented Language Models"
- "REALM: Retrieval-Augmented Language Model Pre-Training"

### 🛠️ Tools & Frameworks
- **LangChain**: RAG framework
- **LlamaIndex**: Data framework for LLMs
- **Haystack**: NLP framework with RAG
- **txtai**: Embeddings database
- **Weaviate/Pinecone/Chroma**: Vector databases

### 🎓 Tutorials
- LangChain RAG Tutorial
- Building Production RAG Systems
- Vector Database Comparison Guide

---

**Next Steps:**
- Explore [LLM section](../llms/README.md) for model details
- Check [tech resources](../tech/README.md) for deployment
- Try [example projects](../projects/README.md)
