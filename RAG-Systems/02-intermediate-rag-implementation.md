# Intermediate RAG Implementation

## Table of Contents
1. [Chunking Strategies](#chunking-strategies)
2. [Vector Database Selection](#vector-database-selection)
3. [Retrieval Techniques](#retrieval-techniques)
4. [Prompt Engineering for RAG](#prompt-engineering-for-rag)
5. [Building Production RAG](#building-production-rag)
6. [Evaluation](#evaluation)

---

## Chunking Strategies

### Why Chunking Matters

**Problem**: Documents too long for embedding models (512-8192 token limits)

**Solution**: Split into chunks, embed separately

### Fixed-Size Chunking

**Simple approach**: Split by token/character count

```python
chunk_size = 500  # tokens
overlap = 50      # tokens for continuity
```

**Pros**: Simple, fast
**Cons**: May break mid-sentence, loses context

### Semantic Chunking

**Split by meaning**: Paragraphs, sections, topics

**Methods**:
- Sentence boundaries
- Paragraph boundaries
- Section headers
- Topic modeling

**Pros**: Preserves meaning
**Cons**: Variable chunk sizes

### Recursive Chunking

**LangChain approach**:
1. Try to split by paragraphs
2. If too large, split by sentences
3. If still too large, split by characters

**Balanced approach**

### Overlap Strategy

**Include overlap** between consecutive chunks

**Example**:
```
Chunk 1: tokens 0-500
Chunk 2: tokens 450-950 (50 token overlap)
```

**Benefit**: Preserves context across boundaries

---

## Vector Database Selection

### Comparison

| Database | Type | Best For |
|----------|------|----------|
| **Pinecone** | Managed | Production, ease of use |
| **Weaviate** | Open/Managed | Flexibility, features |
| **Chroma** | Open | Prototyping, local dev |
| **Qdrant** | Open | Performance, Rust-based |
| **Milvus** | Open | Scale, Kubernetes |

### Key Features

**Filtering**: Combine semantic + metadata search

**Hybrid Search**: keyword + vector search

**Multi-tenancy**: Separate data per user/org

**Persistence**: Data durability

---

## Retrieval Techniques

### Similarity Search

**Basic**: Top-K most similar vectors

```python
results = vector_db.similarity_search(query, k=5)
```

### Maximal Marginal Relevance (MMR)

**Balance**: Relevance + diversity

**Avoids**: Redundant results

**Formula**: Optimize relevance while penalizing similarity to already-selected docs

### Hybrid Search

**Combine**:
- Semantic search (vectors)
- Keyword search (BM25)

**Fusion**: RRF (Reciprocal Rank Fusion)

**Best for**: Precise term matching + semantic understanding

### Parent-Child Chunking

**Idea**: Retrieve small chunks, return larger parent

**Process**:
1. Index small chunks for retrieval
2. Store reference to parent document
3. Return parent for context

**Benefit**: Precise retrieval, broad context

### Hypothetical Document Embeddings (HyDE)

**Process**:
1. LLM generates hypothetical answer to query
2. Embed hypothetical answer
3. Search with that embedding

**Benefit**: Better matches actual answer distributions

---

## Prompt Engineering for RAG

### Prompt Template

```
Context:
{retrieved_documents}

Question: {user_query}

Instructions: Answer based on context. If unsure, say "I don't know."

Answer:
```

### Best Practices

**1. Clear Instructions**:
- Cite sources
- Admit uncertainty
- Stay within context

**2. Context Formatting**:
- Number documents
- Include metadata (title, date)
- Highlight relevance

**3. Few-Shot Examples**:
- Show desired format
- Demonstrate citation style

---

## Building Production RAG

### Architecture

```
User Query
    ↓
Query Processing (rewrite, expand)
    ↓
Retrieval (vector search + filtering)
    ↓
Re-ranking (optional)
    ↓
Context Assembly
    ↓
LLM Generation
    ↓
Post-processing (formatting, citations)
    ↓
Response to User
```

### Implementation with LangChain

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Setup
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=retriever,
    return_source_documents=True
)

# Query
result = qa_chain("What is RAG?")
```

---

## Evaluation

### Retrieval Metrics

**Recall@K**: % of relevant docs in top-K

**Precision@K**: % of top-K that are relevant

**MRR** (Mean Reciprocal Rank): Rank of first relevant result

### Generation Metrics

**Faithfulness**: Answer grounded in retrieved context

**Answer Relevance**: Addresses the question

**Context Relevance**: Retrieved docs relevant to query

### Tools

- **RAGAS**: RAG evaluation framework
- **LangSmith**: Tracing and evaluation
- **Custom evaluators**: LLM-as-judge

---

*Previous: [RAG Fundamentals](01-beginner-rag-fundamentals.md) | Next: [Advanced RAG Optimization](03-advanced-rag-optimization.md)*
