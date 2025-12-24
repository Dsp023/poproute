# Intermediate LLM Architecture

## Table of Contents
1. [Transformer Architecture Deep Dive](#transformer-architecture-deep-dive)
2. [Attention Mechanisms](#attention-mechanisms)
3. [Pre-training Approaches](#pre-training-approaches)
4. [Tokenization](#tokenization)
5. [Model Scaling and Architecture Choices](#model-scaling-and-architecture-choices)
6. [Context Windows](#context-windows)
7. [Training Infrastructure](#training-infrastructure)
8. [Evaluation](#evaluation)

---

## Transformer Architecture Deep Dive

### Self-Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Components**:
- **Q (Query)**: What to look for
- **K (Key)**: What to match against
- **V (Value)**: Actual content
- **d_k**: Key dimension for scaling

**Example**: Processing "The cat sat on the mat"
- Token "sat" might attend heavily to "cat" (subject) and "mat" (complement)

### Multi-Head Attention

**Multiple attention mechanisms in parallel**

Different heads can specialize:
- Syntactic relationships
- Semantic meaning
- Long-range dependencies

**GPT-3**: 96 attention heads per layer
**BERT-base**: 12 heads per layer

### Positional Encoding

**Problem**: Attention has no inherent position awareness

**Solutions**:
1. **Sinusoidal** (original Transformer): Fixed mathematical functions
2. **Learned** (GPT, BERT): Trainable embeddings per position
3. **Relative** (T5): Encode distances between tokens

### Feed-Forward Networks

Two linear transformations with activation:
```
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
```

**Typically expands 4x** then projects back

**Contains ~2/3 of transformer parameters**

---

## Attention Mechanisms

### Scaled Dot-Product

**Scaling by √d_k prevents saturation** in softmax

### Causal Masking

**For autoregressive models** (GPT):
- Can only attend to previous tokens
- Prevents "cheating" by seeing future

### Cross-Attention

**Encoder-Decoder**: Decoder attends to encoder
- Used in translation, image captioning

### Sparse Attention

**For long sequences**, reduce O(n²) complexity:
- **Local**: Sliding window
- **Strided**: Every k-th token
- **Global + Local**: Longformer, BigBird

**Models**: Longformer, BigBird, Reformer

---

## Pre-training Approaches

### Causal Language Modeling (CLM)

**Predict next token**: GPT approach

```
Input: "The cat sat on"
Predict: "the"
```

**Advantages**: Natural for generation
**Used by**: GPT series, LLaMA, Mistral

### Masked Language Modeling (MLM)

**Predict masked tokens**: BERT approach

```
Input: "The [MASK] sat on the mat"
Predict: "cat"
```

**Advantages**: Bidirectional context
**Used by**: BERT, RoBERTa, DeBERTa

### Span Corruption

**Predict masked spans**: T5 approach

```
"The cat <X> on the <Y>" → "<X> sat <Y> mat"
```

**Text-to-text framework**: All tasks as generation

### Contrastive Learning

**Match related items**: CLIP approach
- Image + correct caption = high similarity
- Image + wrong caption = low similarity

**Enables zero-shot transfer**

---

## Tokenization

### Subword Tokenization

**Balance between words and characters**

**Methods**:
1. **BPE** (Byte-Pair Encoding): GPT, RoBERTa
   - Merge frequent character pairs iteratively
   
2. **WordPiece**: BERT
   - Similar to BPE, different merging criterion
   
3. **SentencePiece**: T5, LLaMA
   - Language-agnostic, works on raw bytes

**Vocabulary sizes**: 30K-50K typical

### Special Tokens

- `[CLS]`, `<s>`: Start token
- `[SEP]`, `</s>`: Separator/end
- `[MASK]`: Masked position
- `[PAD]`, `<pad>`: Padding

---

## Model Scaling and Architecture Choices

### Scaling Laws

**Kaplan et al. (2020)**: Performance scales predictably with:
- Model size (parameters)
- Dataset size
- Compute budget

**Power law relationships** enable predicting performance

**Key insight**: Bigger models trained on more data perform better

### Encoder vs Decoder vs Encoder-Decoder

**Encoder-only** (BERT):
- Bidirectional
- Best for understanding tasks
- Classification, NER, Q&A

**Decoder-only** (GPT):
- Unidirectional (causal)
- Best for generation
- Text completion, chat

**Encoder-Decoder** (T5):
- Both components
- Best for transformation tasks
- Translation, summarization

### Architecture Variations

**Parameter sharing** (ALBERT):
- Share weights across layers
- Fewer parameters, same depth

**Sparse models** (Mixtral):
- Mixture of Experts (MoE)
- Activate subset per token

**Efficient attention**:
- Flash Attention: GPU optimization
- Linear attention: O(n) complexity

---

## Context Windows

### What is Context Window?

**Maximum input + output length** model can handle

**Examples**:
- GPT-3: 4K tokens (2048 input + 2048 output)
- GPT-4: 8K, 32K, 128K variants
- Claude 2: 100K → 200K
- Gemini 1.5: 1M tokens

### Challenges with Long Context

**Computational**: O(n²) attention complexity

**Lost in the middle**: Models struggle to use middle of very long contexts

**Solutions**:
- Sparse attention patterns
- Compression/summarization
- Retrieval (RAG) instead of long context

---

## Training Infrastructure

### Hardware Requirements

**GPUs**: A100, H100 (NVIDIA)
- A100: 40GB/80GB memory
- H100: 80GB memory, 2-3x faster

**Multi-GPU training**: Distribute across 100s-1000s of GPUs

**Example**: GPT-3 trained on 10,000 V100 GPUs

### Distributed Training

**Techniques**:
1. **Data Parallelism**: Different batches on different GPUs
2. **Model Parallelism**: Split model across GPUs
3. **Pipeline Parallelism**: Split layers across GPUs
4. **Tensor Parallelism**: Split individual layers

**Frameworks**: DeepSpeed, Megatron, FSDP

### Training Costs

**GPT-3**: ~$4-12 million in compute

**LLaMA 65B**: ~$2-3 million

**Smaller models** (7B): $10K-100K

---

## Evaluation

### Perplexity

**Measures how well model predicts text**

Lower = better

```
Perplexity = exp(average negative log-likelihood)
```

**Use**: Language modeling quality

### Downstream Task Performance

**Benchmarks**:
- **GLUE/SuperGLUE**: NLP tasks (classification, NER, etc.)
- **MMLU**: Massive Multitask Language Understanding (57 tasks)
- **HumanEval**: Code generation
- **HellaSwag**: Commonsense reasoning
- **TruthfulQA**: Factual accuracy

### Zero-Shot vs Few-Shot

**Zero-shot**: No task-specific examples
**Few-shot**: Provide examples in prompt
**Fine-tuned**: Trained on task

**LLMs excel at zero/few-shot** compared to smaller models

### Human Evaluation

**Critical for open-ended generation**:
- Helpfulness
- Harmlessness
- Accuracy
- Coherence

**Costly but necessary** for conversational AI

---

## Key Takeaways

✅ **Transformer architecture**: Self-attention + FFN + residual + layernorm

✅ **Attention**: Multi-head, causal masking, cross-attention variations

✅ **Pre-training**: CLM (GPT), MLM (BERT), span corruption (T5), contrastive (CLIP)

✅ **Tokenization**: Subword methods (BPE, WordPiece, SentencePiece) balance vocab and sequence length

✅ **Scaling laws**: Bigger models + more data + more compute = better performance

✅ **Architecture choices**: Encoder-only, decoder-only, or encoder-decoder based on task

✅ **Context windows**: Vary from 4K to 1M tokens; longer is better but more expensive

✅ **Training**: Requires massive compute (1000s of GPUs), distributed training techniques

✅ **Evaluation**: Perplexity, benchmarks (MMLU, HumanEval), human evaluation

---

## What's Next?

Ready for advanced topics? Continue to:
- [Advanced LLM Fine-Tuning and Deployment](03-advanced-llm-fine-tuning.md)
- [RAG Systems](../RAG-Systems/01-beginner-rag-fundamentals.md)
- [Prompt Engineering](../Prompt-Engineering/01-beginner-prompt-basics.md)

---

*Previous: [LLM Basics](01-beginner-llm-basics.md) | Next: [Advanced LLM Fine-Tuning](03-advanced-llm-fine-tuning.md)*
