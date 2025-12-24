# Advanced Transformer Variants

## BERT | GPT | T5 | Vision Transformers | Efficient Transformers | Future Directions

### BERT (Bidirectional Encoder)

**Pre-training**: Masked LM + Next Sentence Prediction

**Variants**:
- RoBERTa: Improved BERT
- ALBERT: Parameter sharing
- DeBERTa: Disentangled attention

**Best for**: Understanding tasks

### GPT (Generative Pre-trained Transformer)

**Autoregressive**: Predict next token

**Evolution**:
- GPT-1: 117M parameters
- GPT-2: 1.5B
- GPT-3: 175B
- GPT-4: Multimodal, 1.7T (estimated)

**Best for**: Generation tasks

### T5 (Text-to-Text Transfer Transformer)

**Unified framework**: All tasks as text generation

**Training**: Span corruption

**Variants**: FLAN-T5 (instruction-tuned)

### Vision Transformers (ViT)

**Apply transformers to images**

**Process**: Patch embeddings + transformer encoder

**Result**: Matches CNNs with sufficient data

**Extensions**: Swin, DeiT

### Efficient Transformers

**Problem**: O(n²) attention complexity

**Solutions**:
- Sparse attention (Longformer, BigBird)
- Linear attention (Performer)
- Recurrent (RWKV)
- State space models (Mamba)

**Flash Attention**: GPU optimization

### Mixture of Experts (MoE)

**Sparse activation**: Use subset of parameters per token

**Examples**: Switch Transformer, Mixtral

**Benefit**: Large capacity, efficient inference

---

*Previous: [Intermediate Transformers](02-intermediate-transformer-architecture.md)*
