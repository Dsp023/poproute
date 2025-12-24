# Advanced LLM Fine-Tuning and Deployment

## Table of Contents
1. [Fine-Tuning Techniques](#fine-tuning-techniques)
2. [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
3. [RLHF and Alignment](#rlhf-and-alignment)
4. [Inference Optimization](#inference-optimization)
5. [Deployment Strategies](#deployment-strategies)
6. [Monitoring and Evaluation](#monitoring-and-evaluation)
7. [Building LLM Applications](#building-llm-applications)
8. [Advanced Topics](#advanced-topics)

---

## Fine-Tuning Techniques

### Full Fine-Tuning

**Process**: Update all model parameters on your dataset

**Advantages**:
- Maximum adaptation to target task
- Best performance potential

**Disadvantages**:
- Requires significant compute (GPU memory)
- Risk of catastrophic forgetting
- Expensive for large models

**When to Use**:
- Small models (< 1B parameters)
- Abundant compute resources
- Task very different from pre-training

**Example Use Case**: Fine-tune GPT-2 (1.5B) for specific domain

### Instruction Tuning

**Process**: Fine-tune on instruction-following examples

**Dataset Format**:
```
Instruction: Translate to French
Input: Hello, how are you?
Output: Bonjour, comment allez-vous?
```

**Models Using This**: FLAN-T5, Alpaca, Vicuna

**Benefits**:
- Improves zero-shot task performance
- Better instruction following
- Generalization across tasks

### Task-Specific Fine-Tuning

**Specialize for single task** (classification, NER, summarization)

**Typically adds task head** on top of base model

---

## Parameter-Efficient Fine-Tuning (PEFT)

### Why PEFT?

**Problem**: Full fine-tuning is expensive
- GPT-3 175B: Needs 350GB GPU memory
- Multiple task-specific copies wasteful

**Solution**: Update small subset of parameters

### LoRA (Low-Rank Adaptation)

**Key Idea**: Freeze base model, add trainable low-rank matrices

**Mathematics**:
```
W' = W + ΔW
where ΔW = BA (B: d×r, A: r×k)
r << min(d,k) (low rank)
```

**Example**: GPT-3 175B
- Full fine-tuning: 175B parameters
- LoRA: ~10M parameters (~0.006%)

**Advantages**:
- 3x memory reduction
- Faster training
- Multiple adapters for different tasks
- No inference latency

**Hyperparameters**:
- **r (rank)**: 4-64 typically (higher = more capacity)
- **α (scaling)**: Often 2r
- **Target modules**: Which layers to apply LoRA

**Implementations**: PEFT library (Hugging Face)

### QLoRA (Quantized LoRA)

**Combines LoRA with quantization**

**Process**:
1. Load base model in 4-bit quantization
2. Add LoRA adapters in full precision
3. Train adapters

**Breakthrough**: Fine-tune 65B model on single 48GB GPU

**Use Case**: Resource-constrained fine-tuning

### Prefix Tuning

**Add trainable "prefix" tokens to input**

**Frozen base model**, only prefix parameters trained

**Advantages**:
- Even fewer parameters than LoRA
- Task-specific prefixes

**Disadvantages**:
- Reduces effective context length
- Sometimes lower performance

### Adapter Layers

**Insert small trainable modules between transformer layers**

**Architecture**:
```
Layer → Adapter (down-project → nonlinear → up-project) → Layer
```

**Advantages**:
- Modular (swap adapters for different tasks)
- Parameter-efficient

### Prompt Tuning (Soft Prompts)

**Learn continuous prompt embeddings** (not discrete tokens)

**Extremely parameter-efficient**: Only prompt tokens updated

**Works well for large models** (10B+ parameters)

---

## RLHF and Alignment

### Reinforcement Learning from Human Feedback

**Goal**: Align model behavior with human preferences

**Three-Stage Process**:

#### Stage 1: Supervised Fine-Tuning (SFT)

**Train on high-quality demonstrations**:
```
Prompt: "Explain photosynthesis"
Response: [High-quality human-written explanation]
```

**Creates initial instruction-following model**

#### Stage 2: Reward Model Training

**Process**:
1. Generate multiple responses for same prompt
2. Humans rank responses (best to worst)
3. Train reward model to predict human preferences

**Reward Model**: Takes (prompt, response) → scalar score

**Dataset**: Comparison pairs
```
Prompt: "Explain photosynthesis"
Response A: [detailed, accurate]
Response B: [brief, less accurate]
Label: A > B
```

#### Stage 3: RL Fine-Tuning (PPO)

**Use reward model to fine-tune policy (LLM)**

**Algorithm**: Proximal Policy Optimization (PPO)

**Objective**: Maximize reward while staying close to SFT model

**Loss**:
```
L = E[reward(prompt, response)] - β * KL(π || π_SFT)
```
- β: Controls deviation from SFT model
- KL: Prevents model from drifting too far

**Challenges**:
- Reward hacking (exploiting reward model)
- Mode collapse (generating similar responses)
- Computational cost

### Constitutional AI (Anthropic's Approach)

**Alternative to RLHF**: Use AI feedback instead of human

**Process**:
1. Generate responses
2. AI critiques responses based on "constitution" (principles)
3. AI revises responses
4. Train on revised responses

**Advantages**:
- Scalable (no human labeling needed)
- Consistent principles
- Iterative improvement

### Direct Preference Optimization (DPO)

**Newer alternative to RLHF**: Skip reward model

**Directly optimize on preference data**

**Advantages**:
- Simpler (one stage instead of three)
- More stable training
- No reward model training needed

**Recent adoption**: Growing in popularity (2023-2024)

---

## Inference Optimization

### Model Quantization

**Reduce numerical precision**: 32-bit → 16-bit → 8-bit → 4-bit

**Methods**:

#### Post-Training Quantization (PTQ)
- Quantize after training
- No retraining needed
- Some accuracy loss

#### Quantization-Aware Training (QAT)
- Train with quantization in mind
- Better accuracy preservation
- More expensive

**Popular Formats**:
- **FP16** (half precision): 2x memory reduction, minimal accuracy loss
- **INT8**: 4x memory reduction, used in BERT, T5
- **4-bit (GPTQ, AWQ)**: 8x reduction, enables local LLM running

**Tools**: bitsandbytes, GPTQ, AWQ

### KV Cache Optimization

**Problem**: Autoregressive generation recomputes past keys/values

**Solution**: Cache key-value pairs for past tokens

**Memory Trade-off**: O(n) memory for O(1) computation per token

**Optimizations**:
- **Flash Attention**: Efficient attention computation
- **Multi-Query Attention**: Share keys/values across heads (PaLM)
- **Grouped-Query Attention**: Hybrid approach (LLaMA 2)

### Speculative Decoding

**Idea**: Use small "draft" model to predict tokens, verify with large model

**Process**:
1. Draft model generates K tokens quickly
2. Large model verifies in parallel
3. Accept correct tokens, reject others

**Speedup**: 2-3x with no quality loss

**Requirements**: Draft model must be much faster

### Batching and Serving

**Continuous Batching** (Orca, vLLM):
- Dynamic batching of requests
- Better GPU utilization
- Lower latency

**PagedAttention** (vLLM):
- Efficient KV cache memory management
- Inspired by OS virtual memory
- 2-4x throughput improvement

---

## Deployment Strategies

### Cloud API Deployment

**Providers**: OpenAI, Anthropic, Google, Cohere

**Advantages**:
- No infrastructure management
- Automatic scaling
- Latest models

**Disadvantages**:
- Recurring costs
- Data sent to third party
- Rate limits
- Vendor lock-in

**Use Cases**: Prototyping, low-volume applications

### Self-Hosted Deployment

#### Option 1: Cloud VMs (AWS, GCP, Azure)

**Setup**:
1. Provision GPU instances (A100, H100)
2. Deploy model with serving framework
3. Set up load balancing, monitoring

**Advantages**:
- Full control over data
- Customization
- Pay for what you use

**Disadvantages**:
- Infrastructure management
- Scaling complexity

#### Option 2: Specialized ML Platforms

**Platforms**: Hugging Face Inference Endpoints, Replicate, Together AI

**Advantages**:
- Easy deployment
- Automatic scaling
- Optimized infrastructure

#### Option 3: On-Premises

**GPU servers in your data center**

**Use Cases**: Strict data privacy, regulated industries

**Challenges**: Hardware procurement, maintenance

### Edge Deployment

**Run models on devices** (phones, IoT)

**Requirements**:
- Small models (< 1B parameters)
- Quantization (4-bit, 8-bit)
- Model compression

**Frameworks**:
- TensorFlow Lite
- ONNX Runtime
- llama.cpp (CPU inference)
- MLC LLM (mobile)

**Use Cases**: Offline operation, low-latency, privacy

### Hybrid Approaches

**Combine multiple strategies**:
- Small model on device
- Large model in cloud (for complex queries)
- Cascading: Try small model first, fall back to large

---

## Monitoring and Evaluation

### Performance Metrics

**Latency**:
- Time to First Token (TTFT)
- Inter-Token Latency
- Total response time

**Throughput**:
- Requests per second
- Tokens per second

**Cost**:
- Compute cost per 1000 tokens
- Infrastructure costs

### Quality Metrics

**Automated**:
- **BLEU, ROUGE**: For translation, summarization
- **Perplexity**: Language modeling quality
- **BERTScore**: Semantic similarity

**Human Evaluation**:
- Helpfulness
- Harmlessness
- Honesty
- Coherence
- Factuality

**LLM-as-Judge**:
- Use GPT-4 to evaluate responses
- Correlates well with human judgment
- Scalable

### Monitoring in Production

**Track**:
- Response quality (sample evaluation)
- User feedback (thumbs up/down)
- Error rates
- Prompt/response distributions
- Toxic content detection

**Tools**: Langfuse, LangSmith, Arize, WhyLabs

**Alerts**:
- Sudden quality drops
- Latency spikes
- High error rates
- Distribution shifts

---

## Building LLM Applications

### Application Patterns

#### 1. Chatbots and Assistants

**Components**:
- Conversation history management
- Context window management
- Memory (long-term, short-term)
- Function calling for actions

**Frameworks**: LangChain, LlamaIndex, Haystack

#### 2. RAG (Retrieval Augmented Generation)

**See**: [RAG Systems documentation](../RAG-Systems/)

**Key Components**:
- Vector database
- Embedding model
- Retrieval strategy
- Prompt construction

#### 3. Agents

**Autonomous systems** that use tools and reasoning

**Components**:
- **Reasoning loop**: Think, Act, Observe, Repeat
- **Tool use**: Search, calculator, code execution, APIs
- **Planning**: Break complex tasks into steps

**Frameworks**: LangChain Agents, AutoGPT, BabyAGI

**Challenges**:
- Reliability (can make mistakes)
- Cost (many LLM calls)
- Latency

#### 4. Code Generation and Assistance

**GitHub Copilot, Cursor, Tabnine**

**Techniques**:
- Fill-in-the-middle (FIM)
- Context from codebase
- Iterative refinement

#### 5. Content Generation

**Blog posts, marketing copy, creative writing**

**Considerations**:
- Style consistency
- Fact-checking
- Plagiarism detection
- Human review

### Prompt Chaining and Workflows

**Break complex tasks into steps**:
1. Extract key information
2. Generate outline
3. Write sections
4. Review and edit
5. Format output

**Benefits**:
- More reliable than single prompt
- Easier to debug
- Modularity

### Function Calling / Tool Use

**Enable LLMs to call external functions**

**Process**:
1. Define available functions (schemas)
2. LLM decides which function to call with arguments
3. Execute function
4. Return results to LLM
5. LLM continues with results

**Use Cases**:
- Database queries
- API calls
- Calculations
- Web searches

**Support**: GPT-4, Claude, Gemini (native), Open-source (via prompting)

---

## Advanced Topics

### Long Context Models

**Trend**: Increasing context windows
- GPT-4: 8K → 32K → 128K tokens
- Claude: 200K tokens
- Gemini 1.5: 1M tokens

**Challenges**:
- Quadratic attention complexity
- "Lost in the middle" phenomenon

**Solutions**:
- Sparse attention mechanisms
- Recurrent memory architectures

### Multimodal LLMs

**Process text + images (+ audio, video)**

**Models**: GPT-4V, Gemini, LLaVA, BLIP

**Architecture**: Vision encoder + text LLM

**Applications**:
- Image captioning
- Visual question answering
- Document understanding
- Object detection

### Mixture of Experts (MoE)

**Sparse models**: Activate subset of parameters per token

**Example**: Mixtral 8x7B
- 8 experts, activate top-2 per token
- 47B total parameters, 13B active per token

**Benefits**:
- Better performance for size
- Faster inference than dense equivalent

### LLM Agents and Planning

**Agentic workflows**:
- ReAct (Reasoning + Acting)
- Tree of Thoughts
- Reflexion (self-reflection)

**Research areas**: Multi-agent collaboration, emergent abilities

### Constitutional AI and Safety

**Alignment techniques**:
- Red teaming
- Adversarial testing
- Safety fine-tuning
- Refusal training

**Open problems**:
- Jailbreaking
- Prompt injection
- Misalignment
- Emergent capabilities

---

## Practical Recommendations

### For Fine-Tuning

1. **Start with prompt engineering**: Often sufficient
2. **Try few-shot learning**: Before fine-tuning
3. **Use PEFT (LoRA)**: Unless you need full fine-tuning
4. **Quality over quantity**: Better data > more data
5. **Evaluate thoroughly**: On held-out test set

### For Deployment

1. **Start with APIs**: Fastest to market
2. **Optimize prompts first**: Before deploying custom models
3. **Monitor continuously**: Quality can degrade
4. **Plan for scale**: Usage often grows quickly
5. **Budget wisely**: LLM costs can be significant

### For Production Systems

1. **Implement caching**: Many queries repeat
2. **Use guardrails**: Input/output validation
3. **Rate limiting**: Prevent abuse
4. **Fallback strategies**: When LLM fails or slow
5. **Human in the loop**: For critical decisions

---

## Key Takeaways

✅ **Fine-tuning**: Full, instruction, or PEFT (LoRA, QLoRA) based on resources

✅ **RLHF**: Three-stage alignment (SFT, reward model, PPO) or alternatives (DPO, Constitutional AI)

✅ **Optimization**: Quantization, KV cache, speculative decoding for faster inference

✅ **Deployment**: Cloud APIs, self-hosted, edge, or hybrid based on requirements

✅ **Monitoring**: Track latency, throughput, quality, and costs in production

✅ **Applications**: Chatbots, RAG, agents, code generation, content creation

✅ **Advanced**: Long context, multimodal, MoE, agentic workflows

✅ **Best practices**: Start simple, iterate, monitor, and plan for scale

---

## Further Reading

**Papers**:
- LoRA: "Low-Rank Adaptation of Large Language Models"
- RLHF: "Training Language Models to Follow Instructions with Human Feedback"
- DPO: "Direct Preference Optimization"
- Speculative Decoding: "Fast Inference from Transformers via Speculative Decoding"

**Resources**:
- Hugging Face PEFT Documentation
- OpenAI Fine-Tuning Guide
- DeepLearning.AI courses on LLMs
- LangChain documentation

---

*Previous: [Intermediate LLM Architecture](02-intermediate-llm-architecture.md)*
