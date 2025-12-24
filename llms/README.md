# 🤖 Large Language Models (LLMs)

Comprehensive guide to understanding, implementing, and fine-tuning Large Language Models.

## 📚 Table of Contents

1. [Introduction to LLMs](#introduction)
2. [Architecture & Transformers](#architecture)
3. [Working with LLMs](#working-with-llms)
4. [Fine-tuning Techniques](#fine-tuning)
5. [Prompt Engineering](#prompt-engineering)
6. [Practical Applications](#applications)
7. [Resources](#resources)

## Introduction

### What are Large Language Models?

LLMs are neural networks trained on massive amounts of text data to understand and generate human-like text. They power applications like ChatGPT, Claude, and many others.

**Key Characteristics:**
- **Billions of parameters** (GPT-4: ~1.7T, LLaMA 2: 7B-70B)
- **Trained on diverse text** (books, websites, code)
- **Zero-shot & few-shot learning** capabilities
- **Emergent abilities** at scale

### Evolution Timeline

```
2017: Transformer Architecture (Attention Is All You Need)
2018: BERT, GPT-1
2019: GPT-2, T5, RoBERTa
2020: GPT-3
2021: CLIP, DALL-E, Codex
2022: ChatGPT, InstructGPT
2023: GPT-4, Claude, LLaMA, PaLM 2
2024: Gemini, Mixtral, Claude 3
```

## Architecture

### The Transformer Architecture

The foundation of all modern LLMs.

**Core Components:**

1. **Self-Attention Mechanism**
```python
# Simplified attention calculation
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: Query matrix (batch, seq_len, d_k)
    K: Key matrix (batch, seq_len, d_k)
    V: Value matrix (batch, seq_len, d_v)
    """
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Multiply by values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

2. **Multi-Head Attention**
```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and split into heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attention = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        return self.W_o(output)
```

3. **Position-wise Feed-Forward Networks**
```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

### Popular LLM Architectures

#### 1. **GPT (Decoder-only)**
- Autoregressive generation
- Causal (left-to-right) attention
- Best for: Generation tasks

#### 2. **BERT (Encoder-only)**
- Bidirectional attention
- Masked language modeling
- Best for: Understanding tasks

#### 3. **T5 (Encoder-Decoder)**
- Text-to-text framework
- Flexible architecture
- Best for: Translation, summarization

## Working with LLMs

### Using OpenAI API

```python
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def chat_completion(messages, model="gpt-4", temperature=0.7):
    """Send chat completion request"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1000
    )
    return response.choices[0].message.content

# Example usage
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

response = chat_completion(messages)
print(response)
```

### Using Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_text(prompt, max_length=200):
    """Generate text using local model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
response = generate_text("What is the meaning of life?")
print(response)
```

### Using LangChain

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize LLM
llm = OpenAI(temperature=0.7)

# Create prompt template
template = """
You are an expert {role}.

Question: {question}

Please provide a detailed answer:
"""

prompt = PromptTemplate(
    input_variables=["role", "question"],
    template=template
)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run chain
result = chain.run(
    role="data scientist",
    question="What are the best practices for feature engineering?"
)

print(result)
```

## Fine-tuning

### 1. Full Fine-tuning

Training all model parameters (expensive).

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

# Train
trainer.train()
```

### 2. LoRA (Low-Rank Adaptation)

Efficient fine-tuning by training small adapter layers.

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Which layers to adapt
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Only 0.1% of parameters are trainable!
model.print_trainable_parameters()

# Train as normal
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
```

### 3. QLoRA (Quantized LoRA)

Even more efficient - combines quantization with LoRA.

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
```

### 4. Prompt Tuning

Train soft prompts instead of model weights.

```python
from peft import PromptTuningConfig, PromptTuningInit

config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=20,
    prompt_tuning_init_text="Classify if the sentiment is positive or negative:",
    tokenizer_name_or_path=model_name
)

model = get_peft_model(model, config)
```

## Prompt Engineering

The art of crafting effective prompts for LLMs.

### Basic Techniques

#### 1. **Zero-Shot Prompting**
```python
prompt = "Translate English to French: Hello, how are you?"
```

#### 2. **Few-Shot Prompting**
```python
prompt = """
Translate English to French:

English: Hello
French: Bonjour

English: Good morning
French: Bonjour

English: Thank you
French: Merci

English: How are you?
French:
"""
```

#### 3. **Chain-of-Thought (CoT)**
```python
prompt = """
Question: A juggler can juggle 16 balls. Half of the balls are golf balls, 
and half of the golf balls are blue. How many blue golf balls are there?

Let's think step by step:
1. Total balls = 16
2. Golf balls = 16 / 2 = 8
3. Blue golf balls = 8 / 2 = 4

Answer: 4 blue golf balls

Question: If I have 30 apples and give away 1/3 of them, then buy 12 more, 
how many apples do I have?

Let's think step by step:
"""
```

#### 4. **Role Prompting**
```python
prompt = """
You are an expert Python developer with 10 years of experience.
Review this code and suggest improvements:

def calculate(x, y):
    return x + y
"""
```

### Advanced Techniques

#### 1. **Self-Consistency**
Generate multiple reasoning paths and take majority vote.

```python
def self_consistency(question, num_samples=5):
    answers = []
    for _ in range(num_samples):
        prompt = f"{question}\n\nLet's think step by step:"
        response = llm.generate(prompt)
        answers.append(extract_answer(response))
    
    # Return most common answer
    return max(set(answers), key=answers.count)
```

#### 2. **ReAct (Reasoning + Acting)**
Interleave reasoning and actions.

```python
prompt = """
Question: What is the elevation of the highest mountain in California?

Thought: I need to find the highest mountain in California first.
Action: Search[highest mountain in California]
Observation: Mount Whitney is the highest mountain in California.

Thought: Now I need to find the elevation of Mount Whitney.
Action: Search[Mount Whitney elevation]
Observation: Mount Whitney has an elevation of 14,505 feet.

Thought: I now know the final answer.
Answer: The elevation is 14,505 feet.
"""
```

## Applications

### 1. Chatbots

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Multi-turn conversation
conversation.predict(input="Hi, I'm working on a Python project")
conversation.predict(input="Can you help me with error handling?")
```

### 2. Text Summarization

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
[Long article text here...]
"""

summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
print(summary[0]['summary_text'])
```

### 3. Code Generation

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model="gpt-4", temperature=0)

template = """
Write a Python function that {task}.
Include:
- Docstring
- Type hints
- Error handling
- Example usage

Function:
"""

prompt = PromptTemplate(template=template, input_variables=["task"])

code = llm(prompt.format(task="sorts a list of dictionaries by a specific key"))
print(code)
```

### 4. Question Answering

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load and split documents
loader = TextLoader("document.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Ask questions
answer = qa_chain.run("What is the main topic of the document?")
```

## Best Practices

### 1. **Token Management**
```python
def count_tokens(text, model="gpt-4"):
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Check before API call
tokens = count_tokens(prompt)
if tokens > 4000:
    print("Warning: Prompt too long!")
```

### 2. **Error Handling & Retries**
```python
import time
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), 
       stop=stop_after_attempt(3))
def call_llm_with_retry(prompt):
    try:
        return openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        print(f"Error: {e}")
        raise
```

### 3. **Cost Optimization**
```python
# Use cheaper models for simple tasks
def choose_model(task_complexity):
    if task_complexity == "simple":
        return "gpt-3.5-turbo"  # Cheaper
    elif task_complexity == "medium":
        return "gpt-4"
    else:
        return "gpt-4-turbo"  # Most capable
```

## Resources

### 📖 Essential Reading
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (GPT-3 paper)
- "Constitutional AI" (Anthropic)
- "LLaMA: Open and Efficient Foundation Language Models"

### 🛠️ Tools & Libraries
- **Hugging Face Transformers**: Model hub & inference
- **LangChain**: LLM application framework
- **LlamaIndex**: Data framework for LLMs
- **Weights & Biases**: Experiment tracking
- **vLLM**: High-performance inference
- **Text Generation Inference**: Production serving

### 🎓 Courses
- "State of GPT" by Andrej Karpathy
- Hugging Face NLP Course
- DeepLearning.AI LLM courses

### 🌐 Communities
- Hugging Face Forums
- r/LocalLLaMA
- EleutherAI Discord

---

**Next Steps:**
- Explore [RAG systems](../rag/README.md) to enhance LLM knowledge
- Check [practical projects](../projects/README.md)
- Learn about [production deployment](../tech/deployment.md)
