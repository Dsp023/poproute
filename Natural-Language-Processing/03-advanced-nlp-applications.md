# Advanced NLP Applications - Comprehensive Guide

## Table of Contents
1. [Question Answering Systems](#question-answering-systems)
2. [Text Generation](#text-generation)
3. [Machine Translation](#machine-translation)
4. [Information Extraction](#information-extraction)
5. [Multi-Modal NLP](#multi-modal-nlp)
6. [Key Takeaways](#key-takeaways)

---

## Question Answering Systems

### Types of QA

#### 1. Extractive QA

**Task**: Extract answer span from given context

**Example**:
```
Context: "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars 
in Paris, France. It is named after engineer Gustave Eiffel, whose company 
designed and built the tower. It was constructed from 1887 to 1889."

Question: "When was the Eiffel Tower built?"
Answer: "1887 to 1889" (extracted from context)
```

**Approach**: Span prediction
- Predict start and end positions in context

**Architecture**:
```
[CLS] Question [SEP] Context [SEP]
    ↓
BERT Encoder
    ↓
Start/End Position Predictions
```

**Implementation**:
```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
It is named after engineer Gustave Eiffel, whose company designed and built the tower. 
It was constructed from 1887 to 1889.
"""

question = "When was the Eiffel Tower built?"

result = qa_pipeline(question=question, context=context)
print(f"Answer: {result['answer']}")
print(f"Score: {result['score']:.3f}")
print(f"Start: {result['start']}, End: {result['end']}")
```

**Datasets**:
- **SQuAD** (Stanford Question Answering Dataset): 100K+ questions
- **SQuAD 2.0**: Includes unanswerable questions
- **Natural Questions**: Real Google search queries

**Metrics**:
- **Exact Match (EM)**: % predictions exactly matching ground truth
- **F1 Score**: Token-level overlap

**State-of-the-Art**: RoBERTa, ALBERT, ELECTRA on SQuAD (~90% F1)

#### 2. Generative QA

**Task**: Generate answer from knowledge (not extracting)

**Example**:
```
Question: "Why is the sky blue?"
Answer: [Generated explanation about Rayleigh scattering and light wavelengths]
```

**Models**: T5, BART, GPT-3/4

**Implementation**:
```python
from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-base")

question = "Why is the sky blue?"
prompt = f"Answer the following question: {question}"

answer = generator(prompt, max_length=100)[0]['generated_text']
print(answer)
```

**Advantages** over Extractive:
- Can synthesize information
- More natural answers
- Handles questions without explicit answer in text

**Challenges**:
- May hallucinate (generate plausible but incorrect info)
- Harder to verify
- Requires more compute

#### 3. Open-Domain QA

**Task**: Answer questions about anything, without given context

**Pipeline**:
```
Question → Retrieval (find relevant documents) → 
         → Reader (extract/generate answer from documents)
```

**Retrieval Methods**:
- **BM25**: Traditional keyword-based
- **Dense Retrieval**: Embed question and documents, find nearest neighbors
  - DPR (Dense Passage Retrieval)
  - ColBERT

**Modern Approach**: RAG (Retrieval Augmented Generation)
```python
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

question = "When was the first iPhone released?"
input_ids = tokenizer(question, return_tensors="pt")["input_ids"]

generated = model.generate(input_ids)
answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print(answer)
```

**See Also**: [RAG Systems Documentation](../RAG-Systems/) for detailed coverage

#### 4. Conversational QA

**Multi-turn dialogue** with context

**Example**:
```
Q1: "Who wrote Romeo and Juliet?"
A1: "William Shakespeare"

Q2: "When was he born?" (referring to Shakespeare)
A2: "April 1564"
```

**Dataset**: CoQA, QuAC

**Challenges**: Coreference resolution, maintaining context

---

## Text Generation

### Autoregressive Generation

**Process**: Generate one token at a time, conditioning on previous tokens

```
P(text) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ...
```

### Decoding Strategies

#### 1. Greedy Decoding

**Select most probable token** at each step

```python
# Pseudocode
for position in sequence:
    next_token = argmax(P(token | previous_tokens))
```

**Pros**: Fast, deterministic
**Cons**: Can be repetitive, not diverse

#### 2. Beam Search

**Keep top-k most probable sequences**

**Beam size = 3 example**:
```
Step 1: ["I", "The", "A"]  # Top 3 starts
Step 2: ["I am", "I have", "The cat", ...]  # Top 3 continuations each
        → Keep overall top 3
```

**Implementation**:
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Once upon a time"
outputs = generator(
    prompt,
    max_length=50,
    num_beams=5,           # Beam search with beam size 5
    early_stopping=True
)
```

**Pros**: Better quality than greedy
**Cons**: Can still be generic, slower than greedy

#### 3. Top-k Sampling

**Sample from top-k most likely tokens**

```python
# Step 1: Get probabilities
probabilities = model(context)

# Step 2: Keep top-k
top_k_probs, top_k_indices = torch.topk(probabilities, k=50)

# Step 3: Renormalize and sample
top_k_probs = top_k_probs / top_k_probs.sum()
next_token = torch.multinomial(top_k_probs, 1)
```

**Pros**: More diverse than greedy/beam
**Cons**: k is fixed (sometimes need high k, sometimes low)

#### 4. Top-p (Nucleus) Sampling

**Sample from smallest set of tokens whose cumulative probability > p**

**Example with p=0.9**:
```
Token probabilities:
"the": 0.4
"a": 0.3
"an": 0.2
"some": 0.05
"other": 0.03
...

Cumulative: 0.4 → 0.7 → 0.9 (stop here)
Sample from {"the", "a", "an"}
```

**Implementation**:
```python
outputs = generator(
    prompt,
    max_length=100,
    do_sample=True,
    top_p=0.95,           # Nucleus sampling
    temperature=0.8       # Lower = less random
)
```

**Pros**: Adapts to probability distribution
**Cons**: Still can generate repetitive text

#### 5. Temperature

**Control randomness** of sampling

**Formula**:
```
P(token) = exp(logit / temperature) / Σ exp(logits / temperature)
```

**Effects**:
- **Temperature = 0.1**: Very deterministic, focused
- **Temperature = 1.0**: Original distribution
- **Temperature = 2.0**: Very random, creative

**Example**:
```python
# Conservative
outputs = generator(prompt, temperature=0.5, do_sample=True)

# Creative
outputs = generator(prompt, temperature=1.5, do_sample=True)
```

### Controlled Generation

#### 1. Prompt Engineering

**Guide generation** with carefully crafted prompts

**Example**:
```python
prompt = """
Write a professional email to a client apologizing for a delay.

Subject: Apology for Project Delay

Dear [Client Name],
"""

email = generator(prompt, max_length=200)
```

#### 2. Constrained Decoding

**Force generation** to include specific tokens or follow format

**Example**: Generate JSON
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Force generation to be valid JSON
# (Requires libraries like guidance or outlines)
```

#### 3. CTRL (Conditional Transformer Language Model)

**Control codes** to guide generation

```
Control code: Reviews → Generate review
Control code: News → Generate news article
```

### Applications

**Content Creation**:
- Blog posts, articles
- Product descriptions
- Social media posts

**Code Generation**:
- GitHub Copilot
- Code completion
- Bug fixing

**Creative Writing**:
- Story generation
- Poetry
- Dialogue

**Data Augmentation**:
- Generate training examples
- Paraphrase text

---

## Machine Translation

### Neural Machine Translation (NMT)

**Evolution**:
1. **Statistical MT** (2000s): Phrase-based translation
2. **Seq2Seq** (2014): LSTM encoder-decoder
3. **Seq2Seq + Attention** (2015): Attention mechanism
4. **Transformer** (2017): Current standard

### Transformer-Based Translation

**Architecture**:
```
Source Language → Encoder (self-attention) → 
                → Decoder (self-attention + cross-attention) →
                → Target Language
```

**Pre-trained Models**:
- **MarianMT**: 1,000+ language pairs
- **M2M-100**: Many-to-many (100 languages)
- **NLLB** (No Language Left Behind): 200 languages

**Implementation**:
```python
from transformers import MarianMTModel, MarianTokenizer

# English to French
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

translated = model.generate(**inputs)
french_text = tokenizer.decode(translated[0], skip_special_tokens=True)
print(french_text)  # "Bonjour, comment allez-vous?"
```

**Multi-language Translation**:
```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Translate English to French
tokenizer.src_lang = "en_XX"
text = "Hello, how are you?"
encoded = tokenizer(text, return_tensors="pt")

generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print(translation)
```

### Evaluation Metrics

#### BLEU (Bilingual Evaluation Understudy)

**Measures**: N-gram overlap between translation and reference

**Formula**:
```
BLEU = BP × exp(Σ wₙ log pₙ)

pₙ = precision of n-grams
BP = brevity penalty (penalize short translations)
```

**Range**: 0-100 (higher better)
- 10-20: Very poor
-  20-30: Reasonable
- 30-40: Good
- 40-50: Very good
- 50+: Excellent

**Limitation**: Doesn't capture semantics well

#### chrF

**Character n-gram F-score**

**Better for**:
- Morphologically rich languages
- Languages without clear word boundaries

####COMET

**Learned metric** using neural networks

**Correlates better** with human judgment than BLEU

### Challenges

**Idioms and Cultural Context**:
```
English: "It's raining cats and dogs"
Literal translation: Wrong
Correct: Equivalent idiom in target language
```

**Ambiguity**:
```
English: "Can you open the can?"
French: "Pouvez-vous ouvrir la boîte?" (context needed for "can")
```

**Low-Resource Languages**:
- Limited parallel corpora
- Solutions: Multilingual models, unsupervised/semi-supervised MT

---

## Information Extraction

### Task Overview

**Extract structured information** from unstructured text

**Components**:
1. Named Entity Recognition (entities)
2. Relation Extraction (relationships between entities)
3. Event Extraction (events and participants)

### Relation Extraction

**Task**: Identify semantic relationships between entities

**Example**:
```
Text: "Steve Jobs co-founded Apple Inc. in 1976 in Cupertino."

Relations:
- (Steve Jobs, founder-of, Apple Inc.)
- (Apple Inc., founded-in-year, 1976)
- (Apple Inc., located-in, Cupertino)
```

**Approaches**:

#### 1. Pattern-Based

**Define patterns** for relations

```python
import re

text = "Steve Jobs founded Apple"

pattern = r'(\w+ \w+) founded (\w+)'
match = re.search(pattern, text)

if match:
    person, company = match.groups()
    relation = (person, "founder-of", company)
    print(relation)
```

**Pros**: High precision
**Cons**: Low recall, manual effort

#### 2. Supervised Learning

**Train classifier** on labeled examples

**Features**:
- Entity types
- Dependency path between entities
- Words between entities
- POS tags

#### 3. Distant Supervision

**Use knowledge base** to automatically generate training data

**Example**:
```
Knowledge Base: (Barack Obama, president-of, USA)

Find sentences with "Barack Obama" and "USA"
→ Label as "president-of" examples
```

**Pros**: Large training data
**Cons**: Noisy labels

#### 4. Joint Entity and Relation Extraction

**Extract entities and relations simultaneously**

**Models**: SpERT, PURE

```python
# Example with spaCy relation extraction
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple CEO Tim Cook announced new iPhone")

# Custom relation extraction (simplified)
for token in doc:
    if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
        subject = token.text
        verb = token.head.text
        for child in token.head.children:
            if child.dep_ == "dobj":
                object_ = child.text
                print(f"Relation: ({subject}, {verb}, {object_})")
```

### Event Extraction

**Task**: Extract events and their participants

**Example**:
```
Text: "Apple acquired Beats Electronics for $3 billion in 2014."

Event: Acquisition
- Acquirer: Apple
- Acquired: Beats Electronics
- Price: $3 billion
- Date: 2014
```

**ACE (Automatic Content Extraction)** dataset:
- 33 event types
- Event triggers and arguments

### Knowledge Graph Construction

**Build structured KB** from text

**Pipeline**:
```
Text Corpus → Entity Extraction → Relation Extraction → 
            → Entity Linking → Knowledge Graph
```

**Example Knowledge Graph**:
```
(Steve Jobs) -[founder-of]→ (Apple Inc.)
(Apple Inc.) -[headquartered-in]→ (Cupertino)
(Apple Inc.) -[founded-in]→ (1976)
(Steve  Jobs) -[born-in]→ (1955)
```

**Tools**:
- **Stanford OpenIE**: Open information extraction
- **AllenNLP**: IE components
- **spaCy**: Entity and relation extraction

---

## Multi-Modal NLP

### Vision-Language Models

#### CLIP (Contrastive Language-Image Pre-training)

**Task**: Connect images and text

**Training**: Contrastive learning on image-text pairs

**Applications**:
- **Zero-shot image classification**: Classify images without seeing examples
- **Image search**: Find images with text queries
- **Image captioning**: Generate descriptions

**Implementation**:
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cat.jpg")
text = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

for label, prob in zip(text, probs[0]):
    print(f"{label}: {prob:.3f}")
```

#### Image Captioning

**Task**: Generate text description of image

**Models**:
- **BLIP** (Bootstrapping Language-Image Pre-training)
- **GIT** (Generative Image-to-Text)
- **Flamingo** (DeepMind)

**Implementation**:
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("photo.jpg")
inputs = processor(image, return_tensors="pt")

caption_ids = model.generate(**inputs)
caption = processor.decode(caption_ids[0], skip_special_tokens=True)
print(caption)
```

#### Visual Question Answering (VQA)

**Task**: Answer questions about images

**Example**:
```
Image: [Photo of  a beach]
Question: "What is the weather like?"
Answer: "Sunny"
```

**Implementation**:
```python
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

image = Image.open("beach.jpg")
question = "What is the weather like?"

inputs = processor(image, question, return_tensors="pt")
outputs = model(**inputs)

logits = outputs.logits
predicted_idx = logits.argmax(-1).item()
answer = model.config.id2label[predicted_idx]
print(answer)
```

### Audio-Text Models

#### Speech Recognition

**Task**: Transcribe speech to text

**Whisper** (OpenAI):
- Multilingual (99 languages)
- Robust to accents, noise
- State-of-the-art accuracy

**Implementation**:
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Load audio
import librosa
audio, sr = librosa.load("audio.mp3", sr=16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
predicted_ids = model.generate(inputs.input_features)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
```

#### Text-to-Speech (TTS)

**Models**:
- **Tacotron 2**: High-quality synthesis
- **FastSpeech 2**: Fast, controllable
- **VALL-E**: Few-shot voice cloning

---

## Key Takeaways

✅ **Question Answering**: Extractive (SQuAD, BERT), Generative (T5, GPT), Open-Domain (RAG)

✅ **Text Generation**: Autoregressive with various decoding (greedy, beam, top-k, top-p, temperature)

✅ **Machine Translation**: Transformer-based (MarianMT, M2M-100); BLEU for evaluation

✅ **Information Extraction**: NER + Relation Extraction + Event Extraction → Knowledge Graphs

✅ **Multi-Modal**: CLIP (vision-language), BLIP (image captioning), Whisper (speech recognition)

✅ **Evaluation**: BLEU/chrF (translation), EM/F1 (QA), Human evaluation crucial

✅ **Modern Trends**: Pre-trained models, transformers, multimodal learning

✅ **Practical**: Hugging Face Transformers provides easy access to state-of-the-art models

---

## Further Reading

**Papers**:
- "Attention Is All You Need" (Transformer)
- "BERT: Pre-training of Deep Bidirectional Transformers"
- "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
- "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper)

**Resources**:
- Hugging Face Model Hub
- Papers With Code (NLP section)
- Stanford CS224N (NLP with Deep Learning)
- ACL/EMNLP/NAACL conference proceedings

---

*Previous: [Intermediate NLP Techniques](02-intermediate-nlp-techniques.md)*
