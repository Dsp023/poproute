# Intermediate NLP Techniques - Comprehensive Guide

## Table of Contents
1. [Part-of-Speech Tagging](#part-of-speech-tagging)
2. [Named Entity Recognition](#named-entity-recognition)
3. [Sentiment Analysis](#sentiment-analysis)
4. [Text Classification](#text-classification)
5. [Sequence-to-Sequence Models](#sequence-to-sequence-models)
6. [Practical Implementation Examples](#practical-implementation-examples)
7. [Key Takeaways](#key-takeaways)

---

## Part-of-Speech Tagging

### What is POS Tagging?

**Task**: Assign grammatical category to each word

**Common Tags**:
- **NN/NNS**: Noun (singular/plural)
- **VB/VBD/VBG/VBN**: Verb (base/past/gerund/past participle)
- **JJ/JJR/JJS**: Adjective (base/comparative/superlative)
- **RB/RBR/RBS**: Adverb (base/comparative/superlative)
- **DT**: Determiner (the, a, an)
- **IN**: Preposition (in, on, at)
- **PRP**: Personal pronoun (I, you, he)

**Example**:
```
"The quick brown fox jumps over the lazy dog"

The/DT quick/JJ brown/JJ fox/NN jumps/VBZ over/IN the/DT lazy/JJ dog/NN
```

### Algorithms

#### 1. Rule-Based

**Approach**: Hand-crafted rules

**Example Rule**: "If word ends in '-ly', tag as RB (adverb)"

**Pros**: Interpretable, no training data needed
**Cons**: Labor-intensive, doesn't generalize

#### 2. Hidden Markov Models (HMM)

**Probabilistic model**:
- States = POS tags
- Observations = words
- Find most likely tag sequence

**Viterbi Algorithm**: Dynamic programming for optimal path

**Example**:
```python
# Simplified HMM concept
P(tag_sequence | word_sequence) = 
    P(word | tag) × P(tag | previous_tag)
```

#### 3. Conditional Random Fields (CRF)

**Discriminative model**: Models P(tags | words) directly

**Advantages** over HMM:
- Can use rich features
- No independence assumptions
- Better performance

#### 4. Deep Learning Approaches

**BiLSTM (Bidirectional LSTM)**:
```
"The cat sat" 
→ Forward LSTM →
← Backward LSTM ←
→ Combined representation → POS tags
```

**Transformer-Based** (BERT, RoBERTa):
- State-of-the-art performance
- Context-aware
- Pre-trained models available

**Implementation with spaCy**:
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog")

for token in doc:
    print(f"{token.text:10} {token.pos_:5} {token.tag_:5} {spacy.explain(token.tag_)}")

# Output:
# The        DET   DT    determiner
# quick      ADJ   JJ    adjective
# brown      ADJ   JJ    adjective
# fox        NOUN  NN    noun, singular
# jumps      VERB  VBZ   verb, 3rd person singular present
# ...
```

### Applications

**Syntactic Parsing**: Foundation for dependency parsing
**Information Extraction**: Identify noun phrases for entities
**Machine Translation**: Word alignment
**Text-to-Speech**: Pronunciation depends on POS ("read" as noun vs verb)

---

## Named Entity Recognition

### Deep Dive

**Task**: Identify and classify named entities in text

**Standard Entity Types**:
- PERSON: People, fictional characters
- ORG: Companies, agencies, institutions
- GPE: Geopolitical entities (countries, cities, states)
- LOC: Non-GPE locations (mountains, bodies of water)
- DATE: Absolute or relative dates
- TIME: Times smaller than a day
- MONEY: Monetary values
- PERCENT: Percentages
- PRODUCT: Products, vehicles, foods
- EVENT: Named events (hurricanes, battles, wars)

### BIO Tagging Scheme

**Encoding** entity boundaries:
- **B**egin: First token of entity
- **I**nside: Inside entity
- **O**utside: Not an entity

**Example**:
```
"Apple Inc. was founded by Steve Jobs in California"

Apple   → B-ORG
Inc.    → I-ORG
was     → O
founded → O
by      → O
Steve   → B-PERSON
Jobs    → I-PERSON
in      → O
California → B-GPE
```

### Algorithms

#### 1. Rule-Based

**Gazetteers**: Lists of known entities
```python
if word in person_names:
    tag = "PERSON"
```

**Patterns**: Regular expressions
```python
# Email pattern
r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b' → EMAIL
```

**Pros**: High precision for known entities
**Cons**: Low recall, manual maintenance

#### 2. Conditional Random Fields (CRF)

**Features**:
- Word itself
- Capitalization (is_capitalized, is_all_caps)
- Word shape (Xxxxx, XXX, dd/dd/dddd)
- Prefix/suffix (first 3 chars, last 3 chars)
- POS tag
- Context words (previous, next)

**Example Feature Vector**:
```
Word: "Apple"
Features:
- word=apple
- is_capitalized=True
- is_first_word=True
- prev_word=None
- next_word=inc
- pos=NNP
- suffix=ple
```

#### 3. BiLSTM-CRF

**Architecture**:
```
Input: Word embeddings
   ↓
BiLSTM: Contextual representations
   ↓
CRF: Sequence labeling (ensures valid tag sequences)
   ↓
Output: BIO tags
```

**Why CRF on top?**
- Enforces constraints (B-PER can't follow I-ORG)
- Models dependencies between labels

**Implementation**:
```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976."
entities = ner(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} (score: {entity['score']:.3f})")

# Output:
# Apple Inc.: ORG (score: 0.998)
# Steve Jobs: PER (score: 0.999)
# Cupertino: LOC (score: 0.996)
# California: LOC (score: 0.997)
```

#### 4. Transformer-Based (BERT, RoBERTa)

**Fine-tune pre-trained model** on NER task

**State-of-the-art performance**: 90-95% F1 on CoNLL-2003

**Pre-trained NER Models**:
- `dslim/bert-base-NER`
- `dbmdz/bert-large-cased-finetuned-conll03-english`
- `xlm-roberta-large-finetuned-conll03-english`

### Evaluation Metrics

**Strict Match**: Entity boundaries AND type must match
**Lenient Match**: Overlap counts

**Metrics**:
- **Precision**: Correct entities / Predicted entities
- **Recall**: Correct entities / Actual entities
- **F1**: Harmonic mean of precision and recall

---

## Sentiment Analysis

### Beyond Binary Classification

#### Fine-Grained Sentiment

**5-point scale**:
- Very Negative (1 star)
- Negative (2 stars)
- Neutral (3 stars)
- Positive (4 stars)
- Very Positive (5 stars)

#### Aspect-Based Sentiment Analysis (ABSA)

**Task**: Extract aspects and their sentiments

**Example**:
```
Review: "The pizza was delicious but the service was terrible"

Aspects Extracted:
- Food (pizza): POSITIVE
- Service: NEGATIVE
```

**Pipeline**:
1. Aspect extraction (identify aspects)
2. Sentiment classification (per aspect)

**Implementation**:
```python
from transformers import pipeline

# Aspect-based sentiment
classifier = pipeline("sentiment-analysis")

text = "The pizza was delicious but the service was terrible"

# Split by aspects (simple approach)
aspects = {
    "pizza": "The pizza was delicious",
    "service": "the service was terrible"
}

for aspect, sentence in aspects.items():
    result = classifier(sentence)[0]
    print(f"{aspect}: {result['label']} ({result['score']:.3f})")
```

#### Emotion Detection

**Beyond positive/negative**:
- Joy, Sadness, Anger, Fear, Surprise, Disgust

**Dataset**: GoEmotions (58,000 Reddit comments, 27 emotions)

**Use Cases**: Mental health monitoring, customer service

### Algorithms

#### 1. Lexicon-Based

**Sentiment Lexicons**:
- VADER: Valence Aware Dictionary and sEntiment Reasoner
- SentiWordNet
- AFINN

**VADER Example**:
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

texts = [
    "This is amazing!",
    "This is terrible.",
    "This is okay.",
    "This is AMAZING!!!" # emphasis matters
]

for text in texts:
    scores = analyzer.polarity_scores(text)
    print(f"{text:20} → compound: {scores['compound']:.3f}")

# Output shows VADER handles emphasis, punctuation, capitalization
```

**Handles**:
- Negation: "not good" → negative
- Emphasis: "VERY good" → more positive
- Emoticons: ":)" → positive

#### 2. Machine Learning

**Features**:
- BoW / TF-IDF
- N-grams (bigrams, trigrams)
- Word embeddings (average Word2Vec)

**Classifiers**:
- Naive Bayes (baseline)
- Logistic Regression
- SVM
- Random Forest

**Example**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

texts = [...]  # Your texts
labels = [...]  # 0 (negative) or 1 (positive)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
accuracy = clf.score(X_test_vec, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

#### 3. Deep Learning

**LSTM**:
```
Words → Embeddings → LSTM → Dense → Softmax → Sentiment
```

**CNN for Text**:
```
Words → Embeddings → Conv1D (multiple filter sizes) → MaxPool → Dense → Sentiment
```

**Transformer (BERT)**:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load pre-trained sentiment model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name)

# Classify
result = classifier("I absolutely loved this movie!")[0]
print(f"Label: {result['label']}, Score: {result['score']:.3f}")
```

### Challenges

**Sarcasm**: "Great, another delay!" (negative despite "great")
**Domain-specific**: "Unpredictable plot" (positive for thriller, negative for romance)
**Negation**: "not bad" (positive)
**Comparative**: "better than expected" (positive)

---

## Text Classification

### Problem Formulation

**Input**: Document (text)
**Output**: Class label(s)

**Types**:
- **Binary**: Spam vs not spam
- **Multi-class**: Topic classification (sports, politics, tech, ...)
- **Multi-label**: Article can be both "sports" and "business"

### Classic ML Approach

#### Pipeline

**1. Preprocessing**:
```python
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords and word.isalnum()]
    return ' '.join(tokens)
```

**2. Feature Extraction**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=10000,    # Top 10K features
    ngram_range=(1, 2),   # Unigrams + bigrams
    min_df=5,             # Ignore rare terms
    max_df=0.8            # Ignore very common terms
)
```

**3. Classification**:
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Naive Bayes (fast, good baseline)
clf_nb = MultinomialNB()

# Logistic Regression (often best for text)
clf_lr = LogisticRegression(max_iter=1000, C=1.0)

# SVM (powerful but slower)
clf_svm = LinearSVC()
```

**4. Evaluation**:
```python
from sklearn.metrics import classification_report, confusion_matrix

predictions = clf.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```

### Deep Learning Approach

#### 1. FastText

**Efficient text classification** (Facebook Research)

**Architecture**: Average word embeddings → Classification

```python
from gensim.models.fasttext import FastText

# Train FastText classifier
# Very fast, good performance
```

#### 2. CNN for Text

**Convolution over text**:
```
Input: "I love this movie"
Embeddings: [emb(I), emb(love), emb(this), emb(movie)]
Conv filters: [2-gram, 3-gram, 4-gram]
Max pooling: Extract most important features
Output: Classification
```

#### 3. RNN/LSTM

**Sequential processing**:
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

#### 4. Transformer Fine-Tuning (Best Performance)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()
```

### Class Imbalance

**Problem**: 95% class A, 5% class B

**Solutions**:

**1. Resampling**:
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Oversample minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**2. Class Weights**:
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# In sklearn
clf = LogisticRegression(class_weight='balanced')

# In neural networks
model.fit(X, y, class_weight=class_weight_dict)
```

---

## Sequence-to-Sequence Models

### Architecture

**Encoder-Decoder**:
```
Input Sequence → Encoder → Context Vector → Decoder → Output Sequence
```

**Use Cases**:
- Machine Translation: English → French
- Text Summarization: Long text → Short summary
- Question Answering: Question + Context → Answer

### Basic Seq2Seq (RNN-based)

**Components**:

**Encoder**:
```
"Hello how are you" → LSTM → final hidden state (context vector)
```

**Decoder**:
```
Context vector → LSTM → Generate "Bonjour comment allez-vous" token by token
```

**Limitation**: Fixed-size context vector (bottleneck)

### Seq2Seq with Attention

**Innovation**: Decoder attends to all encoder states, not just last

**Mechanism**:
1. Encoder produces hidden states h₁, h₂, ..., hₙ
2. At each decoding step, calculate attention weights over encoder states
3. Create weighted sum (context vector specific to this step)
4. Use context + previous output to generate next token

**Advantage**: No fixed bottleneck, better for long sequences

### Transformer Seq2Seq

**Modern standard**: Multi-head self-attention

**Architecture**:
```
Input → Encoder (self-attention layers) → 
     → Decoder (self-attention + cross-attention to encoder) → 
     → Output
```

**Examples**: T5, BART, MarianMT

### Implementation Example: Summarization

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
[Long article text here - 500+ words]
"""

summary = summarizer(article, max_length=130, min_length=30, do_sample=False)[0]
print(summary['summary_text'])
```

---

## Practical Implementation Examples

### Complete NER System

```python
import spacy
from spacy.training import Example
from spacy.util import minibatch
import random

# Load blank model
nlp = spacy.blank("en")

# Create NER component
ner = nlp.add_pipe("ner")

# Add labels
labels = ["PERSON", "ORG", "GPE"]
for label in labels:
    ner.add_label(label)

# Training data format
TRAIN_DATA = [
    ("Apple Inc. was founded by Steve Jobs", {
        "entities": [(0, 10, "ORG"), (27, 38, "PERSON")]
    }),
    # More examples...
]

# Training loop
optimizer = nlp.begin_training()
for epoch in range(30):
    random.shuffle(TRAIN_DATA)
    losses = {}
    
    for batch in minibatch(TRAIN_DATA, size=8):
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        
        nlp.update(examples, drop=0.5, losses=losses, sgd=optimizer)
    
    print(f"Epoch {epoch}, Losses: {losses}")

# Save model
nlp.to_disk("./my_ner_model")

# Use model
nlp_loaded = spacy.load("./my_ner_model")
doc = nlp_loaded("Microsoft was founded by Bill Gates")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
```

### Multi-Class Text Classification

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)

# Create pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', MultinomialNB()),
])

# Train
text_clf.fit(twenty_train.data, twenty_train.target)

# Evaluate
predicted = text_clf.predict(twenty_test.data)
print(classification_report(twenty_test.target, predicted, target_names=categories))

# Predict new document
docs_new = ['God is love', 'OpenGL on the GPU is fast']
predicted_categories = text_clf.predict(docs_new)
for doc, category_id in zip(docs_new, predicted_categories):
    print(f"{doc} → {categories[category_id]}")
```

---

## Key Takeaways

✅ **POS Tagging**: Foundation for many NLP tasks; modern approaches use BiLSTM or Transformers

✅ **NER**: BIO tagging scheme; CRF and BiLSTM-CRF traditional, BERT-based state-of-the-art

✅ **Sentiment Analysis**: Beyond binary - aspect-based, emotion detection; VADER for lexicon-based, BERT for deep learning

✅ **Text Classification**: TF-IDF + Logistic Regression good baseline, BERT fine-tuning for best performance

✅ **Seq2Seq**: Encoder-decoder architecture; attention mechanism crucial; Transformers are modern standard

✅ **Practical**: spaCy for production, Hugging Face Transformers for state-of-the-art, sklearn for classical ML

✅ **Evaluation**: Precision, recall, F1 for classification and NER; BLEU for translation

✅ **Challenges**: Class imbalance (use resampling/weighted loss), domain adaptation, long sequences

---

## What's Next?

Continue your NLP journey:

- **[Advanced NLP Applications](03-advanced-nlp-applications.md)**: Question answering, text generation, machine translation, multi-modal NLP
- **[Transformers](../Transformers/01-beginner-transformer-basics.md)**: Deep dive into the architecture
- **[LLMs](../Large-Language-Models/01-beginner-llm-basics.md)**: Modern language models

---

*Previous: [NLP Basics](01-beginner-nlp-basics.md) | Next: [Advanced NLP Applications](03-advanced-nlp-applications.md)*
