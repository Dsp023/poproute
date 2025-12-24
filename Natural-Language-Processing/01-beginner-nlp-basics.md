# Natural Language Processing Basics - Comprehensive Beginner Guide

## Table of Contents
1. [What is NLP?](#what-is-nlp)
2. [Why NLP Matters](#why-nlp-matters)
3. [Core NLP Tasks](#core-nlp-tasks)
4. [Text Preprocessing](#text-preprocessing)
5. [Text Representation](#text-representation)
6. [Word Embeddings](#word-embeddings)
7. [Basic NLP with Python](#basic-nlp-with-python)
8. [Popular NLP Libraries](#popular-nlp-libraries)
9. [Getting Started Guide](#getting-started-guide)
10. [Common Challenges](#common-challenges)
11. [Key Takeaways](#key-takeaways)

---

## What is NLP?

**Natural Language Processing (NLP)** is a field of artificial intelligence that focuses on enabling computers to understand, interpret, manipulate, and generate human language.

### Simple Definition

NLP is teaching computers to:
- **Understand** what we say and write
- **Generate** human-like text
- **Translate** between languages
- **Extract** meaning from text

### The Challenge

Human language is inherently complex:
- **Ambiguity**: "I saw her duck" (bird or action?)
- **Context-dependent**: "That's sick!" (bad or awesome?)
- **Idioms**: "Break a leg" doesn't mean literal injury
- **Variations**: Multiple ways to express same idea

### Real-World Examples You Use Daily

**Smartphone**:
- Autocomplete on keyboard
- Voice assistants (Siri, Google Assistant, Alexa)
- Speech-to-text dictation

**Email**:
- Spam filtering
- Smart reply suggestions
- Grammar checking (Grammarly)

**Search**:
- Google search understanding queries
- Autocomplete suggestions
- "Did you mean..." corrections

**Social Media**:
- Hashtag recommendations
- Content moderation
- Trending topics extraction

**Customer Service**:
- Chatbots answering questions
- Ticket routing
- Sentiment analysis of feedback

---

## Why NLP Matters

### Transforming Industries

**Healthcare** 🏥:
- Extracting information from medical records
- Clinical decision support
- Patient sentiment analysis
- Medical literature analysis

**Finance** 💰:
- News sentiment affecting stocks
- Fraud detection from text
- Automated report generation
- Contract analysis

**E-commerce** 🛒:
- Product review analysis
- Recommendation from reviews
- Chatbot customer service
- Search query understanding

**Legal** ⚖️:
- Contract review and analysis
- Legal research
- Document summarization
- Compliance monitoring

**Education** 📚:
- Automated essay grading
- Personalized learning
- Language learning apps (Duolingo)
- Plagiarism detection

### Market Size

NLP market projected to reach **$43.9 billion by 2028** (growing at 20%+ annually)

---

## Core NLP Tasks

### 1. Text Classification

**Task**: Assign predefined categories/labels to text

**Question**: "What category does this text belong to?"

**Common Applications**:

**Spam Detection**:
```
Input: "Congratulations! You've won $1,000,000!"
Output: SPAM

Input: "Meeting at 2pm tomorrow in conference room A"
Output: NOT  SPAM
```

**Sentiment Analysis**:
```
Input: "This movie was absolutely amazing!"
Output: POSITIVE

Input: "Worst experience ever, waste of money"
Output: NEGATIVE

Input: "The product is okay, nothing special"
Output: NEUTRAL
```

**Topic Classification**:
```
Input: "The Lakers defeated the Bulls 110-98..."
Output: SPORTS

Input: "Congress passed the new healthcare bill..."
Output: POLITICS
```

**Real-World Example**: Email providers use text classification to:
- Filter spam
- Categorize emails (Primary, Social, Promotions)
- Prioritize important messages

---

### 2. Named Entity Recognition (NER)

**Task**: Identify and classify named entities (proper nouns) in text

**Common Entity Types**:
- **PERSON**: Names of people
- **ORGANIZATION**: Companies, agencies, institutions
- **LOCATION**: Cities, countries, geographic locations
- **DATE**: Absolute or relative dates or periods
- **TIME**: Times smaller than a day
- **MONEY**: Monetary values
- **PERCENT**: Percentage values

**Example**:
```
Input:
"Apple Inc. was founded by Steve Jobs in Cupertino, California 
on April 1, 1976 with an initial investment of $1,300."

Entities Identified:
- Apple Inc. → ORGANIZATION
- Steve Jobs → PERSON
- Cupertino → LOCATION
- California → LOCATION
- April 1, 1976 → DATE
- $1,300 → MONEY
```

**Applications**:
- **Information Extraction**: Build knowledge bases
- **Question Answering**: Find relevant entities
- **Content Recommendation**: Identify key topics
- **Resume Parsing**: Extract skills, companies, education

---

### 3. Sentiment Analysis

**Task**: Determine the emotional tone or attitude expressed in text

**Levels**:

**Document-Level**:
- Overall sentiment of entire document
- Example: Movie review positive or negative

**Sentence-Level**:
- Sentiment of each sentence
- Example: "The room was nice but service was terrible"
  - Sentence 1: Positive (room)
  - Sentence 2: Negative (service)

**Aspect-Based**:
- Sentiment towards specific aspects
```
Review: "The hotel location was perfect, but rooms were dirty 
and staff was rude"

Aspects:
- Location: POSITIVE
- Rooms: NEGATIVE
- Staff: NEGATIVE
```

**Applications**:
- **Brand Monitoring**: Track customer sentiment
- **Product Reviews**: Aggregate feedback
- **Social Media**: Public opinion on topics
- **Stock Market**: News sentiment affecting prices

---

### 4. Machine Translation

**Task**: Automatically translate text from one language to another

**Evolution**:
1. **Rule-Based** (1950s-1990s): Hand-crafted grammar rules
2. **Statistical MT** (1990s-2010s): Learn from parallel corpora
3. **Neural MT** (2014-present): Deep learning, end-to-end

**Modern Approach**: Transformer-based (2017-)
- Attention mechanism
- State-of-the-art quality
- Example: Google Translate, DeepL

**Challenges**:
- Idiomatic expressions
- Cultural context
- Low-resource languages
-Ambiguity resolution

---

### 5. Question Answering (QA)

**Task**: Automatically answer questions posed in natural language

**Types**:

**Extractive QA**:
- Find answer span in given text
```
Context: "The Eiffel Tower is located in Paris, France. 
It was completed in 1889."

Question: "Where is the Eiffel Tower?"
Answer: "Paris, France" (extracted from context)
```

**Generative QA**:
- Generate answer from knowledge
```
Question: "Why is the sky blue?"
Answer: [Generated explanation about Rayleigh scattering]
```

**Open-Domain QA**:
- Answer questions about anything
- Search knowledge base + extract/generate answer

**Applications**:
- Customer service bots
- Virtual assistants
- Search engines
- Educational tools

---

### 6. Text Summarization

**Task**: Create concise summary preserving key information

**Approaches**:

**Extractive Summarization**:
- Select important sentences from original
- No new text generated
- Faster, safer
```
Original: [5 paragraphs, 500 words]
Summary: [3 key sentences extracted, 50 words]
```

**Abstractive Summarization**:
- Generate new sentences paraphrasing content
- More human-like
- Can be more concise
```
Original: "The company announced record profits... expanding to Asia... 
hiring 500 employees..."
Summary: "Company reports growth with Asian expansion and major hiring."
```

**Applications**:
- News aggregation
- Meeting notes
- Research paper abstracts
- Email digests

---

### 7. Text Generation

**Task**: Automatically generate coherent, contextually relevant text

**Applications**:
- **Content Creation**: Blog posts, product descriptions
- **Code Generation**: GitHub Copilot
- **Chatbots**: Conversational AI
- **Creative Writing**: Stories, poetry
- **Data-to-Text**: Generate reports from data

**Modern Models**: GPT-3, GPT-4, Claude, Gemini

---

### 8. Part-of-Speech (POS) Tagging

**Task**: Label each word with its grammatical category

**Common Tags**:
- **NN**: Noun
- **VB**: Verb
- **JJ**: Adjective
- **RB**: Adverb
- **DT**: Determiner
- **IN**: Preposition

**Example**:
```
"The quick brown fox jumps over the lazy dog"

The/DT quick/JJ brown/JJ fox/NN jumps/VBZ over/IN 
the/DT lazy/JJ dog/NN
```

**Uses**: Foundation for other NLP tasks (parsing, NER, etc.)

---

## Text Preprocessing

Real-world text is messy. Preprocessing cleans and standardizes text for analysis.

### Why Preprocess?

**Raw Text Issues**:
- Inconsistent capitalization
- Punctuation and special characters
- Extra whitespace
- HTML tags, URLs
- Misspellings
- Non-standard formats

**Goal**: Convert raw text to clean, standardized format

### Common Preprocessing Steps

#### 1. Lowercasing

**Purpose**: Treat "Hello", "hello", "HELLO" as same word

```python
text = "Hello World!"
lowercased = text.lower()
# Output: "hello world!"
```

**When to use**: Most tasks (except when case matters, e.g., NER where "Apple" vs "apple" matters)

#### 2. Tokenization

**Purpose**: Split text into individual units (tokens)

**Word Tokenization**:
```python
text = "I love Natural Language Processing!"
tokens = text.split()
# Output: ["I", "love", "Natural", "Language", "Processing!"]
```

**Better Tokenization** (handles punctuation):
```python
import nltk
tokens = nltk.word_tokenize(text)
# Output: ["I", "love", "Natural", "Language", "Processing", "!"]
```

**Sentence Tokenization**:
```python
text = "Hello world. How are you? I'm fine!"
sentences = nltk.sent_tokenize(text)
# Output: ["Hello world.", "How are you?", "I'm fine!"]
```

#### 3. Removing Stopwords

**Stopwords**: Common words with little meaning (the, is, at, which, on)

**Purpose**: Reduce noise, focus on meaningful words

```python
from nltk.corpus import stopwords

text = "This is an example sentence to demonstrate stopword removal"
tokens = text.split()
stop_words = set(stopwords.words('english'))

filtered = [word for word in tokens if word.lower() not in stop_words]
# Output: ["example", "sentence", "demonstrate", "stopword", "removal"]
```

**When NOT to remove**: Sentiment analysis ("not good" → "good" if you remove "not"!)

#### 4. Stemming

**Purpose**: Reduce words to root form (crude way)

**Example**:
```
running → run
runs → run
runner → runner (not reduced)
```

**Implementation**:
```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "runs", "ran", "runner", "easily", "fairly"]

stemmed = [stemmer.stem(word) for word in words]
# Output: ["run", "run", "ran", "runner", "easili", "fairli"]
```

**Pros**: Fast, simple
**Cons**: Not always accurate ("easili" instead of "easy")

#### 5. Lemmatization

**Purpose**: Reduce words to dictionary form (proper way)

**Example**:
```
running → run
better → good
am/is/are → be
```

**Implementation**:
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ["running", "better", "am", "cats", "mice"]

lemmatized = [lemmatizer.lemmatize(word) for word in words]
# Output: ["running", "good", "am", "cat", "mouse"]

# With POS tags for better results
lemmatized_verb = lemmatizer.lemmatize("running", pos='v')
# Output: "run"
```

**Pros**: Accurate, real words
**Cons**: Slower than stemming

#### 6. Removing Punctuation & Special Characters

```python
import string

text = "Hello, World! How's it going? #NLP @2023"
# Remove punctuation
text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
# Output: "Hello World Hows it going NLP 2023"
```

#### 7. Removing URLs, HTML Tags, Emails

```python
import re

text = "Check out https://example.com and email me at user@email.com"

# Remove URLs
text = re.sub(r'http\S+|www\S+', '', text)

# Remove emails
text = re.sub(r'\S+@\S+', '', text)

# Remove HTML tags
html = "<p>This is <b>bold</b> text</p>"
clean = re.sub(r'<.*?>', '', html)
# Output: "This is bold text"
```

---

## Text Representation

Computers work with numbers, not text. We need to convert text to numerical representations.

### 1. Bag of Words (BoW)

**Concept**: Represent text as counts of words, ignoring grammar and order

**Example**:
```
Document 1: "I love NLP"
Document 2: "I love AI"
Document 3: "NLP is great"

Vocabulary: ["I", "love", "NLP", "AI", "is", "great"]

BoW Representation:
Doc 1: [1, 1, 1, 0, 0, 0]
Doc 2: [1, 1, 0, 1, 0, 0]
Doc 3: [0, 0, 1, 0, 1, 1]
```

**Python Implementation**:
```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love NLP",
    "I love AI", 
    "NLP is great"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(bow_matrix.toarray())
```

**Advantages**: Simple, intuitive
**Disadvantages**: 
- Ignores word order ("dog bites man" = "man bites dog")
- High dimensionality for large vocabularies
- Sparse vectors (mostly zeros)

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)

**Concept**: Weight words by importance (rare words are more important)

**Formula**:
```
TF-IDF(word, document) = TF(word, document) × IDF(word)

TF = Count of word in document / Total words in document
IDF = log(Total documents / Documents containing word)
```

**Intuition**:
- High TF: Word appears often in this document
- High IDF: Word is rare across all documents
- High TF-IDF: Word is important for this specific document

**Example**:
```
Doc 1: "The cat sat on the mat"
Doc 2: "The dog sat on the log"

"cat" appears only in Doc 1 → High IDF → High TF-IDF for Doc 1
"the" appears in both → Low IDF → Low TF-IDF
```

**Python Implementation**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "The cat sat on the mat",
    "The dog sat on the log"
]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print(tfidf_vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())
```

**Use Cases**: Information retrieval, document similarity, text classification

---

## Word Embeddings

**Problem with BoW/TF-IDF**: 
- Sparse representations
- No semantic meaning
- "king" and "queen" have no relationship

**Solution**: **Dense vector representations** that capture meaning

### Word2Vec (2013)

**Concept**: Words with similar contexts have similar meanings

**Two Architectures**:

**CBOW (Continuous Bag of Words)**:
- Predict center word from context
```
Context: "I ___ ice cream"
Predict: "like"
```

**Skip-gram**:
- Predict context from center word
```
Word: "like"
Predict context: ["I", "ice", "cream"]
```

**Cool Property**: **Vector arithmetic**!
```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
```

**Python Usage**:
```python
from gensim.models import Word2Vec

sentences = [
    ["I", "love", "NLP"],
    ["I", "love", "AI"],
    ["NLP", "is", "amazing"]
]

# Train Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Get vector
vector = model.wv['NLP']  # 100-dimensional vector

# Find similar words
similar = model.wv.most_similar('love', topn=5)
```

### GloVe (Global Vectors, 2014)

**Difference from Word2Vec**: Uses global word co-occurrence statistics

**Pre-trained Models**: Available for download (trained on Wikipedia, Common Crawl)

**Dimensions**: 50, 100, 200, 300

### FastText (2016)

**Innovation**: Character n-grams instead of whole words

**Advantage**: Handles out-of-vocabulary words

**Example**:
```
"running" = "run" + "nn" + "ing"
If "running" not seen, can approximate from sub  words
```

### Modern: Contextual Embeddings

**BERT, GPT, RoBERTa**: Context-aware embeddings

**Key Difference**: Same word, different vectors based on context

```
"bank" in "river bank" → Vector A
"bank" in "savings bank" → Vector B
```

---

## Basic NLP with Python

### Complete Example: Sentiment Analysis

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample dataset
texts = [
    "I love this product, it's amazing!",
    "Terrible experience, waste of money",
    "Best purchase ever, highly recommend",
    "Awful quality, very disappointed",
    "Great value for money",
    "Not worth it, returning immediately"
]

labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# Preprocessing function
def preprocess(text):
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Join back
    return ' '.join(tokens)

# Preprocess all texts
processed_texts = [preprocess(text) for text in texts]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    processed_texts, labels, test_size=0.3, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Predict
predictions = classifier.predict(X_test_vec)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Test on new text
new_review = "This is absolutely fantastic!"
new_processed = preprocess(new_review)
new_vec = vectorizer.transform([new_processed])
prediction = classifier.predict(new_vec)[0]
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

---

## Popular NLP Libraries

### 1. NLTK (Natural Language Toolkit)

**Purpose**: Educational, comprehensive toolkit

**Strengths**:
- Great for learning NLP concepts
- Extensive documentation
- Many datasets included

**Use Cases**: Education, prototyping, research

**Example**:
```python
import nltk

# Tokenize
from nltk.tokenize import word_tokenize
tokens = word_tokenize("Hello, how are you?")

# POS tagging
from nltk import pos_tag
tags = pos_tag(tokens)

# Named Entity Recognition
from nltk import ne_chunk
entities = ne_chunk(tags)
```

### 2. spaCy

**Purpose**: Industrial-strength NLP

**Strengths**:
- Fast (Cython implementation)
- Production-ready
- Pre-trained models
- Excellent documentation

**Use Cases**: Production systems, large-scale processing

**Example**:
```python
import spacy

# Load model
nlp = spacy.load("en_core_web_sm")

# Process text
doc = nlp("Apple Inc. was founded by Steve Jobs in California.")

# Entities
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

# POS tags
for token in doc:
    print(f"{token.text}: {token.pos_}")

# Dependency parsing
for token in doc:
    print(f"{token.text} → {token.dep_} → {token.head.text}")
```

### 3. Hugging Face Transformers

**Purpose**: State-of-the-art transformer models

**Strengths**:
- Pre-trained models (BERT, GPT, T5, etc.)
- Easy fine-tuning
- Large community

**Use Cases**: Transfer learning, modern NLP tasks

**Example**:
```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]

# Named Entity Recognition
ner = pipeline("ner", grouped_entities=True)
entities = ner("Apple Inc. was founded by Steve Jobs")

# Text generation
generator = pipeline("text-generation", model="gpt2")
text = generator("Once upon a time", max_length=50)
```

### 4. Gensim

**Purpose**: Topic modeling, document similarity

**Strengths**:
- Word2Vec, Doc2Vec
- Topic modeling (LDA)
- Document similarity

**Example**:
```python
from gensim.models import Word2Vec

# Train Word2Vec
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, min_count=1)

# Find similar words
similar = model.wv.most_similar("cat")
```

---

## Getting Started Guide

### Step 1: Prerequisites

**Python**: 3.7+

**Install Libraries**:
```bash
pip install nltk spacy scikit-learn gensim transformers

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('all')"
```

### Step 2: Learn NLP Fundamentals

**Tasks** (in order):
1. **Tokenization** → Split text into words
2. **Text Preprocessing** → Clean and normalize
3. **Text Classification** → Spam detection, sentiment
4. **Named Entity Recognition** → Extract entities
5. **Text Generation** → Simple text completion

### Step 3: First Project - Spam Classifier

**Goal**: Build email spam detector

**Steps**:
1. Get dataset (SMS Spam Collection)
2. Preprocess text
3. TF-IDF vectorization
4. Train Naive Bayes classifier
5. Evaluate performance

**Expected**: ~95% accuracy

### Step 4: Practice Projects

**Beginner**:
1. Sentiment analysis on movie reviews
2. Text summarization (extractive)
3. Language detection

**Intermediate**:
4. Named entity recognition
5. Question answering
6. Text generation with GPT-2

### Learning Resources

**Free Courses**:
- **DeepLearning.AI NLP Specialization** (Coursera)
- **Fast.ai - NLP** (free online)
- **Hugging Face Course** (free)

**Books**:
- "Speech and Language Processing" - Jurafsky & Martin (free online)
- "Natural Language Processing with Python" - Bird, Klein, Loper

**Datasets**:
- IMDB reviews (sentiment)
- SMS Spam Collection
- CoNLL (NER)
- SQuAD (question answering)
- Kaggle NLP datasets

---

## Common Challenges

### Challenge 1: Ambiguity

**Problem**: Same words, different meanings

**Example**:
```
"I saw her duck" 
- Duck (bird) or duck (action)?
```

**Solution**: Context, word embeddings, language models

### Challenge 2: Out-of-Vocabulary Words

**Problem**: Words not in training data

**Solutions**:
- Subword tokenization (BPE)
- Character-level models
- FastText (character n-grams)
- Fallback to "UNK" token

### Challenge 3: Sarcasm and Irony

**Problem**: "Great job!" could be genuine or sarcastic

**Solutions**:
- Context modeling
- Multimodal (text + tone/facial expression)
- Still an open research problem!

### Challenge 4: Low-Resource Languages

**Problem**: Limited training data for many languages

**Solutions**:
- Multilingual models (mBERT, XLM-R)
- Transfer learning
- Data augmentation
- Cross-lingual transfer

### Challenge 5: Domain Adaptation

**Problem**: Model trained on news fails on medical text

**Solutions**:
- Domain-specific pre-training
- Fine-tuning on domain data
- Domain adaptation techniques

---

## Key Takeaways

✅ **NLP** enables computers to understand and generate human language

✅ **Core tasks**: Classification, NER, sentiment analysis, translation, QA, summarization, generation

✅ **Preprocessing** is crucial: Tokenization, lowercasing, stopword removal, stemming/lemmatization

✅ **Representation methods**: Bag of Words → TF-IDF → Word Embeddings → Contextual Embeddings

✅ **Word Embeddings**: Dense vectors capturing semantic meaning (Word2Vec, GloVe, FastText)

✅ **Modern NLP**: Transformer-based (BERT, GPT) models dominate

✅ **Libraries**: NLTK (learning), spaCy (production), Hugging Face (state-of-the-art)

✅ **Start simple**: Text classification → NER → More complex tasks

✅ **Challenges**: Ambiguity, sarcasm, low-resource languages, domain adaptation

---

## What's Next?

Ready to dive deeper? Continue to:

📚 **[Intermediate NLP Techniques](02-intermediate-nlp-techniques.md)**  
Deep dive into POS tagging, NER, sentiment analysis, sequence models

📚 **[Advanced NLP Applications](03-advanced-nlp-applications.md)**  
Question answering, text generation, machine translation, multi-modal NLP

📚 **[LLM Basics](../Large-Language-Models/01-beginner-llm-basics.md)**  
Modern language models and their applications

📚 **[Transformers](../Transformers/01-beginner-transformer-basics.md)**  
The architecture powering modern NLP

---

## Practice Exercises

**Exercise 1**: Text Preprocessing Pipeline
```python
# Create complete preprocessing function
# Input: Raw text
# Output: Clean, tokenized text

def preprocess_text(text):
    # 1. Lowercase
    # 2. Tokenize
    # 3. Remove stopwords
    # 4. Lemmatize
    # 5. Remove punctuation
    pass
```

**Exercise 2**: Build Spam Classifier
- Use SMS Spam Collection dataset
- Achieve >90% accuracy
- Try both Naive Bayes and Logistic Regression

**Exercise 3**: Sentiment Analysis
- IMDB movie reviews dataset
- Compare BoW vs TF-IDF
- Try with BERT fine-tuning

**Exercise 4**: Named Entity Recognition
- Use spaCy on news articles
- Extract all people, organizations, locations
- Build knowledge graph

**Exercise 5**: Word Embeddings Exploration
- Train Word2Vec on your own corpus
- Find analogies (king - man + woman = ?)
- Visualize embeddings with t-SNE

---

**Congratulations!** 🎉 You now have a solid foundation in Natural Language Processing! Start practicing with real datasets and build your first NLP application.

---

*Next: [Intermediate NLP Techniques](02-intermediate-nlp-techniques.md)*
