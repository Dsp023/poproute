# Intermediate Prompt Techniques - Comprehensive Guide

## Table of Contents
1. [Chain-of-Thought Prompting](#chain-of-thought-prompting)
2. [Few-Shot Learning Mastery](#few-shot-learning-mastery)
3. [Role and Persona Prompting](#role-and-persona-prompting)
4. [Prompt Templates and Frameworks](#prompt-templates-and-frameworks)
5. [Contextual Prompting](#contextual-prompting)  
6. [Output Control Techniques](#output-control-techniques)
7. [Prompt Chaining](#prompt-chaining)
8. [Practical Implementation](#practical-implementation)
9. [Key Takeaways](#key-takeaways)

---

## Chain-of-Thought Prompting

### Deep Dive into CoT

**Chain-of-Thought (CoT)** prompting asks the model to show step-by-step reasoning before arriving at the final answer.

**Why It Works**:
- Makes reasoning explicit
- Catches logical errors
- Improves accuracy on complex tasks by 30-50%

### Zero-Shot CoT

**Magic phrase**: "Let's think step by step"

**Example - Math Problem**:
```
Problem: A restaurant has 23 tables. Each table can seat 4 people. 
If 78 people arrive, how many tables will be empty?

Standard prompt:
Q: [problem]
A: 4 tables (might be wrong)

Zero-Shot CoT:
Q: [problem]
Let's think step by step:
A: 
1. Each table seats 4 people
2. Total capacity: 23 × 4 = 92 people
3. People arriving: 78
4. Tables needed: 78 ÷ 4 = 19.5, so 20 tables (round up)
5. Empty tables: 23 - 20 = 3 tables

Answer: 3 tables ✅
```

**When to Use**:
- Math and arithmetic
- Logical reasoning
- Multi-step problems
- When accuracy is critical

### Few-Shot CoT

**Provide examples WITH reasoning steps**

**Template**:
```
[Problem 1]
Let's solve step by step:
1. [Step 1]
2. [Step 2]
3. [Step 3]
Answer: [Answer]

[Problem 2]
Let's solve step by step:
1. [Step 1]
2. [Step 2]
3. [Step 3]
Answer: [Answer]

[Your Problem]
Let's solve step by step:
```

**Example - Word Problems**:
```
Q: If a train travels 60 miles in 1 hour, how far will it travel in 2.5 hours?
Let's think step by step:
1. Speed = 60 miles per hour
2. Time = 2.5 hours
3. Distance = Speed × Time = 60 × 2.5 = 150 miles
Answer: 150 miles

Q: A store sells apples at $3 per pound. If you buy 4.5 pounds, how much do you pay?
Let's think step by step:
1. Price per pound = $3
2. Weight = 4.5 pounds
3. Total cost = $3 × 4.5 = $13.50
Answer: $13.50

Q: A swimming pool is being filled at 20 gallons per minute. How many hours will it 
take to fill a 1,200-gallon pool?
Let's think step by step:
```

### Self-Consistency CoT

**Generate multiple reasoning paths and take majority vote**

**Process**:
1. Generate 5-10 different CoT responses (use temperature > 0)
2. Extract final answers
3. Take most common answer

**Implementation**:
```python
from openai import OpenAI

client = OpenAI()

prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?

Let's think step by step:
"""

# Generate 5 different reasoning paths
responses = []
for i in range(5):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7  # Encourage variation
    )
    responses.append(response.choices[0].message.content)

# Extract answers and vote
# (Parse answers and find most common)
```

**Accuracy Boost**: 5-10% improvement over single CoT

### Least-to-Most Prompting

**Break complex problems into simpler sub-problems**

**Example**:
```
Complex: "What is the last letter of the word formed by concatenating 
the first letters of each word in the sentence 'Every good boy does fine'?"

Least-to-Most:
Step 1: What are the words in the sentence?
→ Every, good, boy, does, fine

Step 2: What is the first letter of each word?
→ E, g, b, d, f

Step 3: Concatenate these letters
→ Egbdf

Step 4: What is the last letter?
→ f
```

**Pattern**:
```
1. Decompose problem into sub-problems
2. Solve each sub-problem in sequence
3. Use previous answers in subsequent steps
4. Synthesize final answer
```

---

## Few-Shot Learning Mastery

### Optimal Number of Examples

**Rule of Thumb**:
- **Simple classification**: 2-3 examples
- **Complex patterns**: 5-7 examples
- **Structured output**: 3-4 examples
- **Edge cases**: 1 example per edge case

**Diminishing Returns**: Beyond 8-10 examples, improvement plateaus

### Example Selection Strategies

#### 1. Diverse Examples

**Cover different scenarios**:
```
Task: Classify email urgency (Low, Medium, High)

❌ Too similar:
"Need report by EOD" → High
"Need presentation by Friday" → High  
"Need data by tomorrow" → High

✅ Diverse:
"Need report by EOD" → High (deadline today)
"Let's schedule a meeting next week" → Low (flexible timing)
"System is down, clients affected!" → High (emergency)
"FYI: New policy update" → Low (informational)
```

#### 2. Edge Cases

**Include unusual but important cases**:
```
Sentiment analysis:
"I love this!" → Positive
"This is terrible" → Negative
"It's okay" → Neutral
"I don't NOT like it" → Positive (double negative!)
"Best worst decision ever" → Depends on context (ambiguous)
```

#### 3. Order Matters

**Most recent examples have more influence**

**Strategy**: Put most similar example last
```
Task: Classify customer query type

General examples (first)...
Most similar to actual query (last)
Actual query
```

### Few-Shot Format Best Practices

#### Format 1: Input-Output Pairs
```
Input: [example 1 input]
Output: [example 1 output]

Input: [example 2 input]
Output: [example 2 output]

Input: [your query]
Output:
```

#### Format 2: Q&A Style
```
Q: [question 1]
A: [answer 1]

Q: [question 2]
A: [answer 2]

Q: [your question]
A:
```

#### Format 3: Labeled Examples
```
### Example 1
[Description]
Result: [output]

### Example 2
[Description]
Result: [output]

### Your Task
[Description]
Result:
```

---

## Role and Persona Prompting

### Effective Role Assignment

**Format**:
```
You are a [role] with [qualifications/expertise] who [defines working style].

[Specific context about current task]

[Task]
```

**Detailed Example**:
```
You are a senior software architect with 15 years experience designing 
distributed systems at Fortune 500 companies. You excel at breaking down 
complex requirements into clean, scalable architectures. You prioritize 
maintainability and performance.

Context: We're building a real-time chat application expecting 100K concurrent users.

Task: Design the system architecture including database choice, caching strategy, 
and message queue approach. Explain trade-offs and reasoning.
```

### role Types

#### Expert Roles
```
"You are an expert [domain] with [X] years of experience"

Examples:
- "expert data scientist specializing in time-series forecasting"
- "seasoned product manager with B2B SaaS experience"
- "senior devops engineer specializing in Kubernetes"
```

#### Teaching Roles
```
"You are a patient tutor who explains concepts clearly using analogies"
"You are a university professor teaching [subject] to undergraduates"
```

#### Creative Roles
```
"You are a creative copywriter known for witty, engaging content"
"You are a storyteller in the style of [famous author]"
```

#### Analytical Roles
```
"You are a critical reviewer who identifies flaws and improvements"
"You are a detail-oriented code reviewer focusing on edge cases"
```

### Multi-Perspective Prompting

**Get different viewpoints on same problem**

**Example**:
```
Prompt 1 (Optimist):
"As an optimistic product manager, evaluate this feature idea focusing on 
potential benefits and opportunities."

Prompt 2 (Skeptic):
"As a critical product analyst, identify risks and potential issues with 
this feature idea."

Prompt 3 (neutral):
"As a balanced product strategist, provide an objective analysis of this 
feature idea covering both pros and cons."
```

**Synthesis**: Combine insights from all perspectives

---

## Prompt Templates and Frameworks

### LangChain Prompt Templates

**Basic Template**:
```python
from langchain import PromptTemplate

template = """
You are a helpful assistant specializing in {domain}.

User query: {query}

Provide a detailed response including:
1. Direct answer
2. Explanation
3. Example

Response:
"""

prompt = PromptTemplate(
    input_variables=["domain", "query"],
    template=template
)

# Use
filled_prompt = prompt.format(domain="Python programming", query="What are decorators?")
```

**Few-Shot Template**:
```python
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [
    {
        "question": "What is 2+2?",
        "answer": "Let's think:\n2 + 2 = 4\nAnswer: 4"
    },
    {
        "question": "What is 5*3?",
        "answer": "Let's think:\n5 * 3 = 15\nAnswer: 15"
    }
]

example_template = """
Question: {question}
{answer}
"""

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=example_template
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Solve these math problems step by step:",
    suffix="Question: {input}\nAnswer:",
    input_variables=["input"]
)
```

### Custom Framework: PREP

**P**urpose - **R**ole - **E**xamples - **P**arameters

```
Purpose: [What should this accomplish?]

Role: [Who is the AI in this scenario?]

Examples: [2-3 examples of desired output]

Parameters:
- Length: [word/character count]
- Format: [structure]
- Tone: [style]
- Constraints: [what to avoid]

Input: [your actual input]
Output:
```

**Example**:
```
Purpose: Generate professional email subject lines that increase open rates

Role: You are an email marketing specialist with 10 years of experience in B2B SaaS

Examples:
Email about product update → "New feature: Save 5 hours/week with automated reports"
Email about webinar → "Join 500+ marketers: Growth hacking webinar tomorrow"
Email about case study → "How Acme Corp increased conversions by 45%"

Parameters:
- Length: 40-60 characters
- Format: Action-oriented, specific benefit
- Tone: Professional but engaging
- Constraints: No clickbait, no excessive punctuation

Input: Email announcing new pricing tiers
Output:
```

---

## Contextual Prompting

### Providing Effective Context

**Types of Context**:

#### 1. Background Information
```
Context: I'm building a web app for freelance writers to track article pitches.
Current tech stack: React frontend, Node.js backend, PostgreSQL database.
Team: 2 developers (including me), 6-month timeline, $50K budget.

Question: Should I add real-time collaboration features in MVP?
```

#### 2. Conversation History
```
Previous conversation:
User: "I need to optimize my Python code"
AI: "Happy to help! What specific part are you looking to optimize?"
User: "The data processing loop takes 10 minutes on 1M rows"

Current question: "Here's my code: [code]. How can I speed it up?"
```

#### 3. Domain-Specific Context
```
Medical context: Patient has type 2 diabetes, on metformin 1000mg BID,
HbA1c 7.2%, no known allergies.

Question: [medical question]

Note: You need appropriate API/model for medical advice
```

### Context Window Management

**Problem**: Limited context window (e.g., 4K, 8K, 128K tokens)

**Strategies**:

#### 1. Summarize Previous Context
```
Previous conversation summary:
- User building e-commerce site
- Using React and Stripe
- Having issues with checkout flow
- Resolved payment processing errors
- Now working on order confirmation emails

Current question: [question about emails]
```

#### 2. Extract Relevant Context Only
```
❌ Include entire codebase (exceeds limit)

✅ Include just relevant functions and their dependencies
```

#### 3. Hierarchical summarization
```
For long documents:
1. Summarize each section
2. Summarize the summaries
3. Use final summary as context
```

---

## Output Control Techniques

### Controlling Length

**Specific Count**:
```
"Write exactly 500 words"
"Generate 5 bullet points"
"Provide 3 examples"
```

**Range**:
```
"Write 200-300 words"
"List 3-5 key points"
```

**Relative**:
```
"In one sentence, ..."
"In a short paragraph, ..."
"In a detailed essay, ..."
```

### Controlling Format

**Structured Data**:
```
"Provide your answer as valid JSON:
{
  "summary": "",
  "confidence": 0.0-1.0,
  "key_points": [],
  "recommendations": []
}"
```

**Tables**:
```
"Create a comparison table with these exact columns:
| Feature | Option A | Option B | Recommendation |"
```

**Code**:
```
"Provide only the function code, no explanation:
```python
def function_name():
    # your code here
```
"
```

### Controlling Tone

**Formal/Informal**:
```
Formal: "Compose a formal business proposal..."
Informal: "Write a casual blog post..."
```

**Technical Level**:
```
"Explain for a 5-year-old"
"Explain for a software engineer"
"Explain for a domain expert"
```

**Emotion**:
```
"Write in an enthusiastic, motivational tone"
"Write in a calm, reassuring tone"
"Write in a neutral, objective tone"
```

### Prohibiting Content

**What NOT to include**:
```
"Do not include:
- Personal opinions
- Information not in the source
- Technical jargon
- Promotional language"
```

**Safety Constraints**:
```
"If you don't know the answer, say 'I don't know' rather than guessing"
"If information is sensitive, decline politely"
```

---

## Prompt Chaining

### What is Prompt Chaining?

**Breaking complex tasks into sequential prompts**

**Why?**:
- Each step is simpler and more reliable
- Can verify intermediate results
- Easier to debug
- Can branch based on intermediate outputs

### Simple Chain Example

**Task**: Analyze customer reviews and create strategy

**Monolithic Approach** (❌):
```
"Read these 500 customer reviews, identify themes, sentiment, 
prioritize issues, and create action plan"
→ Overwhelming, inconsistent results
```

**Chained Approach** (✅):
```
Step 1: Summarize reviews
"Summarize these reviews into key themes"
→ Output: [5 key themes]

Step 2: Sentiment analysis
"For each theme, determine overall sentiment: [themes]"
→ Output: [themes with sentiment scores]

Step 3: Prioritization
"Rank these themes by: 1) negative sentiment, 2) frequency: [data]"
→ Output: [prioritized list]

Step 4: Action plan
"For top 3 issues, suggest concrete action items: [top issues]"
→ Output: [action plan]
```

### Implementation with Code

```python
from openai import OpenAI

client = OpenAI()

def prompt_chain(reviews_text):
    # Step 1: Extract themes
    step1_prompt = f"""
    Analyze these customer reviews and identify the 5 main themes:
    
    Reviews:
    {reviews_text}
    
    Format: List each theme with brief description.
    """
    
    themes = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": step1_prompt}]
    ).choices[0].message.content
    
    # Step 2: Sentiment for each theme
    step2_prompt = f"""
    For each theme below, determine sentiment (Positive/Negative/Mixed) 
    and score (-1 to 1):
    
    Themes:
    {themes}
    
    Format as JSON: {{"theme": "", "sentiment": "", "score": 0.0}}
    """
    
    sentiments = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": step2_prompt}]
    ).choices[0].message.content
    
    # Step 3: Create action plan
    step3_prompt = f"""
    Based on these sentiment-scored themes, create top 3 action items:
    
    {sentiments}
    
    For each action:
    - Clear description
    - Expected impact
    - Implementation difficulty (Low/Medium/High)
    """
    
    action_plan = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": step3_prompt}]
    ).choices[0].message.content
    
    return {
        "themes": themes,
        "sentiments": sentiments,
        "action_plan": action_plan
    }
```

### Conditional Chaining

**Branch based on intermediate results**:

```python
def analyze_code(code):
    # Step 1: Check if code has issues
    check_prompt = f"Does this code have any bugs or errors? Answer Yes or No first, then explain.\n\n{code}"
    check = llm(check_prompt)
    
    if "Yes" in check:
        # Step 2a: Debug
        debug_prompt = f"Fix the bugs in this code and explain changes:\n\n{code}"
        return llm(debug_prompt)
    else:
        # Step 2b: Optimize
        optimize_prompt = f"This code works correctly. Suggest performance optimizations:\n\n{code}"
        return llm(optimize_prompt)
```

---

## Practical Implementation

### Real-World Example: Content Generation Pipeline

```python
def content_pipeline(topic, target_audience, word_count):
    """Complete content generation with prompting chain"""
    
    # Step 1: Research & Outline
    outline_prompt = f"""
    Create a detailed outline for an article about "{topic}" 
    targeting {target_audience}.
    
    Include:
    - Hook/introduction angle
    - 3-5 main sections
    - Key points for each section
    - Conclusion angle
    
    Target length: {word_count} words total
    """
    outline = llm(outline_prompt)
    
    # Step 2: Write Introduction
    intro_prompt = f"""
    Based on this outline:
    {outline}
    
    Write an engaging introduction (150-200 words) that:
    - Hooks the reader
    - States the problem/question
    - Previews what they'll learn
    
    Target audience: {target_audience}
    """
    introduction = llm(intro_prompt)
    
    # Step 3: Write Each Section
    sections = []
    for section in extract_sections(outline):
        section_prompt = f"""
        Write the "{section}" section of the article.
        
        Outline context:
        {outline}
        
        Requirements:
        - {word_count // 5} words for this section
        - Include specific examples
        - Actionable insights
        - Natural flow from: {introduction if not sections else sections[-1]}
        
        Target audience: {target_audience}
        """
        sections.append(llm(section_prompt))
    
    # Step 4: Write Conclusion
    conclusion_prompt = f"""
    Write a conclusion that:
    - Summarizes key takeaways
    - Provides call-to-action
    - Leaves lasting impression
    
    Article summary:
    Intro: {introduction}
    Sections: {' '.join(sections[:2])}...  # Abbreviated
    
    Length: 150-200 words
    """
    conclusion = llm(conclusion_prompt)
    
    # Step 5: Final Polish
    final_article = f"{introduction}\n\n{'\n\n'.join(sections)}\n\n{conclusion}"
    
    polish_prompt = f"""
    Review and improve this article:
    
    {final_article}
    
    Improvements:
    1. Fix any grammatical errors
    2. Improve transitions between sections
    3. Ensure consistent tone
    4. Optimize for readability
    
    Provide the polished version.
    """
    
    polished = llm(polish_prompt)
    
    return {
        "outline": outline,
        "article": polished,
        "word_count": len(polished.split())
    }
```

---

## Key Takeaways

✅ **Chain-of-Thought**: "Let's think step by step" improves reasoning by 30-50%

✅ **Few-Shot Learning**: 
   - 2-3 examples for simple tasks
   - 5-7 for complex patterns
   - Diverse examples cover edge cases
   - Order matters (most relevant last)

✅ **Role Prompting**: Specific expertise + qualifications + working style

✅ **Prompt Templates**: Reusable structures for consistency (LangChain, custom frameworks)

✅ **Context Management**: Provide relevant background, manage token limits with summarization

✅ **Output Control**: Specify length, format, tone, constraints

✅ **Prompt Chaining**: Break complex tasks into sequential, manageable steps

✅ **Best Practice**: Test prompts iteratively, start simple, add complexity as needed

---

## Further Practice

**Exercise 1**: Create a chain for customer support
- Input: Customer complaint
- Output: Action plan with priority
- Steps: Classify → Extract details → Suggest solution → Create response

**Exercise 2**: Build few-shot examples  
- Task: Convert casual to formal email
- Create: 5 diverse examples covering different scenarios

**Exercise 3**: Template creation
- Recurring task you do often
- Build reusable template
- Test with 3 different inputs

---

*Previous: [Prompt Basics](01-beginner-prompt-basics.md) | Next: [Advanced Prompting](03-advanced-prompt-optimization.md)*
