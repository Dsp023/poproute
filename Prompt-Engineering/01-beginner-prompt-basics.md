# Prompt Engineering Basics - Comprehensive Beginner Guide

## Table of Contents
1. [What is Prompt Engineering?](#what-is-prompt-engineering)
2. [Why Prompt Engineering Matters](#why-prompt-engineering-matters)
3. [Basic Principles](#basic-principles)
4. [Core Prompting Techniques](#core-prompting-techniques)
5. [Prompt Structure](#prompt-structure)
6. [Common Patterns](#common-patterns)
7. [Common Mistakes](#common-mistakes)
8. [Best Practices](#best-practices)
9. [Practical Examples](#practical-examples)
10. [Tools and Resources](#tools-and-resources)
11. [Key Takeaways](#key-takeaways)

---

## What is Prompt Engineering?

**Prompt Engineering** is the art and science of crafting effective instructions (prompts) for AI language models to produce desired outputs.

### Simple Definition

**Prompt**: The text you input to an AI model
**Engineering**: Systematic approach to optimize prompts

**Analogy**: Just like asking a question to a human expert:
- Vague question → Vague answer
- Well-formulated question → Precise, useful answer

### Why It's a "Thing"

**Same model, different results**:
```
Prompt 1: "Write about AI"
Output: Generic 200-word essay

Prompt 2: "Write a 500-word technical article explaining transformer architecture 
for software engineers, including code examples and diagrams"
Output: Detailed, technical, structured article with code
```

**The difference?** The prompt!

---

## Why Prompt Engineering Matters

### Real-World Impact

**Productivity Gains**:
- **Before**: 30 minutes to write email
- **After** (good prompt): 2 minutes to review and send AI-generated email

**Cost Savings**:
- Better prompts = fewer API calls
- Get it right first time instead of multiple attempts

**Quality Improvements**:
- 70% useful output → 95% useful output
- Just by improving the prompt

### Business Applications

**Customer Service**:
- Chatbot response quality directly affects customer satisfaction
- Good prompts = fewer escalations to humans

**Content Creation**:
- Marketing teams using AI for copy
- Prompt quality = content quality

**Code Generation**:
- GitHub Copilot, cursor effectiveness
- Better prompts = better code suggestions

**Data Analysis**:
- Prompt LLMs to analyze data, generate insights
- Clear prompts = accurate analysis

---

## Basic Principles

### 1. Be Specific and Clear

The more specific you are, the better the output.

**Vague Examples**:
```
❌ "Tell me about dogs"
❌ "Write code"
❌ "Summarize this"
```

**Specific Examples**:
```
✅ "List 5 characteristics of Golden Retrievers, including:
   - Temperament
   - Exercise needs (hours per day)
   - Average size (weight and height)
   - Common health issues
   - Grooming requirements
   Format as a bulleted list."

✅ "Write a Python function that takes a list of numbers and returns 
   the median. Include:
   - Function name: calculate_median
   - Type hints
   - Docstring
   - Handle edge cases (empty list)
   - Include unit tests using pytest"

✅ "Summarize this 500-word article in exactly 3 bullet points, 
   each focusing on a different main idea. Each bullet should be 
   1-2 sentences maximum."
```

**Impact**:
- Vague: 50% chance of getting what you want
- Specific: 90%+ chance

### 2. Provide Context

Context helps the AI understand your situation and tailor the response.

**No Context**:
```
❌ "Write an email"
```

**With Context**:
```
✅ "Write a professional email to my manager requesting a meeting.

Context:
- I'm a software engineer
- Want to discuss concerns about Q3 project timeline
- Manager's name: Sarah
- Prefer meeting next week (week of March 15)
- Tone: Professional but friendly
- Length: 3-4 short paragraphs"
```

**Context Elements to Include**:
- **Audience**: Who is this for?
- **Purpose**: What should this accomplish?
- **Constraints**: Length, format, tone
- **Background**: Relevant information
- **Domain**: Technical level, industry

### 3. Use Examples (Few-Shot Learning)

Show the AI what you want with examples.

**Zero-Shot** (No Examples):
```
Classify the sentiment of: "This movie was okay"
```

**Few-Shot** (With Examples):
```
Classify sentiment as Positive, Negative, or Neutral:

"I absolutely loved this movie!" → Positive
"Worst movie I've ever seen" → Negative
"The movie was okay, nothing special" → Neutral
"This movie exceeded all my expectations!" → Positive

"This product is decent for the price" → ?
```

**Why It Works**:
- Shows patterns
- Clarifies format
- Reduces ambiguity

**How Many Examples?**
- 1-2: Simple tasks
- 3-5: Complex patterns
- 5+: Rarely needed, may hit token limits

### 4. Specify Output Format

Tell the AI exactly how you want the output structured.

**Without Format**:
```
❌ "Compare Python and JavaScript"
```

**With Format**:
```
✅ "Compare Python and JavaScript in a table:

| Feature | Python | JavaScript |
|---------|--------|------------|
| Typing | | |
| Primary Use | | |
| Performance | | |
| Learning Curve | | |
| Popular Frameworks | | |

Fill in each cell with 1-2 sentences."
```

**Format Options**:
- **Tables**: Structured comparisons
- **JSON**: Structured data
```json
{
  "summary": "",
  "key_points": [],
  "sentiment": ""
}
```
- **Markdown**: Headers, lists, code blocks
- **Code**: Specific language syntax
- **XML/HTML**: Web content

### 5. Set Tone and Style

Guide the AI's writing style.

**Examples**:
```
"Explain quantum computing:"

→ "like I'm 5 years old"
→ "as an academic paper"
→ "in a casual, friendly tone"
→ "as a technical documentation for engineers"
→ "using analogies a marketer would understand"
→ "in the style of a Twitter thread"
```

**Tone Modifiers**:
- **Formal**: "Write in professional business language"
- **Casual**: "Use conversational, friendly tone"
- **Technical**: "Use technical terminology, assume expert audience"
- **Simple**: "Use simple language, avoid jargon"
- **Persuasive**: "Write convincingly to persuade readers"
- **Educational**: "Explain clearly with examples for learners"

---

## Core Prompting Techniques

### 1. Zero-Shot Prompting

**Definition**: Ask the model to perform a task without any examples

**When to Use**: Simple, well-defined tasks

**Example**:
```
Translate the following English text to French:
"Hello, how are you today?"
```

**Pros**:
- Simple and fast
- No need for examples
- Works well for common tasks

**Cons**:
- May not understand complex or specific patterns
- Less control over format

### 2. Few-Shot Prompting

**Definition**: Provide examples of input-output pairs

**Format**:
```
[Task Description]

Example 1:
Input: [example input 1]
Output: [example output 1]

Example 2:
Input: [example input 2]
Output: [example output 2]

Now complete:
Input: [your actual input]
Output:
```

**Real Example - Email Classification**:
```
Classify emails as: Sales, Support, or General

Email: "Hi, I'd like to buy 10 licenses"
Category: Sales

Email: "My account isn't working properly"
Category: Support

Email: "What are your office hours?"
Category: General

Email: "I'm interested in your enterprise plan"
Category:
```

**Benefits**:
- Shows exact format wanted
- Clarifies edge cases
- Better accuracy than zero-shot

### 3. Chain-of-Thought (CoT) Prompting

**Definition**: Ask the model to show its reasoning step-by-step

**Magic Phrase**: "Let's think step by step"

**Example - Math Problem**:
```
Without CoT:
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?
A: 11 ✅ (but no reasoning)

With CoT:
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?

Let's solve this step by step:
1. Roger starts with 5 tennis balls
2. He buys 2 cans of tennis balls
3. Each can contains 3 tennis balls
4. So 2 cans contain: 2 × 3 = 6 tennis balls
5. Total tennis balls: 5 + 6 = 11 tennis balls

A: 11 tennis balls ✅ (with clear reasoning)
```

**When to Use**:
- Math problems
- Logical reasoning
- Complex multi-step tasks
- When you need to verify reasoning

**Accuracy Improvement**:
- Simple tasks: +5-10%
- Complex reasoning: +30-50%!

### 4. Role Prompting

**Definition**: Assign a specific role/persona to the AI

**Format**:
```
You are a [role] with [qualifications].
[Task]
```

**Examples**:
```
"You are a senior Python developer with 10 years of experience 
specializing in web scraping and data processing.

Review this code and suggest improvements:
[code]"

"You are a professional copywriter specializing in SaaS marketing.

Write a compelling headline for a project management tool 
that emphasizes team collaboration."

"You are an expert data scientist familiar with scikit-learn.

Help me choose the right machine learning algorithm for 
predicting customer churn with the following dataset characteristics:
- 50,000 samples
- 25 features (mix of categorical and numerical)
- Binary classification
- Class imbalance (10% churn rate)"
```

**Why It Works**:
- Activates relevant knowledge
- Sets appropriate tone
- Improves domain-specific accuracy

### 5. Instruction Following

**Clear, direct commands**:

**Good Instructions**:
```
✅ "List three advantages and three disadvantages"
✅ "Write exactly 500 words"
✅ "Include a code example in Python"
✅ "Do not use technical jargon"
✅ "Cite sources if making factual claims"
```

**Structure**:
```
[Action Verb] + [Subject] + [Constraints] + [Format]

Examples:
"Summarize [this article] in [100 words] as [bullet points]"
"Create [a function] that [calculates median] in [Python] with [type hints]"
"Explain [recursion] like [I'm a beginner] using [a real-world analogy]"
```

---

## Prompt Structure

### Recommended Template

```
[Role/Context] (Optional)
You are an expert [role] with [qualifications]

[Task] (Required)
[Clear instruction about what to do]

[Context/Input] (Optional but recommended)
[Background information or input data]

[Constraints] (Optional but recommended)
- Length: [word count or character limit]
- Format: [desired output format]
- Tone: [desired tone/style]
- Exclude: [what not to include]

[Examples] (Optional for complex tasks)
Example 1: [input] → [output]
Example 2: [input] → [output]

[Output Format] (Optional but recommended)
[Specific structure for response]
```

### Example Using Template

```
[Role]
You are a technical writer creating documentation for software engineers.

[Task]
Write API documentation for a user authentication endpoint.

[Context]
Endpoint: POST /api/auth/login
Parameters: email (string), password (string)
Returns: JWT token if successful, error message if failed

[Constraints]
- Include: Description, parameters, example request, example response, error codes
- Format: Markdown
- Length: 200-300 words
- Tone: Technical but clear

[Output Format]
## Endpoint Name
Description

### Parameters
| Name | Type | Required | Description |
...

### Example Request
```
[code]
```

### Example Response
...

### Error Codes
...
```

---

## Common Patterns

### Pattern 1: Information Extraction

```
Extract the following information from the text:
- [Field 1]
- [Field 2]
- [Field 3]

Text: """
[input text]
"""

Format as JSON.
```

**Example**:
```
Extract these details from the job posting:
- Company name
- Job title
- Location
- Salary range (if mentioned)
- Required years of experience

Job Posting: """
Google is hiring a Senior Software Engineer in Mountain View, CA.
5+ years of experience required. Competitive salary.
"""

Output as JSON.
```

### Pattern 2: Text Transformation

```
Transform the following [format A] to [format B]:

[Input]

[Optional: Show example transformation]
```

**Example**:
```
Convert this Python code to equivalent JavaScript:

Python:
def calculate_sum(numbers):
    return sum(numbers)

JavaScript:
```

### Pattern 3: Comparison

```
Compare [A] and [B] in terms of:
- [Criterion 1]
- [Criterion 2]
- [Criterion 3]

Present as: [format]
```

### Pattern 4: Iterative Refinement

```
[Task]

Then, review your response and improve it by:
- [Improvement criterion 1]
- [Improvement criterion 2]

Provide both the initial version and the improved version.
```

**Example**:
```
Write a headline for a fitness app.

Then, review and create 3 variations that:
- Are more attention-grabbing
- Emphasize different benefits
- Use power words
```

### Pattern 5: Error Handling

```
[Task]

Important:
- If you don't know the answer, say "I don't know" instead of guessing
- If information is ambiguous, ask for clarification
- If task is impossible, explain why
```

---

## Common Mistakes

### Mistake 1: Being Too Vague

**Problem**:
```
❌ "Tell me about machine learning"
```

**Solution**:
```
✅ "Explain the difference between supervised and unsupervised learning 
   in machine learning. Include:
   - Definition of each
   - 2 examples of each
   - When to use which
   Target audience: Software engineers with no ML background
   Length: 300 words"
```

### Mistake 2: No Constraints

**Problem**:
```
❌ "Write a blog post about productivity"
```
*(Could be 100 words or 5,000 words, any tone, any format)*

**Solution**:
```
✅ "Write a 800-word blog post about productivity tips for remote workers.
   - Include 5 actionable tips
   - Each tip: 1 paragraph with example
   - Tone: Friendly and encouraging
   - Target: Remote workers in tech
   - Format: Introduction, 5 tips, conclusion"
```

### Mistake 3: Unrealistic Expectations

**Problem**:
```
❌ "Analyze this 50-page PDF and create a complete business strategy"
```
*(Too complex, vague, and exceeds context window)*

**Solution**:
```
✅ Break into steps:
   1. "Summarize key findings from pages 1-20"
   2. "Identify main challenges mentioned"
   3. "Based on [challenges], suggest 3 strategic priorities"
```

### Mistake 4: Not Iterating

**Problem**: Using first prompt, getting poor result, giving up

**Solution**: Iterate!
```
Version 1: "Write an email to my team"
Result: Too formal

Version 2: "Write a casual email to my engineering team"
Result: Better, but too long

Version 3: "Write a casual, brief email (3-4 sentences) to my 
engineering team announcing a new meeting schedule"
Result: Perfect!
```

### Mistake 5: Ignoring Model Limitations

**Problem**:
```
❌ "What happened yesterday in the stock market?"
```
*(Model has knowledge cutoff)*

**Solution**:
```
✅ Provide the data:
"Given this stock market data from yesterday [paste data], 
analyze the key trends and notable movements"
```

---

## Best Practices

### ✅ DO's

**1. Start Simple, Add Complexity**
```
Version 1: "Summarize this article"
Version 2: "Summarize this article in 100 words"
Version 3: "Summarize this article in 100 words, focusing on key findings"
```

**2. Use Delimiters**
```
Separate sections clearly:

Article: """
[long article text]
"""

Questions:
1. What is the main argument?
2. What evidence is provided?
```

Common delimiters: `"""`, `###`, `---`, `<article>...</article>`

**3. Be Explicit About What NOT to Do**
```
"Summarize this article.
Do NOT:
- Include your opinions
- Add information not in the article
- Exceed 150 words"
```

**4. Request Self-Verification**
```
"[Complete task]

Then, review your answer and verify:
1. Does it answer the question?
2. Is the format correct?
3. Are there any errors?

Provide both the answer and your verification."
```

**5. Test with Edge Cases**
```
Test your prompt with:
- Very short inputs
- Very long inputs
- Ambiguous inputs
- Multiple valid interpretations
```

### ❌ DON'Ts

**1. Don't Assume Context**
- Model doesn't remember previous conversations (unless explicitly provided)
- Always include necessary context

**2. Don't Use Ambiguous Language**
```
❌ "Make it better"
✅ "Improve by: 1) Adding specific examples, 2) Using simpler language, 3) Making it more concise"
```

**3. Don't Ignore Privacy**
```
❌ Don't paste: SSNs, passwords, private keys, PII
✅ Use: Sanitized examples, dummy data
```

---

## Practical Examples

### Example 1: Code Review

**Poor Prompt**:
```
❌ "Review this code"
```

**Good Prompt**:
```
✅ "Review this Python function for:
1. Code quality and best practices
2. Potential bugs
3. Performance issues
4. Readability improvements

For each issue found, provide:
- Line number
- Issue description
- Suggested fix with code example

Code:
```python
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
    return result
```
```

### Example 2: Data Analysis

**Poor Prompt**:
```
❌ "Analyze this data"
```

**Good Prompt**:
```
✅ "Analyze this customer churn data and provide:

1. Summary statistics (mean, median, distribution)
2. Key patterns or trends
3. Potential correlations
4. 3 actionable recommendations to reduce churn

Data:
[CSV data or summary]

Format your response with clear headers for each section.
Include specific numbers to support your findings."
```

### Example 3: Content Creation

**Poor Prompt**:
```
❌ "Write a product description"
```

**Good Prompt**:
```
✅ "Write a product description for our noise-canceling headphones.

Target audience: Remote workers and students
Key features to highlight:
- 40-hour battery life
- Active noise cancellation
- Comfortable for all-day wear
- $149 price point

Tone: Professional but approachable
Length: 150-200 words
Structure:
- Hook (1-2 sentences)
- Features and benefits (2-3 paragraphs)
- Call to action (1 sentence)

Emphasize value proposition: Productivity and focus."
```

---

## Tools and Resources

### Prompt Testing Platforms

**OpenAI Playground**:
- Test different models
- Adjust temperature, tokens
- Compare outputs
- URL: platform.openai.com/playground

**Claude.ai**:
- Free access to Claude
- Long context window (200K tokens)
- Good for testing complex prompts

**Poe.com**:
- Test multiple models (GPT-4, Claude, Gemini, etc.)
- Compare outputs side-by-side
- Free tier available

**Google AI Studio**:
- Test Gemini models
- Structured prompts
- Free access

### Prompt Libraries

**Awesome ChatGPT Prompts**:
- GitHub repository
- Community-contributed prompts
- Categories: Writing, coding, education, etc.

**PromptBase**:
-Marketplace for prompts
- Paid and free prompts
- Quality-rated

**FlowGPT**:
- Community prompts
- Voting system
- Popular use cases

### Learning Resources

**Courses**:
- **Learn Prompting** (learnprompting.org): Free comprehensive course
- **DeepLearning.AI ChatGPT Prompt Engineering**: Short course by Andrew Ng
- **Prompt Engineering Guide** (promptingguide.ai): Documentation-style guide

**Official Documentation**:
- OpenAI Prompt Engineering Guide
- Anthropic Prompt Engineering docs
- Google Gemini prompting best practices

**Communities**:
- r/PromptEngineering (Reddit)
- Discord servers for various AI tools
- Twitter/X hashtags: #PromptEngineering

---

## Key Takeaways

✅ **Prompt engineering** = Crafting effective instructions for AI models

✅ **Impact**: Same model, 10x better results with better prompts

✅ **Core principles**:
   - Be specific and clear
   - Provide context
   - Use examples (few-shot)
   - Specify format
   - Set tone/style

✅ **Techniques**:
   - Zero-shot: Simple tasks
   - Few-shot: Show patterns
   - Chain-of-Thought: Step-by-step reasoning
   - Role prompting: Activate expertise

✅ **Structure**: Role → Task → Context → Constraints → Format → Examples

✅ **Common mistakes**:
   - Too vague
   - No constraints
   - Not iterating
   - Unrealistic expectations

✅ **Best practices**:
   - Start simple, add complexity
   - Use delimiters
   - Test edge cases
   - Request verification
   - Iterate based on results

✅ **Remember**: Prompting is iterative—first attempt rarely perfect!

---

## Practice Exercises

**Exercise 1**: Improve these prompts
```
❌ "Write code to sort a list"
→ Your improved version: ?

❌ "Explain AI"
→ Your improved version: ?
```

**Exercise 2**: Create a prompt for:
- Extracting meeting action items from transcript
- Converting informal email to professional tone
- Generating unit tests for a function

**Exercise 3**: Test prompt variations
- Take one task
- Create 3 different prompts
- Compare outputs
- Identify what works best

**Exercise 4**: Build a prompt template
- For a recurring task you do
- Make it reusable with placeholders
- Test with 3 different inputs

---

**Congratulations!** 🎉 You now understand prompt engineering fundamentals. Start practicing with real tasks and iterate to find what works best for you!

---

*Next: [Intermediate Prompt Techniques](02-intermediate-prompt-techniques.md)*
