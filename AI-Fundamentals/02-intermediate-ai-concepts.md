# Intermediate AI Concepts

## Table of Contents
1. [Introduction](#introduction)
2. [AI Problem-Solving Approaches](#ai-problem-solving-approaches)
3. [Search Algorithms](#search-algorithms)
4. [Knowledge Representation](#knowledge-representation)
5. [Expert Systems](#expert-systems)
6. [AI Agents and Environments](#ai-agents-and-environments)
7. [Reasoning Under Uncertainty](#reasoning-under-uncertainty)
8. [Planning and Decision Making](#planning-and-decision-making)
9. [Practical Applications](#practical-applications)
10. [Key Takeaways](#key-takeaways)

---

## Introduction

Now that you understand AI basics, this guide explores how AI systems actually solve problems. We'll cover the fundamental techniques and algorithms that power intelligent systems.

**Prerequisites**: 
- Understanding of [AI Basics](01-beginner-ai-basics.md)
- Basic programming knowledge helpful but not required
- Logical thinking and problem-solving mindset

---

## AI Problem-Solving Approaches

### Problem Formulation

AI problem-solving begins with clearly defining the problem:

**Components of a Well-Defined Problem**:
1. **Initial State**: Where we start
2. **Goal State**: Where we want to end up
3. **Actions**: Available moves/operations
4. **Transition Model**: How actions change states
5. **Path Cost**: Cost of taking actions

**Example: Route Finding**
- **Initial State**: Current location (e.g., New York)
- **Goal State**: Destination (e.g., San Francisco)
- **Actions**: Drive on available roads
- **Transition Model**: Moving from one city to another via roads
- **Path Cost**: Distance, time, or fuel consumption

### Types of Problems

#### 1. **Search Problems**
Finding a sequence of actions to reach a goal.
- Route planning
- Puzzle solving (8-puzzle, Rubik's cube)
- Game playing (chess, checkers)

#### 2. **Optimization Problems**
Finding the best solution among many possibilities.
- Resource allocation
- Scheduling
- Portfolio optimization

#### 3. **Constraint Satisfaction Problems (CSP)**
Finding assignments that satisfy all constraints.
- Sudoku
- Map coloring
- Class scheduling

#### 4. **Adversarial Search Problems**
Problems involving opponents (game theory).
- Chess
- Poker
- Competitive bidding

---

## Search Algorithms

Search algorithms are fundamental to AI problem-solving. They systematically explore possible solutions.

### Uninformed Search Strategies

These strategies have no additional information about the goal beyond the problem definition.

#### 1. Breadth-First Search (BFS)

**How it works**:
- Explore all nodes at depth d before exploring depth d+1
- Uses a queue (FIFO - First In, First Out)

**Properties**:
- ✅ Complete: Always finds a solution if one exists
- ✅ Optimal: Finds shortest path (if costs are equal)
- ❌ Space complexity: Can use lots of memory

**Use Case**: Finding shortest path in unweighted graphs

**Example**: Social network connections (finding shortest friend chain)

```
Start → [Level 1] → [Level 2] → [Level 3] → Goal
```

#### 2. Depth-First Search (DFS)

**How it works**:
- Explore as far as possible down one branch before backtracking
- Uses a stack (LIFO - Last In, First Out)

**Properties**:
- ❌ Not complete: Can get stuck in infinite paths
- ❌ Not optimal: May not find shortest path
- ✅ Space efficient: Only stores path from root to current node

**Use Case**: Maze solving, topological sorting

**Example**: Exploring a file system directory structure

```
Start → Deep path 1 → Deeper → Deepest
     ↓ (backtrack)
     → Deep path 2 → ...
```

#### 3. Uniform Cost Search (UCS)

**How it works**:
- Expands node with lowest path cost
- Uses a priority queue

**Properties**:
- ✅ Complete and optimal
- ❌ Can be slow if solution is at great depth

**Use Case**: Finding cheapest path when costs differ

**Example**: Finding cheapest flight route

### Informed Search Strategies (Heuristic Search)

These use problem-specific knowledge to find solutions more efficiently.

#### 1. Greedy Best-First Search

**How it works**:
- Expands node that appears closest to goal (based on heuristic)
- Uses heuristic function h(n) estimating cost to goal

**Properties**:
- ❌ Not optimal: Can be misled by heuristic
- ✅ Fast in practice

**Example**: GPS navigation using straight-line distance to destination

#### 2. A* Search ⭐ (Most Important)

**How it works**:
- Combines actual cost g(n) and heuristic h(n)
- Evaluation function: f(n) = g(n) + h(n)
  - g(n): Cost from start to current node
  - h(n): Estimated cost from current node to goal

**Properties**:
- ✅ Complete and optimal (with admissible heuristic)
- ✅ Optimally efficient
- Most widely used informed search algorithm

**Example**: Video game pathfinding, route planning

**Visual Example**:
```
Start (0+10=10) → Node A (3+6=9) → Node B (5+2=7) → Goal (7+0=7)
                ↓                                      ↑
              Node C (4+8=12)  ──────────────────────→ (not explored)
```

#### 3. Heuristic Functions

**Admissible Heuristic**: Never overestimates actual cost to goal
- Example: Straight-line distance (always ≤ actual road distance)

**Consistent Heuristic**: Satisfies triangle inequality
- h(n) ≤ cost(n,a,n') + h(n')

**Common Heuristics**:
- **Manhattan Distance**: Sum of absolute differences (grid-based)
- **Euclidean Distance**: Straight-line distance
- **Hamming Distance**: Number of misplaced elements

### Local Search Algorithms

For problems where the path doesn't matter, only the final state.

#### 1. Hill Climbing
- Move to the best neighboring state
- Can get stuck in local maxima
- Fast but not complete

#### 2. Simulated Annealing
- Probabilistically accept worse moves to escape local maxima
- "Temperature" parameter decreases over time
- More robust than hill climbing

#### 3. Genetic Algorithms
- Population-based search
- Uses selection, crossover, mutation
- Inspired by biological evolution

**Use Cases**: Optimization problems, scheduling, design

---

## Knowledge Representation

AI systems need ways to represent and reason about knowledge.

### Knowledge Representation Methods

#### 1. Propositional Logic

**Basics**:
- Statements that are true or false
- Logical operators: AND (∧), OR (∨), NOT (¬), IMPLIES (→)

**Example**:
```
P: "It is raining"
Q: "The ground is wet"
Rule: P → Q (If it's raining, then the ground is wet)
```

**Use Case**: Simple rule-based systems

#### 2. First-Order Logic (Predicate Logic)

**Enhanced expressiveness**:
- Objects, properties, relations
- Quantifiers: ∀ (for all), ∃ (there exists)

**Example**:
```
∀x (Dog(x) → Animal(x))
"All dogs are animals"

∃x (Cat(x) ∧ Friendly(x))
"There exists a friendly cat"
```

**Use Case**: Complex reasoning systems, knowledge bases

#### 3. Semantic Networks

**Graph-based representation**:
- Nodes represent concepts
- Edges represent relationships

**Example**:
```
[Dog] --is-a--> [Animal]
  |
  has-property --> [4 legs]
```

**Use Case**: Conceptual understanding, natural language processing

#### 4. Frames

**Structured knowledge representation**:
- Frames represent objects or concepts
- Slots hold attributes and values

**Example**:
```
Frame: Car
  - Type: Vehicle
  - Has: 4 wheels
  - Requires: Fuel or Electricity
  - Can: Transport passengers
```

**Use Case**: Expert systems, knowledge management

#### 5. Ontologies

**Formal naming and definition of concepts**:
- Hierarchical organization
- Relationships between concepts
- Widely used in semantic web

**Example**:
```
Animal
  ├── Mammal
  │   ├── Dog
  │   └── Cat
  └── Bird
      ├── Eagle
      └── Sparrow
```

**Use Case**: Knowledge graphs, semantic search, information integration

---

## Expert Systems

AI systems that mimic human expert decision-making in specific domains.

### Architecture of Expert Systems

#### 1. **Knowledge Base**
- Domain-specific facts and rules
- Represents expert knowledge

#### 2. **Inference Engine**
- Reasoning mechanism
- Applies rules to known facts

#### 3. **User Interface**
- Interaction with users
- Asks questions, provides explanations

#### 4. **Explanation Facility**
- Justifies conclusions
- Shows reasoning process

### Inference Methods

#### Forward Chaining (Data-Driven)

**Process**:
1. Start with known facts
2. Apply rules to derive new facts
3. Continue until goal reached or no new facts

**Example**:
```
Facts: Patient has fever, cough
Rules: 
  IF fever AND cough THEN possibly flu
  IF possibly flu AND no vaccination THEN high risk
Conclusion: High risk
```

**Use Case**: Diagnosis, monitoring, prediction

#### Backward Chaining (Goal-Driven)

**Process**:
1. Start with goal
2. Find rules that conclude the goal
3. Work backwards to verify premises

**Example**:
```
Goal: Is patient at high risk?
Check: Is there possibly flu?
  Check: Is there fever AND cough?
    Facts confirm: Yes
Conclusion: High risk
```

**Use Case**: Question-answering, planning

### Famous Expert Systems

1. **MYCIN** (1970s)
   - Medical diagnosis for bacterial infections
   - Performed at expert level
   - Pioneered explanation facilities

2. **DENDRAL** (1965)
   - Chemical structure analysis
   - First successful expert system

3. **XCON** (1980s)
   - Computer system configuration
   - Saved Digital Equipment Corporation millions

---

## AI Agents and Environments

### What is an AI Agent?

An **agent** is anything that perceives its environment through sensors and acts upon it through actuators.

**Components**:
- **Sensors**: Perceive the environment (cameras, microphones, data feeds)
- **Actuators**: Act on the environment (motors, displays, outputs)
- **Agent Function**: Maps perceptions to actions

### Types of Agents

#### 1. Simple Reflex Agents
- Act based on current percept only
- Condition-action rules
- No memory of past

**Example**: Thermostat (if temp > 72°F, turn on AC)

#### 2. Model-Based Reflex Agents
- Maintain internal state
- Track aspects of environment not currently perceived
-

**Example**: Vacuum cleaner remembering which rooms are clean

#### 3. Goal-Based Agents
- Have explicit goals
- Search and plan to achieve goals
- More flexible than reflex agents

**Example**: Navigation system finding route to destination

#### 4. Utility-Based Agents
- Have utility function measuring "happiness"
- Choose actions that maximize expected utility
- Handle trade-offs

**Example**: Autonomous car balancing safety, speed, comfort, fuel efficiency

#### 5. Learning Agents
- Improve performance through experience
- Have learning element modifying behavior
- Most sophisticated agent type

**Example**: AlphaGo learning to play Go through self-play

### Environment Types

#### Fully Observable vs Partially Observable
- **Fully**: Agent can see entire state (chess)
- **Partially**: Limited perception (poker, real world)

#### Deterministic vs Stochastic
- **Deterministic**: Actions have predictable outcomes (chess)
- **Stochastic**: Outcomes involve randomness (dice games, real world)

#### Episodic vs Sequential
- **Episodic**: Each action is independent (image classification)
- **Sequential**: Current action affects future (chess, driving)

#### Static vs Dynamic
- **Static**: Environment doesn't change while agent deliberates (crossword)
- **Dynamic**: Environment changes (real-time games, stock trading)

#### Discrete vs Continuous
- **Discrete**: Finite number of states/actions (chess)
- **Continuous**: Infinite possibilities (driving, robot control)

---

## Reasoning Under Uncertainty

Real-world problems involve uncertainty. AI must handle incomplete information and probabilistic outcomes.

### Probability Basics in AI

#### Bayesian Reasoning

**Bayes' Theorem**:
```
P(A|B) = P(B|A) × P(A) / P(B)
```

**Use in AI**: Update beliefs based on new evidence

**Example**: Medical diagnosis
```
P(Disease|Symptom) = P(Symptom|Disease) × P(Disease) / P(Symptom)
```

#### Bayesian Networks

**Directed acyclic graph**:
- Nodes represent variables
- Edges represent dependencies
- Efficient probabilistic reasoning

**Example**:
```
[Weather] → [Traffic] → [Arrival Time]
                ↓
            [Accident]
```

**Use Case**: Diagnosis, prediction, decision support

### Fuzzy Logic

**Handle imprecise information**:
- Partial truth values (not just 0 or 1)
- Linguistic variables (hot, warm, cold)

**Example**: Temperature control
- "If temperature is warm, turn fan to medium"
- Warm might be 0.7 true at 75°F

**Use Case**: Control systems, consumer electronics

---

## Planning and Decision Making

### Classical Planning

**STRIPS Representation**:
- **State**: Set of conditions
- **Actions**: Preconditions and effects
- **Goal**: Desired state

**Example**: Robot navigation
```
Action: Move(from, to)
  Preconditions: At(robot, from), Path(from, to)
  Effects: At(robot, to), ¬At(robot, from)
```

### Decision Theory

**Utility Theory**:
- Assign values to outcomes
- Choose action maximizing expected utility

**Decision Trees**:
- Visual representation of choices and outcomes
- Calculate expected value of each path

### Markov Decision Processes (MDPs)

**Framework for sequential decision-making**:
- States, actions, transition probabilities
- Rewards for state-action pairs
- Goal: Find optimal policy

**Use Case**: Robotics, game AI reinforcement learning

---

## Practical Applications

### Application 1: GPS Navigation
**Techniques Used**:
- A* search algorithm
- Graph representation of roads
- Heuristic: straight-line distance

### Application 2: Spam Filtering
**Techniques Used**:
- Bayesian classification
- Probabilistic reasoning
- Learning from labeled examples

### Application 3: Game AI
**Techniques Used**:
- Minimax algorithm with alpha-beta pruning
- Evaluation functions
- Search tree exploration

### Application 4: Medical Diagnosis Systems
**Techniques Used**:
- Expert systems
- Bayesian networks
- Forward/backward chaining

---

## Key Takeaways

✅ **Search algorithms** are fundamental to problem-solving (BFS, DFS, A*)

✅ **A* search** combines actual cost and heuristics for optimal pathfinding

✅ **Knowledge representation** methods include logic, semantic networks, frames

✅ **Expert systems** mimic human experts using knowledge bases and inference

✅ **AI agents** perceive environments and act to achieve goals

✅ **Uncertainty** is handled through probability, Bayesian reasoning, fuzzy logic

✅ **Planning** involves representing states, actions, and goals formally

✅ **Real-world AI** combines multiple techniques for robust solutions

---

## What's Next?

1. **Advanced AI Topics**: Explore [Advanced AI](03-advanced-ai-topics.md)
2. **Machine Learning**: Learn data-driven approaches in [ML Fundamentals](../Machine-Learning/01-beginner-ml-fundamentals.md)
3. **Specialized Domains**:
   - [NLP](../Natural-Language-Processing/01-beginner-nlp-basics.md) for language understanding
   - [Computer Vision](../Computer-Vision/01-beginner-cv-basics.md) for image analysis
   - [LLMs](../Large-Language-Models/01-beginner-llm-basics.md) for modern language AI

---

## Practice Exercises

1. **Implement A***: Write pseudocode for A* search for finding shortest path in a grid
2. **Design Expert System**: Create rules for a simple medical diagnosis system
3. **Agent Design**: Describe a utility-based agent for stock trading
4. **Bayesian Reasoning**: Calculate probability of disease given symptoms using Bayes' theorem
5. **MDP Formulation**: Formulate a simple game as an MDP

---

*Previous: [AI Basics](01-beginner-ai-basics.md) | Next: [Advanced AI Topics](03-advanced-ai-topics.md)*
