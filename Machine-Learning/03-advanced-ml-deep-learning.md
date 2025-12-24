# Advanced ML - Deep Learning

## Table of Contents
1. [Introduction to Deep Learning](#introduction-to-deep-learning)
2. [Neural Network Fundamentals](#neural-network-fundamentals)
3. [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
4. [Recurrent Neural Networks (RNNs)](#recurrent-neural-networks-rnns)
5. [Advanced Architectures](#advanced-architectures)
6. [Optimization Techniques](#optimization-techniques)
7. [Transfer Learning](#transfer-learning)
8. [Deep Learning Frameworks](#deep-learning-frameworks)

---

## Introduction to Deep Learning

**Deep Learning**: Subset of ML using neural networks with multiple layers.

**Why "Deep"?**: Multiple hidden layers (vs shallow networks with 1-2 layers)

**Key Breakthroughs**:
- **2006**: Geoffrey Hinton's pre-training technique
- **2012**: AlexNet wins ImageNet (deep CNN for images)
- **2017**: Transformer architecture (Attention Is All You Need)
- **2018+**: BERT, GPT revolutionize NLP

**Applications**:
- Computer Vision (image classification, object detection)
- Natural Language Processing (translation, text generation)
- Speech Recognition
- Game Playing (AlphaGo, OpenAI Five)
- Generative AI (DALL-E, Midjourney, ChatGPT)

---

## Neural Network Fundamentals

### The Perceptron (Single Neuron)

**Components**:
1. **Inputs** (x₁, x₂, ..., xₙ)
2. **Weights** (w₁, w₂, ..., wₙ)
3. **Bias** (b)
4. **Activation Function** (f)

**Computation**:
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
output = f(z)
```

### Activation Functions

#### 1. Sigmoid
```
σ(z) = 1 / (1 + e^(-z))
```
**Range**: (0, 1)  
**Use**: Output layer for binary classification  
**Problem**: Vanishing gradients

#### 2. Tanh (Hyperbolic Tangent)
```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```
**Range**: (-1, 1)  
**Better than sigmoid**: Zero-centered

#### 3. ReLU (Rectified Linear Unit) ⭐
```
ReLU(z) = max(0, z)
```
**Most popular for hidden layers**

**Pros**:
- Fast computation
- Mitigates vanishing gradient
- Sparse activation

**Cons**: Dying ReLU (neurons output 0 forever)

#### 4. Leaky ReLU
```
LeakyReLU(z) = max(0.01z, z)
```
**Fix for dying ReLU**: Allows small negative values

#### 5. Softmax
```
softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
```
**Use**: Multi-class classification output  
**Output**: Probabilities summing to 1

### Multi-Layer Perceptron (MLP)

**Architecture**:
```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Output Layer
```

**Forward Propagation**: Compute outputs from inputs

**Backpropagation**: Update weights using gradient descent
1. Calculate loss (how wrong predictions are)
2. Compute gradients (derivatives of loss w.r.t. weights)
3. Update weights: `w = w - learning_rate * gradient`

### Loss Functions

**Regression**:
- **Mean Squared Error (MSE)**: `L = (1/n) Σ (y - ŷ)²`

**Binary Classification**:
- **Binary Cross-Entropy**: `L = -[y log(ŷ) + (1-y) log(1-ŷ)]`

**Multi-Class Classification**:
- **Categorical Cross-Entropy**: `L = -Σ yᵢ log(ŷᵢ)`

---

## Convolutional Neural Networks (CNNs)

**Purpose**: Designed for image/spatial data.

**Key Insight**: Local pixel patterns matter (edges, textures, objects).

### CNN Layers

#### 1. Convolutional Layer

**Operation**: Slide filter/kernel over image, compute dot product.

**Filter**: Small matrix (e.g., 3×3) of learnable weights

**Example - Edge Detection Filter**:
```
[-1 -1 -1]
[ 0  0  0]
[ 1  1  1]
```

**Hyperparameters**:
- **Number of filters**: How many features to detect
- **Filter size**: Usually 3×3 or 5×5
- **Stride**: Step size (1 = every pixel, 2 = every other pixel)
- **Padding**: Add zeros around border to maintain size

**Output**: Feature maps (one per filter)

#### 2. Pooling Layer

**Purpose**: Downsample feature maps (reduce size, increase robustness).

**Max Pooling** (most common):
- Take maximum value in each region (e.g., 2×2)
- Reduces size by half

**Average Pooling**:
- Take average instead of max

**Effect**: Translation invariance (object slightly shifted → same detection)

#### 3. Fully Connected Layer

**Standard neural network layer at the end**

**Purpose**: Combine features for final classification

### Canonical CNN Architecture

```
Input Image (e.g., 224×224×3)
    ↓
[Conv → ReLU → Pool] × N
    ↓
Flatten
    ↓
[Fully Connected → ReLU] × M
    ↓
Output (Softmax)
```

### Famous CNN Architectures

#### LeNet-5 (1998)
- First successful CNN
- Handwritten digit recognition

#### AlexNet (2012)
- Breakthrough on ImageNet
- 8 layers, ReLU, Dropout
- GPU training

#### VGG (2014)
- Very deep (16-19 layers)
- 3×3 filters only
- Simple, uniform architecture

#### ResNet (2015)
- **Residual connections** / **Skip connections**
- Solves vanishing gradient in very deep networks
- 50, 101, 152 layer variants

**Key Innovation**:
```
x → [Conv → ReLU → Conv] → + → ReLU
↓___________________________|
      (skip connection)
```

#### EfficientNet (2019)
- Compound scaling (depth, width, resolution)
- State-of-the-art accuracy with fewer parameters

---

## Recurrent Neural Networks (RNNs)

**Purpose**: Sequential data (text, time series, audio).

**Key Idea**: Maintain hidden state updated at each time step.

### Vanilla RNN

**At each time step t**:
```
h_t = tanh(W_hh * h_(t-1) + W_xh * x_t + b)
y_t = W_hy * h_t + b
```

**h_t**: Hidden state (memory)  
**x_t**: Input at time t  
**y_t**: Output at time t

**Problem**: Vanishing/exploding gradients over long sequences

### Long Short-Term Memory (LSTM)

**Solution to vanishing gradient**: Gates control information flow.

**Gates**:
1. **Forget Gate**: What to forget from cell state
2. **Input Gate**: What new information to store
3. **Output Gate**: What to output

**Cell State**: Long-term memory running through sequence

**Applications**:
- Language modeling
- Machine translation
- Speech recognition
- Time series prediction

### Gated Recurrent Unit (GRU)

**Simplified LSTM**: Fewer gates, faster training.

**Often comparable performance to LSTM**.

### Bidirectional RNN

**Process sequence in both directions**:
- Forward: left-to-right
- Backward: right-to-left

**Benefit**: Context from both past and future

**Use**: When full sequence available (not real-time)

---

## Advanced Architectures

### Autoencoders

**Purpose**: Unsupervised learning of efficient data representations.

**Architecture**:
```
Input → Encoder → Latent Code (Bottleneck) → Decoder → Reconstruction
```

**Training**: Minimize reconstruction error

**Applications**:
- Dimensionality reduction
- Denoising
- Anomaly detection
- Generative modeling (VAE)

### Variational Autoencoders (VAE)

**Generative model**: Learn probability distribution of data.

**Difference from AE**: Encoder outputs mean and variance (not single code).

**Use**: Generate new data samples

### Generative Adversarial Networks (GANs)

**Two networks competing**:
1. **Generator**: Creates fake data
2. **Discriminator**: Distinguishes real from fake

**Training**: Adversarial game
- Generator tries to fool discriminator
- Discriminator tries to detect fakes
- Both improve together

**Applications**:
- Image generation (faces, art)
- Image-to-image translation (StyleGAN)
- Data augmentation
- Super-resolution

**Challenges**: Training instability, mode collapse

**Variants**: DCGAN, StyleGAN, CycleGAN, Pix2Pix

### Attention Mechanisms

**Key Idea**: Focus on relevant parts of input.

**Seq2Seq with Attention**:
- Encoder processes input sequence
- Decoder generates output, attending to relevant encoder states

**Example - Translation**:
When translating "the cat sat on the mat" to French,
"le chat" should attend to "the cat".

**Self-Attention**: Attention within same sequence (foundation of Transformers)

---

## Optimization Techniques

### Gradient Descent Variants

#### 1. Stochastic Gradient Descent (SGD)

Update weights using one random sample at a time.

**Pros**: Fast updates, memory efficient  
**Cons**: Noisy updates, slower convergence

#### 2. Mini-Batch Gradient Descent

Update using small batch of samples (e.g., 32, 64, 128).

**Standard in deep learning**: Balance of speed and stability

#### 3. Momentum

**Idea**: Accumulate gradients like a ball rolling downhill.

**Effect**: Faster convergence, escape local minima

#### 4. Adam (Adaptive Moment Estimation) ⭐

**Most popular optimizer**.

**Combines**:
- Momentum (first moment)
- Adaptive learning rates (second moment)

**Hyperparameters**: Learning rate (default 0.001), β₁=0.9, β₂=0.999

### Learning Rate Scheduling

**Problem**: Fixed learning rate suboptimal.

**Strategies**:
1. **Step Decay**: Reduce by factor every N epochs
2. **Exponential Decay**: Continuous exponential decrease 
3. **Cosine Annealing**: Cosine curve decrease
4. **Learning Rate Warmup**: Start small, increase, then decrease
5. **ReduceLROnPlateau**: Reduce when validation loss plateaus

### Regularization

#### 1. L2 Regularization (Weight Decay)

Add penalty: `Loss = Loss_data + λ * Σw²`

**Effect**: Prevents large weights

#### 2. Dropout

**Random drop neurons during training** (e.g., 50% probability).

**Effect**: Prevents co-adaptation, acts like ensemble

**At inference**: Use all neurons (scaled by dropout probability)

#### 3. Batch Normalization

**Normalize activations** within each mini-batch.

**Benefits**:
- Faster training
- Higher learning rates possible
- Regularization effect
- Reduces sensitivity to initialization

**Where**: Usually after conv/FC layer, before activation

#### 4. Data Augmentation

**Create variations of training data**.

**Image Augmentation**:
- Random crops, flips, rotations
- Color jittering
- Mixup, CutMix

**Text Augmentation**:
- Synonym replacement
- Back-translation
- Random insertion/deletion

### Early Stopping

**Stop training when validation loss stops improving**.

**Prevents overfitting** while finding best model.

---

## Transfer Learning

**Idea**: Use knowledge from one task to improve learning on another.

**Why Effective**: Low-level features (edges, textures) are universal.

### Approaches

#### 1. Feature Extraction

**Process**:
1. Take pre-trained model (e.g., ResNet on ImageNet)
2. Remove final layer
3. Freeze all weights (don't train)
4. Add new classifier for your task
5. Train only new classifier

**Use When**: Small dataset, similar task

#### 2. Fine-Tuning

**Process**:
1. Take pre-trained model
2. Replace final layer
3. Train entire network on your data (with small learning rate)

**Variants**:
- Fine-tune all layers
- Fine-tune only top layers (freeze bottom)
- Gradual unfreezing (start top, progressively unfreeze lower)

**Use When**: Moderate dataset

### Pre-Trained Models

**Computer Vision**:
- ResNet, VGG, EfficientNet (ImageNet)
- CLIP (vision-language)

**NLP**:
- BERT, RoBERaTA (text understanding)
- GPT (text generation)
- T5 (text-to-text)

**Multimodal**:
- CLIP (image-text)
- Flamingo (vision-language)

**Benefit**: Start from strong baseline, require less data.

---

## Deep Learning Frameworks

### TensorFlow / Keras

**TensorFlow**: Google's DL framework  
**Keras**: High-level API (now part of TensorFlow)

**Pros**:
- Production-ready (TensorFlow Serving, TF Lite)
- TensorBoard visualization
- Large ecosystem

**Use**: Production deployment, mobile/edge

### PyTorch

**Facebook's DL framework**.

**Pros**:
- Pythonic, intuitive
- Dynamic computation graphs
- Popular in research

**Use**: Research, prototyping

### JAX

**Google's numerical computing library**.

**Features**:
- Auto-differentiation
- JIT compilation
- GPU/TPU acceleration

**Use**: High-performance computing, research

### Comparison

| Feature | TensorFlow/Keras | PyTorch |
|---------|------------------|---------|
| Ease of Use | High (Keras) | High |
| Flexibility | Medium | High |
| Debugging | Medium | Easy |
| Production | Excellent | Good |
| Research | Good | Excellent |

---

## Key Takeaways

✅ **Deep Learning** uses multi-layer neural networks for complex pattern recognition

✅ **CNNs** excel at image/spatial data using convolution and pooling

✅ **RNNs/LSTMs** handle sequential data like text and time series

✅ **Advanced architectures** (Autoencoders, GANs, Attention) enable generative modeling

✅ **Optimization** (Adam, learning rate scheduling) crucial for training

✅ **Regularization** (dropout, batch norm, data augmentation) prevents overfitting

✅ **Transfer learning** leverages pre-trained models for faster, better results

✅ **Frameworks** (TensorFlow, PyTorch) provide tools for efficient development

---

## What's Next?

- **Modern Architectures**: [Transformers](../Transformers/01-beginner-transformer-basics.md)
- **Language Models**: [LLMs](../Large-Language-Models/01-beginner-llm-basics.md)
- **Applications**: [NLP](../Natural-Language-Processing/01-beginner-nlp-basics.md), [Computer Vision](../Computer-Vision/01-beginner-cv-basics.md)
- **Deployment**: [MLOps](../MLOps-Deployment/01-beginner-mlops-basics.md)

---

*Previous: [Intermediate ML](02-intermediate-ml-algorithms.md)*
