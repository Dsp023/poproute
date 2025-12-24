# Intermediate Computer Vision Techniques

## Table of Contents
1. [Convolutional Neural Networks Deep Dive](#convolutional-neural-networks-deep-dive)
2. [Image Classification Architectures](#image-classification-architectures)
3. [Object Detection](#object-detection)
4. [Image Segmentation](#image-segmentation)
5. [Data Augmentation](#data-augmentation)
6. [Transfer Learning for Vision](#transfer-learning-for-vision)
7. [Popular CV Frameworks](#popular-cv-frameworks)
8. [Practical Implementation](#practical-implementation)

---

## Convolutional Neural Networks Deep Dive

### CNN Architecture Components

#### 1. Convolutional Layers

**Operation**: Slide filters over image, compute dot products

**Filter/Kernel**: Small matrix of learnable weights (typically 3×3, 5×5)

**Convolution Formula**:
```
Output[i,j] = Σ Input[i+m, j+n] × Filter[m,n] + bias
```

**Hyperparameters**:
- **Filter size**: 3×3 (modern standard), 5×5, 7×7
- **Number of filters**: 32, 64, 128, 256... (increases with depth)
- **Stride**: Step size (1 = every pixel, 2 = every other pixel)
- **Padding**: 'same' (preserve size) or 'valid' (reduce size)

**Example - Edge Detection Filter**:
```
Vertical edge:        Horizontal edge:
[-1  0  1]           [-1 -1 -1]
[-1  0  1]           [ 0  0  0]
[-1  0  1]           [ 1  1  1]
```

**Feature Maps**: Each filter produces one feature map (activation map)

#### 2. Pooling Layers

**Purpose**: Downsample feature maps (reduce dimensions, parameters)

**Max Pooling** (most common):
- Take maximum value in each region
- Example: 2×2 max pooling reduces size by half

**Average Pooling**:
- Take average instead of max

**Benefits**:
- Translation invariance (small shifts don't affect output much)
- Reduces overfitting
- Computational efficiency

**Modern trend**: Some architectures skip pooling (use stride convolutions)

#### 3. Activation Functions

**ReLU** (Rectified Linear Unit) - Standard:
```
f(x) = max(0, x)
```

**Variants**:
- **Leaky ReLU**: f(x) = max(0.01x, x) (fixes dying ReLU)
- **ELU** (Exponential Linear Unit): Smooth for negative values
- **Swish/SiLU**: f(x) = x × sigmoid(x) (better for deep networks)

#### 4. Batch Normalization

**Normalizes activations** within mini-batch

**Formula**:
```
BN(x) = γ × (x - μ_batch) / σ_batch + β
```

**Benefits**:
- Faster training (higher learning rates)
- Reduces internal covariate shift
- Regularization effect

**Placement**: Usually after convolution, before activation

#### 5. Dropout

**Randomly drop neurons** during training

**Typical rate**: 0.5 (drop 50%)

**Prevents overfitting** in fully connected layers

**Not used during inference**

---

## Image Classification Architectures

### AlexNet (2012) - The Breakthrough

**Architecture**:
- 5 convolutional layers
- 3 fully connected layers
- ReLU activations
- Dropout, data augmentation

**Innovation**: Deep CNN on ImageNet, GPU training

**Result**: 16.4% error (vs 26% previous best)

### VGG (2014) - Uniform Design

**Key Ideas**:
- Small 3×3 filters throughout
- Deep (16-19 layers)
- Simple, uniform architecture

**VGG-16 Structure**:
```
[Conv3×3, Conv3×3, MaxPool] × 2 (64 filters)
[Conv3×3, Conv3×3, MaxPool] × 2 (128 filters)
[Conv3×3, Conv3×3, Conv3×3, MaxPool] × 2 (256 filters)
[Conv3×3, Conv3×3, Conv3×3, MaxPool] × 2 (512 filters)
Fully Connected × 3
```

**Impact**: Showed depth matters

**Drawback**: 138M parameters (large)

### ResNet (2015) - Skip Connections

**Problem**: Very deep networks degrade (training error increases!)

**Solution**: Residual connections (skip connections)

**Residual Block**:
```
Input → [Conv → BN → ReLU → Conv → BN] → Add → ReLU
  ↓____________________________________________↑
              (skip connection)
```

**Key Insight**: Easier to learn residual F(x) than full mapping H(x)
```
H(x) = F(x) + x
```

**Variants**: ResNet-50, ResNet-101, ResNet-152 (depth = number of layers)

**Impact**: Enabled 100+ layer networks

**Still widely used**: Pre-trained ResNet-50 is industry standard

### Inception/GoogLeNet (2014) - Multi-Scale

**Inception Module**: Multiple filter sizes in parallel

**Structure**:
```
Input → [1×1 Conv] → Concatenate
     → [1×1 → 3×3 Conv] → 
     → [1×1 → 5×5 Conv] → 
     → [MaxPool → 1×1] →
```

**1×1 convolutions**: Reduce dimensions (computational efficiency)

**Idea**: Let network learn which filter size is best

### MobileNet (2017) - Efficient Mobile

**For mobile/edge devices**: Small, fast

**Depthwise Separable Convolutions**:
1. Depthwise: Separate filter per channel
2. Pointwise: 1×1 convolution to combine

**Benefits**: 8-10× fewer parameters than standard convolution

**Variants**: MobileNetV2, MobileNetV3

### EfficientNet (2019) - Compound Scaling

**Systematically scale**: Depth, width, resolution together

**Best accuracy-efficiency trade-off** as of creation

**EfficientNet-B0 to B7**: Different scale factors

**Method**: Neural architecture search + scaling

---

## Object Detection

### Problem Definition

**Task**: Localize and classify objects in image

**Output**: 
- Bounding boxes (x, y, width, height)
- Class labels
- Confidence scores

**Metrics**:
- **IoU** (Intersection over Union): Overlap between predicted and ground truth boxes
- **mAP** (mean Average Precision): Detection quality across classes

### Two-Stage Detectors

#### R-CNN Family

**R-CNN (2014)**:
1. Propose ~2000 regions (selective search)
2. Warp each to fixed size
3. CNN feature extraction per region
4. SVM classification

**Fast R-CNN (2015)**:
- CNN on full image (not per region)
- ROI pooling
- Single network for classification + bbox regression

**Faster R-CNN (2016)**:
- Replace selective search with Region Proposal Network (RPN)
- End-to-end trainable
- ~7 FPS

**Mask R-CNN (2017)**:
- Adds segmentation mask prediction
- Instance segmentation

### One-Stage Detectors (Faster)

#### YOLO (You Only Look Once)

**Key Idea**: Single pass through network

**Process**:
1. Divide image into grid (e.g., 7×7)
2. Each cell predicts bounding boxes + classes
3. Non-max suppression to remove duplicates

**Versions**:
- YOLOv1 (2016): Original, fast but less accurate
- YOLOv3 (2018): Multi-scale predictions, better accuracy
- YOLOv5, YOLOv7, YOLOv8 (2020-2023): State-of-the-art speed/accuracy

**Speed**: Real-time (30-100+ FPS)

**Trade-off**: Slightly less accurate than two-stage on small objects

#### SSD (Single Shot Detector)

**Multi-scale feature maps** for detection

**Predictions at multiple layers** (different resolutions)

**Good balance** of speed and accuracy

### Modern Approaches

**DETR** (2020): Transformer-based detection
- Treats detection as set prediction
- No hand-designed components (NMS, anchors)

**EfficientDet**: Efficient object detection
- Uses EfficientNet backbone
- BiFPN (feature pyramid)

---

## Image Segmentation

### Semantic Segmentation

**Task**: Label every pixel with class

**Not distinguishing instances** (all cars = "car", not car 1, car 2)

#### FCN (Fully Convolutional Network)

**Replace FC layers with convolutions**

**Upsampling** to restore resolution:
- Transposed convolution (deconvolution)
- Bilinear interpolation

**Skip connections** from encoder to decoder

#### U-Net (2015)

**Encoder-decoder with skip connections**

**Architecture**:
```
     Input
       ↓
  [Encoder] ──→ [Decoder]
       ↓    ↖  ↙     ↓
  [Encoder] ──→ [Decoder]
       ↓    ↖  ↙     ↓
    [Bottom]
```

**Skip connections**: Preserve fine-grained details

**Originally for medical imaging**, now widely used

#### DeepLab

**Atrous/Dilated convolutions**: Expand receptive field without increasing parameters

**Atrous Spatial Pyramid Pooling (ASPP)**: Multi-scale context

**State-of-the-art** semantic segmentation

### Instance Segmentation

**Task**: Separate individual objects

**Example**: Car 1, Car 2, Car 3 (each separate)

**Mask R-CNN**: Most popular
- Extends Faster R-CNN
- Adds mask prediction branch

---

## Data Augmentation

**Artificially expand training data** by transformations

### Geometric Transformations

**Random Crop**: 
- Extract random patches
- Forces model to learn from parts

**Random Flip**:
- Horizontal (almost always used)
- Vertical (task-dependent)

**Random Rotation**:
- Small angles (±15°) typical
- Large angles for some tasks

**Random Scale/Zoom**:
- Resize image by random factor

**Random Translation**:
- Shift image horizontally/vertically

### Photometric Transformations

**Brightness**: Adjust overall lightness

**Contrast**: Adjust difference between light/dark

**Saturation**: Adjust color intensity

**Hue**: Shift color spectrum

**Gaussian Noise**: Add random noise

**Gaussian Blur**: Blur image

### Advanced Techniques

**Cutout** (2017):
- Randomly mask out square regions
- Forces model to use all features

**Mixup** (2018):
- Blend two images and labels
```
x_mixed = λ × x1 + (1-λ) × x2
y_mixed = λ × y1 + (1-λ) × y2
```

**CutMix** (2019):
- Cut and paste patches between images
- Mix labels proportionally

**AutoAugment**:
- Learn augmentation policy via RL
- Task-specific optimal augmentations

### Albumentations Library

**Popular Python library** for augmentation

**Features**:
- Fast (C++ backend)
- Many transformations
- Easy composition

**Example**:
```python
import albumentations as A

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
```

---

## Transfer Learning for Vision

### Why Transfer Learning?

**Problem**: Small dataset, training from scratch overfits

**Solution**: Use pre-trained network (usually ImageNet)

**Intuition**: Low-level features (edges, textures) are universal

### Approach 1: Feature Extraction

**Process**:
1. Load pre-trained model (e.g., ResNet-50 on ImageNet)
2. Remove final classification layer
3. Freeze all weights (no training)
4. Add new classifier for your task
5. Train only new classifier

**When to use**:
- Very small dataset (< 1000 images)
- Similar to ImageNet (natural images)

**Fast and simple**

### Approach 2: Fine-Tuning

**Process**:
1. Load pre-trained model
2. Replace final layer
3. Train entire network (or top layers) with small learning rate

**When to use**:
- Moderate dataset (1000-100K images)
- Somewhat different from ImageNet

**Strategies**:
- Fine-tune all layers (larger dataset)
- Fine-tune top N layers only (smaller dataset)
- Gradual unfreezing (start top, progressively unfreeze lower)

**Learning rate**: 10-100× smaller than training from scratch

### Pre-trained Model Zoo

**TorchVision** (PyTorch):
- ResNet, VGG, EfficientNet, MobileNet
- Pre-trained on ImageNet

**Keras Applications** (TensorFlow):
- ResNet50, VGG16, InceptionV3, EfficientNet
- Pre-trained weights included

**Hugging Face Transformers**:
- Vision Transformers (ViT)
- CLIP, DINO

### Domain Adaptation

**When source and target domains differ**

**Techniques**:
- Fine-tune with target domain data
- Multi-task learning
- Adversarial adaptation

---

## Popular CV Frameworks

### OpenCV

**Classical computer vision** library

**Features**:
- Image I/O, processing
- Classical algorithms (edge detection, feature matching)
- Real-time computer vision

**Language**: C++ with Python bindings

**Use cases**: Pre-processing, classical CV, production systems

### PyTorch + torchvision

**Deep learning** framework

**torchvision**:
- Pre-trained models
- Datasets (CIFAR, ImageNet, COCO)
- Transforms
- Utilities

**Strengths**:
- Research-friendly
- Dynamic computation
- Excellent documentation

### TensorFlow + Keras

**Deep learning framework**

**Keras**: High-level API

**TF ecosystem**:
- TensorBoard visualization
- TF Serving (production)
- TF Lite (mobile/edge)

**Strengths**:
- Production-ready
- Comprehensive ecosystem
- Good for deployment

### Detectron2

**Facebook AI Research**

**State-of-the-art** detection and segmentation

**Features**:
- Mask R-CNN, Faster R-CNN
- Panoptic segmentation
- Pre-trained models

**Built on PyTorch**

### MMDetection

**OpenMMLab** comprehensive detection toolbox

**Features**:
- 50+ detection algorithms
- Modular design
- Extensive pre-trained models

---

## Practical Implementation

### Best Practices

**1. Data Preparation**:
- Normalize images (mean/std from ImageNet)
- Consistent preprocessing
- Proper train/val/test split

**2. Start Simple**:
- Begin with pre-trained model
- Baseline before complex models
- Understand data first

**3. Monitor Training**:
- Plot train/val loss curves
- Watch for overfitting
- Use TensorBoard or Weights & Biases

**4. Hyperparameter Tuning**:
- Learning rate most important
- Batch size (as large as GPU allows)
- Optimizer (Adam, SGD with momentum)

**5. Debugging**:
- Overfit single batch (sanity check)
- Visualize predictions
- Check data augmentation

### Common Pitfalls

❌ **Not using pre-trained weights**
✅ Use ImageNet pre-trained models

❌ **Too high learning rate with fine-tuning**
✅ Use 10-100× smaller LR

❌ **Forgetting to normalize images**
✅ Apply same normalization as pre-training

❌ **Insufficient data augmentation**
✅ Use geometric + photometric augmentations

❌ **Class imbalance ignored**
✅ Use weighted loss or sampling

---

## Key Takeaways

✅ **CNNs**: Convolution, pooling, activation, batch norm - building blocks of vision models

✅ **Architectures**: VGG (simple), ResNet (skip connections), EfficientNet (optimized)

✅ **Object detection**: Two-stage (Faster R-CNN) vs one-stage (YOLO) trade speed vs accuracy

✅ **Segmentation**: Semantic (pixel labels) vs instance (separate objects)

✅ **Data augmentation**: Essential for small datasets - geometric + photometric transforms

✅ **Transfer learning**: Feature extraction or fine-tuning from ImageNet pre-trained models

✅ **Frameworks**: PyTorch/torchvision for research, TensorFlow for production

✅ **Best practices**: Start simple, use pre-trained models, monitor training, augment data

---

## What's Next?

- **Advanced CV**: [Advanced CV Applications](03-advanced-cv-applications.md)
- **Transformers for Vision**: [Transformer Basics](../Transformers/01-beginner-transformer-basics.md)
- **Practical Projects**: Build image classifier, object detector, or segmentation model

---

*Previous: [CV Basics](01-beginner-cv-basics.md) | Next: [Advanced CV Applications](03-advanced-cv-applications.md)*
