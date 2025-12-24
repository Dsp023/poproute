# Advanced Computer Vision Applications

## Table of Contents
1. [Vision Transformers](#vision-transformers)
2. [Advanced Object Detection](#advanced-object-detection)
3. [Video Understanding](#video-understanding)
4. [3D Computer Vision](#3d-computer-vision)
5. [Generative Models for Images](#generative-models-for-images)
6. [Multi-Modal Vision-Language Models](#multi-modal-vision-language-models)
7. [Self-Supervised Learning](#self-supervised-learning)
8. [Neural Architecture Search](#neural-architecture-search)
9. [Production CV Systems](#production-cv-systems)

---

## Vision Transformers

### ViT (Vision Transformer)

**Key Idea**: Apply transformers directly to image patches

**Architecture**:
1. **Patch Embedding**: Split image into patches (16×16)
   - 224×224 image → 14×14 = 196 patches
2. **Linear Projection**: Flatten patches to vectors
3. **Position Embeddings**: Add position information
4. **Transformer Encoder**: Standard transformer blocks
5. **Classification Head**: MLP on [CLS] token

**Formula**:
```
Image → Patches → Flatten → Linear Projection + Position Embedding
→ Transformer Encoder × L → MLP Head → Class
```

**Results**: Matches or beats CNNs when trained on large datasets

**Requirement**: Needs lots of data (ImageNet-21K or JFT-300M)

**Advantages**:
- Global receptive field from first layer
- Captures long-range dependencies
- Interpretable attention maps

**Disadvantages**:
- Data-hungry
- Computationally expensive
- Less inductive bias than CNNs

### Swin Transformer

**Hierarchical vision transformer**

**Key Innovation**: Shifted windows for efficiency

**Advantages over ViT**:
- O(n) complexity vs O(n²) (where n = image size)
- Multi-scale features (like CNNs)
- Better for dense prediction (detection, segmentation)

**Architecture**:
- Patch merging gradually reduces resolution
- Window-based self-attention
- Shifted windows for cross-window connections

**Performance**: State-of-the-art on many vision benchmarks

### DeiT (Data-Efficient ViT)

**Training vision transformers with less data**

**Techniques**:
- Knowledge distillation from CNN teacher
- Stronger data augmentation
- Regularization

**Result**: Competitive with ViT trained on less data

### Hybrid Models

**Combine CNNs and Transformers**

**Examples**:
- CNN backbone + Transformer head
- Early Convolutions + Transformer layers (ConViT)

**Benefits**: Best of both worlds

---

## Advanced Object Detection

### Transformer-Based Detection

#### DETR (Detection Transformer)

**End-to-end object detection** with transformers

**Architecture**:
1. CNN backbone extracts features
2. Transformer encoder-decoder
3. Set prediction (parallel detection)

**Key Innovation**: No hand-crafted components
- No NMS (non-max suppression)
- No anchor boxes
- No region proposals

**Loss**: Bipartite matching + Hungarian algorithm

**Advantages**:
- Simpler pipeline
- Direct set prediction
- Panoptic segmentation extension

**Challenges**:
- Slow convergence
- High memory usage

**Successors**: Deformable DETR, Efficient DETR (address issues)

### Anchor-Free Detection

**YOLOv8**, **FCOS**, **CenterNet**

**Idea**: Predict center point detection,bounding box parameters

**Advantages**:
- Fewer hyperparameters
- Better for arbitrary aspect ratios
- Simpler design

### Few-Shot Object Detection

**Detect new classes** with few examples (1-30 images)

**Approaches**:
- Meta-learning
- Transfer learning from base classes
- Metric learning

**Applications**: Rare object detection, custom domains

---

## Video Understanding

### Action Recognition

**Task**: Classify actions in videos (running, jumping, cooking)

**Challenges**:
- Temporal modeling (motion patterns)
- Computational cost (many frames)

**Approaches**:

#### 1. Two-Stream Networks
- Spatial stream: RGB frames (appearance)
- Temporal stream: Optical flow (motion)
- Fuse predictions

#### 2. 3D CNNs
- **C3D**, **I3D**: 3D convolutions over space and time
- Expensive but effective

#### 3. Two-Stream I3D (Inflated 3D)
- Inflate 2D filters to 3D
- Pre-train on ImageNet, fine-tune on videos

#### 4. Transformers for Video
- **TimeSformer**: Spatial + temporal attention
- **Video Vision Transformer** (ViViT)

**Datasets**: Kinetics, UCF-101, AVA

### Video Object Detection & Tracking

**Task**: Detect & track objects across frames

**Approaches**:
- Frame-by-frame detection + tracking algorithm
- Temporal feature aggregation
- Recurrent networks

**Popular**: **FairMOT**, **ByteTrack**

### Video Segmentation

**Task**: Segment objects in video (temporal consistency)

**Variants**:
- Video instance segmentation
- Video panoptic segmentation
- Video object segmentation (VOS)

---

## 3D Computer Vision

### Depth Estimation

**Task**: Predict depth (distance) for each pixel

**Approaches**:

#### Monocular Depth (Single Image)
- **MiDaS**: Generalizable depth estimation
- Transformer-based models
- Self-supervised learning from videos

#### Stereo Depth (Two Cameras)
- Disparity estimation
- Classical: SGM (Semi-Global Matching)
- Deep learning: PSMNet, RAFT-Stereo

**Applications**: AR/VR, robotics, autonomous driving

### 3D Object Detection

**Task**: Detect objects in 3D space (x, y, z, orientation, dimensions)

**Input Modalities**:

#### LiDAR Point Clouds
- **PointNet**, **PointNet++**: Direct point cloud processing
- **VoxelNet**: Voxelize point cloud
- **PointPillars**: Pillar-based for speed

#### RGB + Depth (RGB-D)
- Frustum-based methods
- Voting-based (VoteNet)

#### Camera-Only  
- **FCOS3D**, **DETR3D**: 3D from 2D images
- Monocular 3D detection

**Applications**: Autonomous vehicles, robotics

### 3D Reconstruction

**Task**: Reconstruct 3D models from images

**Methods**:
- **NeRF** (Neural Radiance Fields): Novel view synthesis
- **Structure from Motion** (SfM): Classical approach
- **Multi-View Stereo** (MVS): Dense reconstruction

**NeRF**:
- Represents scene as neural network
- Input: (x, y, z, viewing direction)
- Output: color, density
- Render novel views via ray tracing

**Extensions**: Instant-NGP (fast NeRF), Mip-NeRF

### Point Cloud Processing

**PointNet Family**:
- Direct processing of unordered points
- Permutation invariant
- Classification, segmentation, detection

---

## Generative Models for Images

### Generative Adversarial Networks (GANs)

**Architecture**: Generator vs Discriminator

**Evolution**:
- **DCGAN** (2015): Convolutional GANs
- **Progressive GAN** (2017): Grow resolution progressively
- **StyleGAN** (2018): Style-based generator
  - Control image style at multiple scales
  - Disentangled latent space
- **StyleGAN2, StyleGAN3**: Improvements

**Applications**:
- Image generation (faces, art)
- Image-to-image translation (Pix2Pix, CycleGAN)
- Super-resolution (ESRGAN)
- Editing (StyleCLIP)

**Challenges**:
- Training instability
- Mode collapse
- Evaluation (FID, IS scores)

### Diffusion Models

**Key Idea**: Learn to denoise noisy images iteratively

**Process**:
1. **Forward (Training)**: Gradually add noise to image
2. **Reverse (Sampling)**: Learn to remove noise step-by-step

**Models**:
- **DDPM** (Denoising Diffusion Probabilistic Models)
- **DALL-E 2**: Text-to-image with diffusion
- **Stable Diffusion**: Open-source text-to-image
- **Imagen** (Google): Text-to-image

**Advantages over GANs**:
- More stable training
- Better sample quality
- Mode coverage

**Applications**:
- Text-to-image generation
- Image editing (inpainting, outpainting)
- Super-resolution

### Variational Autoencoders (VAEs)

**Probabilistic generative model**

**Architecture**: Encoder → Latent space → Decoder

**Loss**: Reconstruction + KL divergence

**Applications**:
- Image generation (less sharp than GANs/Diffusion)
- Representation learning
- Anomaly detection

---

## Multi-Modal Vision-Language Models

### CLIP (Contrastive Language-Image Pre-training)

**Key Idea**: Joint embedding of images and text

**Training**:
- 400M (image, caption) pairs
- Contrastive loss (match correct pairs)

**Capabilities**:
- Zero-shot image classification
- Text-based image retrieval
- Image-based text retrieval

**Impact**: Enables many downstream applications

### Vision-Language Tasks

#### Image Captioning
- **BLIP**, **BLIP-2**: Bootstrapped language-image pre-training
- **Flamingo**: Few-shot learning

#### Visual Question Answering (VQA)
- Answer questions about images
- **LXMERT**, **ViLBERT**: Vision-language transformers

#### Text-to-Image Generation
- **DALL-E 2**, **Stable Diffusion**, **Imagen**, **Midjourney**

#### Image-Text Matching
- Determine if caption matches image

### GPT-4V (Vision)

**Multimodal GPT-4**: Processes images + text

**Capabilities**:
- Image understanding
- Visual reasoning
- OCR and document analysis
- Chart/diagram interpretation

**Impact**: General-purpose vision-language AI

### Flamingo, LLaVA

**Visual instruction following**

**Architecture**: Vision encoder + LLM

**Training**: Instruction tuning on vision-language tasks

---

## Self-Supervised Learning

**Learn representations without labels**

### Contrastive Learning

**SimCLR**:
- Augment same image twice → positive pair
- Different images → negative pairs
- Maximize similarity of positive, minimize negative

**MoCo** (Momentum Contrast):
- Queue of negative examples
- Momentum-updated encoder

**Benefits**: Learns generalizable features

### Masked Image Modeling

**MAE** (Mask Auto Encoder):
- Mask random patches (75%)
- Reconstruct masked patches
- Like BERT for images

**Results**: Strong pre-training, transfers well

### DINO

**Self-distillation with no labels**

**Student-teacher framework** with vision transformers

**Discovers semantic segmentation** without labels

---

## Neural Architecture Search (NAS)

**Automatically design neural networks**

### Methods

**1. Reinforcement Learning**:
- Controller generates architectures
- Train and evaluate
- Reward based on performance

**2. Evolutionary Algorithms**:
- Mutate and evolve architectures
- Select best performers

**3. Gradient-Based** (DARTS):
- Continuous relaxation of search space
- Differentiable architecture search

**4. Efficient NAS**:
- Once-for-all (OFA) networks
- Weight sharing during search

### Results

**EfficientNet**: NAS + compound scaling

**Best accuracy-efficiency trade-offs**

**Challenge**: Computational cost (thousands of GPU-hours)

---

## Production CV Systems

### Deployment Considerations

**Model Optimization**:
- **Quantization**: FP32 → INT8 (4x smaller, faster)
- **Pruning**: Remove less important weights
- **Knowledge Distillation**: Train smaller model from larger
- **Compiler Optimization**: TensorRT, ONNX Runtime

**Hardware Targets**:
- **Cloud**: GPUs (V100, A100, H100)
- **Edge**: Jetson, Coral, mobile GPUs
- **Mobile**: ARM CPUs, mobile GPU, NPU

**Frameworks**:
- **TensorFlow Lite**: Mobile/edge
- **PyTorch Mobile**: iOS/Android
- **ONNX**: Cross-platform interchange
- **TensorRT**: NVIDIA GPU optimization
- **OpenVINO**: Intel optimization

### Real-Time Systems

**Latency Requirements**:
- Real-time video: <33ms (30 FPS)
- Interactive: <100ms
- Batch processing: seconds-minutes OK

**Optimization Strategies**:
- Model selection (MobileNet vs ResNet)
- Resolution reduction
- Frame skipping
- Asynchronous processing

### Monitoring and Maintenance

**Track**:
- Inference latency
- Throughput (images/sec)
- Model accuracy on production data
- Data distribution shifts

**Retraining Pipeline**:
- Collect new data
- Automatic labeling (semi-supervised)
- Periodic retraining
- A/B testing

### Best Practices

✅ **Start with pre-trained models**  
✅ **Profile and optimize** bottlenecks  
✅ **Test on target hardware** early  
✅ **Monitor production performance**  
✅ **Plan for model updates**  
✅ **Handle edge cases gracefully**

---

## Key Takeaways

✅ **Vision Transformers**: ViT applies transformers to image patches, Swin Transformer for efficiency

✅ **Advanced detection**: DETR (transformer-based), anchor-free methods, few-shot detection

✅ **Video**: Action recognition (3D CNNs, transformers), tracking, video segmentation

✅ **3D Vision**: Depth estimation, 3D detection (LiDAR/camera), NeRF for reconstruction

✅ **Generative**: GANs (StyleGAN), Diffusion models (Stable Diffusion), VAEs

✅ **Multi-modal**: CLIP (vision-language), DALL-E (text-to-image), GPT-4V

✅ **Self-supervised**: SimCLR, MAE, DINO learn without labels

✅ **NAS**: Automatically find optimal architectures (EfficientNet)

✅ **Production**: Quantization, deployment frameworks, real-time optimization

---

## Further Reading

**Papers**:
- "An Image is Worth 16x16 Words" (ViT)
- "End-to-End Object Detection with Transformers" (DETR)
- "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
- "Masked Autoencoders Are Scalable Vision Learners" (MAE)
- "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion)

**Resources**:
- Papers With Code (Vision leaderboards)
- Hugging Face Transformers (vision models)
- PyTorch Vision tutorials
- NVIDIA Developer Blog

---

*Previous: [Intermediate CV Techniques](02-intermediate-cv-techniques.md)*
