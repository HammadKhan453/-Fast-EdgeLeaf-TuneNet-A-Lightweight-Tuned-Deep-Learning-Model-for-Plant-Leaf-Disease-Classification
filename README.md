🍃 Fast-EdgeLeaf TuneNet: A Lightweight & Tuned Deep Learning Model for Plant Leaf Disease Classification

Fast-EdgeLeaf TuneNet is a research-grade, computationally efficient CNN model designed for plant disease classification on leaf images.
The model is built for:

High speed

Low memory usage

Edge/IoT deployment

High accuracy on agricultural datasets

Tunability for various plant species and disease types

This repository contains the full training pipeline, dataset processing, model architecture, tuning components, and evaluation tools.

🌍 Problem Overview

Plant diseases significantly reduce crop productivity. Automated leaf-disease detection using CNNs is a popular research direction, but existing models have critical limitations:

❌ Limitations of Previous Approaches
1. Classical ML Methods

Examples: SVM, Random Forest, KNN
Problems:

Require manual feature extraction

Weak performance on raw images

Fail under varying lighting and background noise

2. Traditional Custom CNNs

Most past research uses shallow CNNs (3–4 conv layers).
Problems:

Fail to capture multiscale textures

Easily overfit small agricultural datasets

Limited generalization

3. Heavy Transfer Learning Models

VGG16, ResNet50, EfficientNetB4
Problems:

Too heavy for mobile/edge deployment

Slow training

Require large memory

Domain gap (trained on ImageNet, not leaf images)

🚀 Why Fast-EdgeLeaf TuneNet? (Your Model)

This model is created specifically to solve the problems above:

✨ Lightweight — suitable for real-time mobile/IoT
✨ Fast training
✨ Efficient feature extraction
✨ Tunable blocks for research and experimentation
✨ High accuracy even with limited datasets

Designed for datasets such as Tomato Disease, PlantVillage, and custom agricultural datasets.

🏗️ Fast-EdgeLeaf TuneNet Architecture (Reconstructed from Notebook)

The notebook constructs a custom lightweight CNN fused with tuned MobileNet components, combining:

EfficientNet/MobileNet-like depthwise separable convolutions

Custom convolution blocks for texture extraction

Feature scaling layers

Dense classifier head optimized for multi-class disease detection

Below is the architecture breakdown:

🔹 1. Input Block
Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 3))


Standardized for leaf images.

🔹 2. Shallow Feature Extractor (Fast Block)

Captures basic texture edges, color gradients.

Conv2D(32, 3×3, activation='relu')
BatchNormalization
MaxPooling2D

🔹 3. Edge Extraction Block

Customized block focusing on disease edges, discoloration streaks, and vein patterns.
Built using Depthwise Separable Convolutions for faster computations.

DepthwiseConv2D(3×3)
PointwiseConv2D(64)
BatchNorm
ReLU
MaxPooling2D

🔹 4. TuneNet Feature Block (The Core Innovation)

A tuned multi-branch block with adjustable depth:

Branch A – Fine-Grain Extractor

Conv2D(64, 3×3)
Conv2D(64, 3×3)


Branch B – Wide Receptive Field Extractor

Conv2D(128, 5×5)


Feature Fusion:

Concatenate([Branch A, Branch B])


Purpose:

Detects multiple scales

Combines coarse+fine leaf textures

Improves disease-spot detection accuracy

🔹 5. High-Level Features
Conv2D(128)
Conv2D(256)
BatchNorm
Dropout
GlobalAveragePooling2D


The dropout stabilizes training on smaller datasets.

🔹 6. Classification Head
Dense(256, activation='relu')  
Dropout(0.3)
Dense(num_classes, activation='softmax')

🎯 Model Advantages
Feature	Benefit
Lightweight CNN + Depthwise Conv	Faster on edge devices
Tunable feature blocks	Research-friendly modifications
Multiscale feature extraction	Better disease spot detection
Regularization-heavy design	Prevents overfitting
Efficient training pipeline	Ideal for Kaggle + local GPU
📊 Dataset & Pipeline

The notebook includes:

✔ Automatic dataset downloading (KaggleHub)
✔ Train–validation split
✔ Real-time augmentation
✔ Preprocessing pipelines
✔ Class visualization
✔ Batch inspection
✔ Training callbacks:

EarlyStopping

ReduceLROnPlateau (dynamic LR tuning)

Checkpointing

📈 Evaluation Tools Included

Confusion Matrix

Accuracy–Loss curves

Class distribution plot

Predictions visualization
