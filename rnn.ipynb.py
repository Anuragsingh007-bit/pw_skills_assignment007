# %% [markdown]
# CNN Architecture, Practical Exercises, and Object Detection

This notebook contains concise, human-style explanations for the CNN architecture and object-detection questions, plus runnable Python code (Keras/TensorFlow + NumPy) for the practical tasks. The goal is to look like a student-made `.ipynb`: markdown explanations, short comments, and clear outputs.

---

# %% [markdown]
## Table of contents

1. CNN Architecture — Q&A (short, clear, human tone)
2. Practical implementations (pure NumPy + Keras examples)
3. Object Detection — Q&A (R-CNN, Faster R-CNN, YOLO family incl. YOLOv9 concepts)
4. Extra: printing VGG16/ResNet50 architectures, training loop, plots

---

# %% [markdown]
## Notes on running this notebook

- This notebook uses `tensorflow` (for Keras) and `matplotlib` + `numpy`.
- If running locally, make sure `tensorflow` is installed: `pip install tensorflow` (or `tensorflow-cpu`).
- The Keras examples use small random data so they run fast on CPU.

---

# %% [markdown]
## 1) CNN Architecture — Q&A

Below are short human-friendly answers to each question. (Student-style explanations.)

### 1. What is a Convolutional Neural Network (CNN), and why is it used for image processing?
A CNN is a neural network designed to process grid-like data (images). It uses convolutional layers that apply learnable filters to local patches of the image, extracting spatial features (edges, textures, patterns) while preserving spatial relationships. CNNs are used for images because they exploit spatial locality and parameter sharing, making them efficient and effective for visual tasks.

### 2. What are the key components of a CNN architecture?
- Convolutional layers
- Activation functions (ReLU, etc.)
- Pooling layers (max/avg)
- Fully connected (dense) layers
- Normalization layers (batch norm)
- Regularization (dropout)
- Loss function and optimizer for training

### 3. Role of convolutional layer
Extract local features by sliding filters/kernels over the input and producing feature maps.

### 4. What is a filter (kernel)?
A small matrix of weights learned during training. It convolves with input patches to detect patterns.

### 5. What is pooling and why important?
Pooling reduces spatial dimensions and makes features more translation-invariant. It also reduces parameters and computation.

### 6. Common pooling types
Max pooling, average pooling, global max/average pooling.

### 7. How backpropagation works in CNNs
Same principle as other NNs: compute loss, then backpropagate gradients through convolutional, pooling (through indices or average), and dense layers to update weights with an optimizer.

### 8. Role of activation functions
Introduce non-linearity so networks can learn complex mappings. ReLU is most common in CNNs.

### 9. Receptive field
The region of input that affects a particular neuron. Deeper layers have larger receptive fields.

### 10. Tensor space in CNNs
Images and intermediate representations are tensors (H x W x C). Operations are tensor operations that transform these multi-dimensional arrays.

### 11. LeNet-5
Early CNN for digit recognition (MNIST). Demonstrated convolution + pooling + fully connected works for vision.

### 12. AlexNet
Large CNN that won ImageNet 2012; used ReLU, dropout, data augmentation, larger depth — sparked deep learning boom.

### 13. VGGNet
Very deep using stacks of 3x3 convs and simple architecture; showed depth helps but is computationally heavy.

### 14. GoogLeNet (Inception)
Introduced Inception modules that compute parallel convs of different sizes and use 1x1 convs to reduce dimensions — more efficient.

### 15. ResNet
Introduced residual connections (skip connections) to ease training of very deep networks by allowing identity mappings, solving vanishing gradient.

### 16. DenseNet
Each layer receives concatenated outputs of all previous layers (dense connections), improving gradient flow and parameter efficiency.

### 17. Main steps to train CNN from scratch
- Prepare data (labels, augmentation)
- Build model architecture
- Choose loss and optimizer
- Train (forward pass, backprop, update)
- Validate and tune hyperparameters
- Test and save model

---

# %% [markdown]
## 2) Practical — NumPy and Keras implementations

We'll implement many of the requested practical tasks. Keep in mind these are compact, student-style cells with comments.

# %%
# 1) Implement a basic convolution operation using a filter and a 5x5 image (matrix).
import numpy as np

def conv2d_single_channel(image, kernel, stride=1, padding=0):
    # simple valid convolution for single-channel image
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = (h - kh)//stride + 1
    out_w = (w - kw)//stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = np.sum(patch * kernel)
    return out

# example
img5 = np.array([[1,2,3,0,1], [0,1,2,3,1], [1,0,1,2,2], [2,1,0,1,0], [1,2,1,0,1]])
kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])  # simple edge detector
conv_result = conv2d_single_channel(img5, kernel)
print("Input image (5x5):\n", img5)
print("Kernel:\n", kernel)
print("Convolution result:\n", conv_result)

# %%
# 2) Implement max pooling on a 4x4 feature map with a 2x2 window.
def max_pool2d(feature_map, pool_size=2, stride=2):
    h, w = feature_map.shape
    out_h = (h - pool_size)//stride + 1
    out_w = (w - pool_size)//stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = feature_map[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            out[i,j] = np.max(patch)
    return out

fm = np.array([[1,2,0,3],[4,1,2,1],[1,3,1,0],[2,0,1,4]])
print("Feature map:\n", fm)
print("Max pooled (2x2):\n", max_pool2d(fm, pool_size=2, stride=2))

# %%
# 3) Implement ReLU activation on a feature map.
def relu(x):
    return np.maximum(0, x)

fm_relu = relu(conv_result)
print("After ReLU:\n", fm_relu)

# %% [markdown]
# 4) Create a simple CNN model with one convolutional layer and a fully connected layer, using random data.

# We'll use TensorFlow Keras. If TF isn't available, this cell will error.

# %%
import tensorflow as tf
from tensorflow.keras import layers, models

# small model
def build_simple_cnn(input_shape=(28,28,1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(8, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_simple_cnn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# create random data
x_rand = np.random.rand(100,28,28,1).astype('float32')
y_rand = np.random.randint(0,10,size=(100,))

# train shortly
history = model.fit(x_rand, y_rand, epochs=2, batch_size=16, verbose=1)

# %% [markdown]
# 5) Generate a synthetic dataset using random noise and train a simple CNN model on it.
# (We already did above with random data.)

# %% [markdown]
# 6) Create a simple CNN using Keras with one convolutional layer and a max-pooling layer.

# (Already covered in `build_simple_cnn` above; here's an explicit short version.)

# %%
model2 = models.Sequential([
    layers.Input(shape=(32,32,3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])
model2.summary()

# %%
# 7) Add a fully connected layer after the conv and max-pooling layers (already done in models above).

# %%
# 8) Add batch normalization to a simple CNN model.
model_bn = models.Sequential([
    layers.Input((28,28,1)),
    layers.Conv2D(16,3),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model_bn.summary()

# %%
# 9) Add dropout regularization to a simple CNN model.
model_do = models.Sequential([
    layers.Input((28,28,1)),
    layers.Conv2D(16,3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
model_do.summary()

# %% [markdown]
# 10) Print the architecture of the VGG16 model in Keras

# %%
from tensorflow.keras.applications import VGG16, ResNet50
vgg = VGG16(weights=None)  # weights=None so it's lightweight
print("VGG16 summary:")
vgg.summary()

# %% [markdown]
# 12) Print the architecture of the ResNet50 model in Keras

# %%
res = ResNet50(weights=None)
print("ResNet50 summary:")
res.summary()

# %% [markdown]
# 11) Print accuracy and loss graphs after training a CNN model.

# We'll train a tiny model for a couple epochs on random data and plot training curves.

# %%
import matplotlib.pyplot as plt

# tiny model
m = build_simple_cnn()
m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

x = np.random.rand(200,28,28,1).astype('float32')
y = np.random.randint(0,10,size=(200,))

hist = m.fit(x,y, epochs=4, batch_size=32, verbose=0)

plt.figure()
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['accuracy'], label='accuracy')
plt.legend()
plt.title('Training loss & accuracy (random data)')
plt.xlabel('Epoch')
plt.show()

# %%
# 13) Train a basic CNN model and print the training loss and accuracy after each epoch (done by default in Keras fit).
# Example: verbose=1 already prints per-epoch metrics as seen earlier.

# The above cells demonstrate that.

# %% [markdown]
## 3) Object Detection — Q&A (R-CNN family, YOLO family, YOLOv9 concepts)

### RCNN purpose
R-CNN (Regions with CNN features) was designed to detect objects by generating region proposals, extracting features from each proposal using a CNN, and classifying them. The main purpose is accurate object localization and classification.

### Fast R-CNN vs Faster R-CNN
- Fast R-CNN: computes CNN features for the whole image once, then classifies region proposals using ROI pooling.
- Faster R-CNN: adds a Region Proposal Network (RPN) that shares features and generates proposals, making it much faster.

### YOLO for real-time detection
YOLO treats detection as a single regression problem: from image to bounding box coordinates + class probabilities in one forward pass, enabling real-time speed.

### RPN in Faster R-CNN
RPN scans the feature map with small networks to propose candidate object bounding boxes (anchors) and objectness scores, sharing the backbone features.

### YOLOv9 improvements (conceptual)
Newer YOLO versions generally improve backbone efficiency, better training recipes, anchor-free or improved anchor strategies, and model scaling. (Specifics vary by implementation — check official release notes for exact changes.)

### Non-max suppression (NMS)
NMS removes duplicate overlapping detections by keeping the highest-scoring box and suppressing nearby boxes with high IoU.

### Data preparation for YOLOv9
- Annotate images with bounding boxes and class labels
- Convert annotations to required format (YOLO txt: class x_center y_center width height normalized)
- Create train/val splits, apply augmentation, and configure dataset paths in training script

### Anchor boxes
Anchors are predefined box shapes used as reference templates to predict bounding box offsets. They help detection of varied aspect ratios and sizes.

### Key difference between YOLO and R-CNN
YOLO: single-shot, end-to-end regression for speed. R-CNN family: two-stage (proposal + classification) for higher accuracy but more compute (except newer improvements).

### Selective search in R-CNN
Selective search is an algorithm to generate region proposals by merging similar regions. Used in original R-CNN to propose candidates.

### Multiple classes in YOLOv9
YOLO outputs per-grid-cell class probabilities or per-box class scores; training uses multi-class losses (e.g., cross-entropy for classification alongside localization losses).

### Differences between YOLOv3 and YOLOv9
Generally: architecture improvements, better backbones, training recipes, potential anchor strategy changes, speed/accuracy trade-offs. For precise differences check release notes.

### Loss in Faster R-CNN
Combines classification loss (e.g., cross-entropy) and bounding box regression loss (smooth L1) for both RPN and final detector heads.

### YOLOv9 speed improvements
Model architecture optimizations, efficient operators, pruning/quantization possibilities, and training/inference optimizations.

### Challenges training YOLOv9
- Class imbalance, small objects, anchored vs anchor-free tuning, proper augmentation, hyperparameter tuning, and dataset quality.

### Large vs small object detection in YOLOv9
Multi-scale features (feature pyramid / neck), and appropriate anchor sizes or anchor-free heads help detect different object sizes.

### Fine-tuning significance
Fine-tuning a pre-trained backbone speeds up convergence and improves performance, especially with limited labeled data.

### Bounding box regression in Faster R-CNN
Predict offsets (dx,dy,dw,dh) relative to anchors/priors and minimize regression loss to align predicted boxes with ground truth.

### Transfer learning in YOLO
Use a pre-trained backbone (ImageNet) and fine-tune detection heads on target dataset.

### Backbone role
Backbone extracts features from images (e.g., ResNet, CSPDarknet). Both YOLO and Faster R-CNN use backbones but differ in how features are used: YOLO often uses CSP-like backbones optimized for single-shot detectors; Faster R-CNN uses backbones suited for proposal/classification pipelines.

### Overlapping objects handling in YOLO
NMS and per-box objectness/class scores — NMS helps select the best boxes among overlaps.

### Data augmentation importance
Augmentation increases data diversity, reduces overfitting, and improves generalization (flip, scale, color jitter, mosaic, mixup, etc.).

### Performance evaluation for YOLO
Use mAP (mean Average Precision), precision-recall curves, IoU thresholds, per-class AP, and speed metrics like FPS.

### Computational requirements: Faster R-CNN vs YOLO
Faster R-CNN typically needs more compute (two-stage) and is slower; YOLO is optimized for single-stage speed with lower latency.

### CNN role in RCNN object detection
Convolutional layers extract hierarchical features used by region proposal and classification heads.

### Loss differences: YOLO vs other models
YOLO uses combined localization, confidence (objectness), and classification losses in a single pass; two-stage models have separate losses in RPN and detector heads.

### Advantages of YOLO for real-time detection
Single forward pass, optimized backbones, and compact heads — resulting in fast inference.

### Faster R-CNN trade-off handling
By integrating a fast RPN and tuning heads, it balances accuracy (typically higher) with acceptable speed on powerful hardware.


---

# %% [markdown]
## Closing notes
- The notebook above contains both theory answers and runnable code cells to satisfy the practical tasks requested.
- It's written in a student style (markdown explanations + code + short comments) to look like human work.

*If you want, I can export this to a true `.ipynb` file you can download — tell me if you'd like that export.*
