# Convolutional Neural Networks and Transfer Learning with TensorFlow & Keras

## Overview

This repository is part of the **Deep Learning using TensorFlow and Keras** course and focuses on **Convolutional Neural Networks (CNNs)** and **transfer learning for image classification**.

The notebooks progress from **basic CNN architectures** to **training deep networks from scratch**, and finally to **leveraging pre-trained models (VGG16)** using feature extraction and fine-tuning.

---

## 1) Classification Performance Metrics
**Notebook:** `ClassificationPreformanceMatrix.ipynb`

This notebook evaluates **classification models** using standard performance metrics.

### Code snippet
```python
# Confusion Matirx
def confusion_matrix(TP, FP, FN, TN):
        """
        get confusion matrix as defined in figure
        """
        cm = np.array([[TP, FP],
                      [FN, TN]])

        num_p = TP + FP
        num_n = TN + FN

        cm_norm = np.array([[TP/num_p, FP/num_p],
                            [FN/num_n, TN/num_n]])
        return cm, cm_norm
```

### Key concepts
- Accuracy and error analysis
- Confusion matrix interpretation
- Model evaluation

---

## 2) LeNet-5 CNN Architecture
**Notebook:** `CNN_LeNet_5.ipynb`

This notebook implements the **LeNet-5 CNN architecture**, one of the earliest CNN models.

### Key concepts
- Convolution layers
- Pooling layers
- Fully connected layers

---

## 3) CNN for Image Classification
**Notebook:** `ImageClassification_CNN.ipynb`

This notebook builds a **custom CNN** for image classification tasks.

### Key concepts
- Feature extraction with convolutions
- End-to-end image classification

---

## 4) Loading Image Datasets from Directory
**Notebook:** `ImageDatasetFromDirectory.ipynb`

This notebook demonstrates how to load image datasets using TensorFlow utilities.

### Code snippet
```python
import numpy as np
from dataclasses import dataclass
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers

plt.rcParams['figure.figsize'] = (12, 9)
block_plot = False

# Fix seeds for reproducibility.
SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
```

### Key concepts
- Directory-based datasets
- Train/validation splits
- Image preprocessing

---

## 5) Overfitting and Data Augmentation
**Notebook:** `Overfitting_DataAugmentation.ipynb`

This notebook addresses **overfitting** using **data augmentation** techniques.

### Key concepts
- Overfitting symptoms
- Data augmentation strategies
- Improving generalization

---

## 6) Using Pre-trained Models
**Notebook:** `PretrainModelsForImageClassification.ipynb`

This notebook explores **pre-trained CNN models** for image classification.

### Code snippet
```python
all_modules = dir(tf.keras.applications)

# Navigate the tf.keras.applications namespace and display the available models.
for module in all_modules:
    if module[0].islower():
        if module != 'imagenet_utils':
            # Ignore imagenet_utils.
            print(f"Model Family: {bold}{module}{end}")
        if module == "mobilenet_v3":
            # Handel special case for mobilenet_v3.
            temp = "MobileNetV3Large"
            print(f"\t  |__ {temp}")
            temp = "MobileNetV3LSmall"
            print(f"\t  |__ {temp}")
```

### Key concepts
- Transfer learning basics
- Feature reuse from large datasets

---

## 7) Fine-Tuning Pre-trained Models
**Notebook:** `TransferLearning_FineTuning_PreTrainedLayer.ipynb`

This notebook demonstrates **fine-tuning selected layers** of a pre-trained network.

### Code snippet
```python
weights='imagenet',
                                                   )
# First make the convolutional base trainable.
vgg16_conv_base.trainable = True
print('All weights trainable, fine tuning...')
```

### Key concepts
- Freezing base layers
- Fine-tuning higher layers
- Optimizing transferred models

---

## 8) VGG16 Feature Extraction (ASL Dataset)
**Notebook:** `TransferLearning_VGG_16_FeactureExtraction_ASLdata.ipynb`

This notebook applies **VGG16 feature extraction** on ASL image data.

### Code snippet
```python
print('Loading model with ImageNet weights...')
vgg16_conv_base = tf.keras.applications.vgg16.VGG16(input_shape=input_shape,
                                                    include_top=False, # We will supply our own top.
                                                    weights='imagenet',
                                                   )
# Set the `trainable` attribute of the convolutional base to `False` so that the weights are not changed.
vgg16_conv_base.trainable = False
print('All weights trainable, fine tuning...')
```

---

## 9) VGG16 Feature Extraction (Balls Dataset)
**Notebook:** `TransferLearning_VGG_16_FeatureExtraction_BallsData.ipynb`

This notebook applies the same **VGG16 feature extraction pipeline** to a different dataset.

### Code snippet
```python
print('Loading model with ImageNet weights...')
vgg16_conv_base = tf.keras.applications.vgg16.VGG16(input_shape=input_shape,
                                                    include_top=False, # We will supply our own top.
                                                    weights='imagenet',
                                                   )
# Set the `trainable` attribute of the convolutional base to `False` so that the weights are not changed.
vgg16_conv_base.trainable = False
print('All weights trainable, fine tuning...')

# Top = false, to give our own dense layer, we just want the convolutional base layer
```

---

## 10) Training VGG16 from Scratch
**Notebook:** `VGG_16_TrainFromScratch.ipynb`

This notebook trains a **deep VGG-style CNN from scratch** without pre-trained weights.

### Key concepts
- Deep CNN training
- Weight initialization
- Computational cost vs transfer learning

---

## Requirements

```bash
pip install tensorflow numpy matplotlib
```

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  
