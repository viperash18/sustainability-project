# Deep Neural Network Compression Project

## Overview

This project implements a complete pipeline for compressing deep neural networks while preserving high accuracy. A convolutional neural network is trained on the CIFAR-10 dataset and then compressed using pruning and quantization techniques.

The pipeline includes:
- Magnitude-based pruning
- K-means weight quantization
- Efficient storage using `.npz` format

The objective is to reduce model size and memory usage with minimal loss in performance.

---

## Quick Pipeline Summary

Running the project performs the following steps:

1. Train baseline CNN model  
2. Apply pruning (50% sparsity)  
3. Apply quantization (k = 16 clusters)  
4. Save compressed model  
5. Reload and evaluate performance  

---

## Features

- End-to-end compression pipeline  
- Lightweight CNN architecture for CIFAR-10  
- Pruning and quantization integration  
- Fine-tuning after compression  
- Disk size and memory comparison  
- Model reconstruction support  

---

## Model Architecture

- Model: SmallCIFARNet  
- Custom CNN designed for CIFAR-10 classification  
- Suitable for compression experiments  

---

## Technologies Used

- Python  
- PyTorch  
- NumPy  
- CIFAR-10 Dataset  

---

## Project Structure

```
DNN_Compression_Project/
│── main.py

├── compressed_models/
│   ├── model.pth
│   ├── compressed.npz

├── compression/
│   ├── prune.py
│   ├── quantization.py
│   ├── conv2d.py
│   ├── linear.py

├── data/
│   └── data_loader.py

├── models/
│   ├── model_cifar.py
│   ├── model_fruits.py

├── utils/
│   ├── config.py
│   ├── loading.py
│   ├── memory_profiler.py
│   ├── test_eval.py
```

---

## Installation

```
pip install torch torchvision numpy
```

---

## How to Run

```
python main.py
```

---

## Loading Saved Models

### Load Compressed Model

```python
from utils.loading import load_model_from_npz

model = load_model_from_npz("compressed_models/compressed.npz")
model.eval()
```

### Load Original Model

```python
import torch

model = torch.load("compressed_models/model.pth")
model.eval()
```

---

## Results

Baseline Model:
- Accuracy: 92.05%

After Pruning (50% sparsity):
- Accuracy: 93.75%

After Quantization (k = 16):
- Accuracy: 92.86%

Final Decompressed Model:
- Accuracy: 92.86%

---

## Compression Performance

- Original Model Size: 8.45 MB  
- Compressed Model Size: 1.32 MB  
- Compression Ratio: 6.42x  

Memory Usage:
- Allocated: 44.69 MB  
- Reserved: 344.00 MB  

---

## Core Concepts

Magnitude-Based Pruning:
- Removes low-importance weights  
- Produces sparse networks  
- Improves efficiency  

K-Means Quantization:
- Groups weights into clusters  
- Replaces weights with centroids  
- Reduces precision and storage  

Accuracy vs Compression:
- Trade-off between size and performance  
- Achieved high compression with minimal accuracy loss  

---

## Key Achievements

- Maintained over 92% accuracy after compression  
- Reduced model size by more than 6x  
- Preserved performance after decompression  

---

## Future Improvements

- Structured pruning  
- Quantization-aware training  
- Deployment on edge devices  
- Hardware-aware optimizations  

---

## Contributors

- Naveen Ola  
- Pratik Beniwal 
- Ashwary Rathore  
- Ninad Tagade
- Madhur Kumar  
- Pradeep Meena  
- Samayraj Meena  
- Sarthak Gupta  
- Ayaan Abbas  
- Arnav Agarwal

