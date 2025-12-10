# WingLandmarks
WingLandmarks is a complete deep learning pipeline for automated detection and analysis of vein junctions in insect wing images. Combining computer vision algorithms with YOLOv8 pose estimation, it provides researchers with accurate morphological data for taxonomic studies and population genetics.

# 🦋 WingLandmarks: Automated Insect Wing Landmark Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-blue.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5061/dryad.qz612jmh1-blue)](https://doi.org/10.5061/dryad.qz612jmh1)

**An end-to-end computer vision pipeline for automated detection and analysis of vein junctions in insect wings using hybrid traditional methods and deep learning.**

---

## 🔬 Overview

**WingLandmarks** is a research-grade computer vision pipeline designed to automate the detection of anatomical landmarks (vein junctions) on insect wings. This tool addresses a critical bottleneck in insect morphometrics by replacing manual landmark annotation with an automated, reproducible, and high-throughput analysis system.

The pipeline combines:
- **Traditional computer vision methods** (SIFT/ORB/junctions) for robust feature extraction
- **Deep learning** (YOLOv8-Pose) for precise landmark localization
- **Advanced preprocessing** (clustering, normalization) for optimal data preparation

### 🎯 Target Applications
- **Taxonomic studies** and species identification
- **Population genetics** and evolutionary biology
- **Agricultural pest monitoring** (Ceratitis fruit flies)
- **High-throughput morphometric analysis**

---

## ✨ Key Features

### 🚀 **Performance**
- **High accuracy**: 76.4% mAP@50 for keypoint detection
- **Fast processing**: Seconds per image vs. manual minutes/hours
- **GPU acceleration**: CUDA-optimized for rapid processing

### 🔧 **Technical Capabilities**
- **Hybrid approach**: Combines classical CV with deep learning
- **Smart clustering**: Automatic reduction of duplicate landmarks
- **Multi-format outputs**: YOLO, COCO, CSV, JSON annotations
- **End-to-end pipeline**: From raw images to trained models
- **Reproducible analysis**: Consistent results across datasets

---

## 🧬 Scientific Background

### The Morphometric Challenge
Insect wing venation patterns contain valuable taxonomic and phylogenetic information. Traditional morphometric analysis requires manual annotation of homologous landmarks—a time-consuming process subject to observer bias.

### Dataset Foundation
This project uses the **"Tsetse fly wing landmark data for morphometrics"** dataset (Dryad DOI:10.5061/dryad.qz612jmh1), containing:
- **14,354 wing pairs** from field-collected specimens
- **Two key species**: *Glossina pallidipes* and *Glossina morsitans*
- **High-resolution microscopy images** with associated biological metadata

### Research Significance
Automating this process enables:
- **Large-scale comparative studies**
- **Real-time species identification**
- **Monitoring of morphological changes** in response to environmental factors
- **Integration with genomic data** for genotype-phenotype mapping

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) for accelerated processing
- 8GB+ RAM for dataset processing

### Quick Installation
```bash
# Clone the repository
git clone https://github.com/mounasaidi/WingLandmarks.git
cd WingLandmarks

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional but recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
