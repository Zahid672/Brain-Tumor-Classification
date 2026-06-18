# Brain Tumor Classification using Multi-Level Information Fusion

This is the official repository for our paper: **"Hybrid Information Fusion Framework: Deep Representation Fusion and Hyperparameter-Tuned Classifier Ensembling for Brain Tumor Classification"** 

The repository provides a comprehensive implementation of a multi-level information fusion framework for brain tumor classification from MRI images. The proposed approach combines:


- MRI preprocessing and augmentation
- Deep feature extraction using pre-trained CNNs and Vision Transformers (ViTs)
- Representation-level fusion of heterogeneous deep features
- Hyperparameter optimization of machine learning classifiers
- Decision-level classifier ensembling
- Extensive evaluation on three publicly available brain MRI datasets

## Highlights

- Evaluation of **25 pre-trained deep feature extractors**
  - 12 CNN models
  - 13 Vision Transformer models
- Evaluation of **9 machine learning classifiers**
- Feature-level fusion of top-performing deep representations
- Classifier-level ensemble learning using optimized classifiers
- Experiments on both binary and multi-class brain tumor classification datasets

## Architecture

Our proposed architecture

![Brain-Tumor-Classification](Figures/Proposed_Model.png)


## Datasets

The framework was evaluated on:

1. **BT-small-2c** (Tumor vs Non-Tumor)
2. **BT-large-2c** (Tumor vs Non-Tumor)
3. **BT-large-4c** (Glioma, Meningioma, Pituitary, Normal)

## Repository Structure

```text
Brain-Tumor-Classification/
├── datasets/
├── preprocessing/
├── feature_extraction/
├── feature_fusion/
├── hyperparameter_optimization/
├── classifier_ensemble/
├── results/
├── notebooks/
└── README.md