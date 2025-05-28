# Military Aircraft Segmentation Project

This repository contains scripts for training and evaluating segmentation models for military aircraft detection using the YOLO framework.

## Scripts Overview

### 1. Download Data (`1-download-data.py`)
- Connects to the Segments.ai API to download aircraft image dataset
- Uses a predefined API key to access the "AIRCRAFT_GROUPED" dataset
- Exports the dataset to COCO-instance format for further processing

### 2. Convert to YOLO Segmentation Format (`2-convert_to_yolo_seg.py`)
- Converts COCO-instance annotations to YOLO segmentation format
- Processes both polygon and RLE (Run-Length Encoding) segmentation formats
- Normalizes coordinates and creates label files in the YOLO format
- Outputs the converted annotations to the "yolo_labels" directory

### 3. Split YOLO Dataset (`3-split_yolo_dataset.py`)
- Splits the dataset into train (80%), validation (10%), and test (10%) sets
- Ensures balanced distribution of classes across splits
- Creates appropriate directory structure for YOLO training
- Generates a data.yaml file with dataset configuration
- Maps class IDs to aircraft types: combat, uav, support, transport, helicopter, tiltrotor

### 4. Augment YOLO Dataset (`4-augment_yolo.py`)
- Implements data augmentation for the training set
- Applies various transformations: horizontal/vertical flipping, rotation, brightness/contrast adjustments
- Preserves proper segmentation coordinates during transformations
- Outputs augmented images and corresponding label files
- Reports class distribution before and after augmentation

### 5. Compare Augmentations (`5-compare-aug.py`)
- Visualizes original and augmented image pairs with their segmentation masks
- Creates overlay masks with different colors for each class
- Displays class information alongside visualizations
- Helpful for visual verification of the augmentation process

### 6. Display Class Examples (`6-display_class_examples.py`)
- Creates a visual grid showing examples of each aircraft class
- Displays one representative image for each of the 6 classes
- Useful for understanding the dataset and verifying class distribution

### 7. Finetune YOLO Models (`7-finetune-yolo.py`)
- Configures and executes training of YOLO segmentation models
- Starts from pretrained YOLO8n-seg weights
- Saves training parameters, model weights, and performance metrics
- Implements early stopping and periodic model saving
- Outputs training metadata for later analysis

### 8. Evaluate Models (`8-evaluate_models.py`)
- Evaluates all trained models on the test dataset
- Detects model weights files automatically
- Calculates and reports key metrics: mAP50, mAP50-95 for both detection and segmentation
- Saves evaluation results to the "tests" directory
- Handles multiple model versions and configurations

### 9. Rashomon Analysis (`9-rashomon.py`)
- Implements the Rashomon effect analysis for multiple trained models
- Calculates pixel-wise disagreement between different models (M_PDA metric)
- Identifies the model with the lowest pixel discrepancy
- Saves analysis results to the "rashomon" directory
- Useful for model selection in ensemble learning approaches
