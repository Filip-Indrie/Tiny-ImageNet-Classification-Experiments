# Tiny ImageNet Classification Experiments

## Overview

This repository contains a series of experiments focused on image classification on the **Tiny ImageNet** dataset using multiple deep learning architectures. The goal of this project is to analyze how neural network architectures, optimizers, and weight initialization strategies affect performance.
The project includes implementations of both architectures inspired by research papers and custom-designed models, all evaluated under the same training protocol to ensure a fair comparison.

## Dataset

The dataset used is **Tiny ImageNet**, which contains 200 object classes, each with 500 training images, 50 validation images, and 50 test images. All images are 64 × 64 pixels. All images were resized to 128 × 128 pixels to preserve more spatial detail for deeper networks.

## Goals

The main objectives are:
1. To evaluate how optimizer choice and weight initialization influence model performance.
2. To compare different CNN architectures under the same training conditions.
3. To identify architectural traits that contribute to improved performance.

## Optimizers and Weight Initialization Methods

### Optimizers Tested
* SGD (learning rate = 0.1)
* SGD with Nesterov momentum (learning rate = 0.1)
* Adam (learning rate = 1e-3)
* AdamW (learning rate = 1e-3)

### Weight Initialization Methods
* Default (framework initialization)
* Xavier Uniform
* Xavier Normal
* Kaiming Uniform
* Kaiming Normal

## Hyperparameter Search Strategy

ResNet-50 was used as a baseline model to evaluate all combinations of optimizers and weight initializations. Based on validation accuracy, the best-performing combination was **AdamW with Kaiming Uniform initialization**. This combination was then used to train all the other models.

## Training Protocol

All models were trained for a maximum of 100 epochs, using early stopping with a patience of 10 epochs, while monitoring the validation accuracy.

## Metrics Recorded

For every training run, the following metrics were logged:
* Training accuracy
* Training loss
* Validation accuracy
* Validation loss
* Best epoch (based on validation accuracy)
* Total training time (in seconds)

## Models

The project experiments with both architectures from research papers and custom model designs. All models are trained using the same optimizer, initialization method, data preprocessing and training procedure to enable direct and fair performance comparison.

