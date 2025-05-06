# CV-NLP-Transfer-Learning

This repository contains python notebooks demonstrating the application of transfer learning techniques in both Computer Vision (CV) and Natural Language Processing (NLP).

## Description

The primary goal of this repository is to showcase practical examples of leveraging pre-trained models to solve specific tasks efficiently. This includes using established architectures for image classification and fine-tuning transformer models for various NLP tasks.

## Notebooks

1.  **CIFAR-100 Image Classification with Transfer Learning**
    * **Description:** Explores using pre-trained models (VGG19, EfficientNetB0) for classifying images from the CIFAR-100 dataset. Demonstrates adapting models trained on ImageNet to a different, smaller dataset. Discusses potential challenges with architecture suitability for small images (e.g., EfficientNetB0 performance).
    * **Libraries:** TensorFlow/Keras, Matplotlib, NumPy.

2.  **Fine-tuning & Using Hugging Face Transformers for NLP**
    * **Description:** Demonstrates how to use the Hugging Face `transformers` library to:
        * Load pre-trained transformer models (e.g., BERT, GPT-2, DistilBERT).
        * Use pre-trained models directly for tasks like sentiment analysis, text generation, or feature extraction (zero-shot).
        * Fine-tune a pre-trained transformer on a specific dataset for a downstream task (e.g., text classification, question answering).
    * **Libraries:** Hugging Face `transformers`, `datasets`, PyTorch/TensorFlow, NumPy.

## Key Concepts Demonstrated

* **Transfer Learning:** Utilizing knowledge gained from one task/dataset (e.g., ImageNet, large text corpora) to improve performance on a different but related task/dataset.
* **Fine-tuning:** Adjusting the weights of a pre-trained model on a new dataset specific to the target task.
* **Pre-trained Models:** Using models readily available with pre-learned weights (VGG19, EfficientNetB0, BERT, etc.).
* **Computer Vision:** Image Classification (CIFAR-100).
* **Natural Language Processing:** Various tasks using Transformers (e.g., classification, generation).
* **Frameworks/Libraries:** TensorFlow, Keras, PyTorch, Hugging Face Transformers, Google Colab.
