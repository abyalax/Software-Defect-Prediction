# Cross-Domain Generalization Evaluation of CodeBERT and CodeT5-small for Just-in-Time Software Defect Prediction on the Defectors Dataset

## Background

Software quality is one of the critical aspects of software development. One of the common challenges in this process is the occurrence of *software defects*, which are errors in program code that may cause software to behave unexpectedly or fail to meet its intended requirements. Manually identifying software defects requires considerable time and resources. Therefore, Software Defect Prediction (SDP) has been developed to identify software components that are likely to contain defects using machine learning approaches.

One approach to SDP is *Just-in-Time Software Defect Prediction* (JIT-SDP), which aims to predict whether a code change is likely to introduce software defects. This approach enables early identification of potential risks during software development, allowing code reviews and testing activities to be performed more efficiently and effectively.

The advancement of pre-trained language models for source code has created opportunities to apply deep learning approaches to JIT-SDP. Models such as CodeBERT and CodeT5 have been developed to capture semantic representations of source code and have demonstrated capabilities across various software engineering tasks, including code defect detection.

However, software defect prediction performance can be affected by differences in project characteristics. Projects developed for Machine Learning or Deep Learning purposes may exhibit different coding patterns, structures, and complexity compared to non-Machine Learning projects. These differences may lead to performance degradation when a model trained on one domain is applied to another.

Mahbub et al. (2023) introduced the Defectors dataset, which contains 213,420 Python source code file-change records collected from 24 open-source repositories. These repositories cover various domains, including Machine Learning, automation, and the Internet of Things. This makes the dataset suitable for evaluating software defect prediction across diverse project characteristics.

The original Defectors dataset uses a random record-level data split, in which almost all repositories are represented in both the training and testing sets. Such a splitting strategy is insufficient for measuring cross-project and cross-domain generalization. Therefore, this research reconstructs the data split at the repository level.

Based on these challenges, this research focuses on evaluating the cross-domain generalization capabilities of CodeBERT and CodeT5-small for JIT-SDP. Both models are evaluated across Machine Learning and non-Machine Learning domains under within-domain and cross-domain training and testing scenarios. TF-IDF + Logistic Regression is employed as a baseline for comparison.

This research is expected to provide empirical insights into the capabilities of Transformer-based models for software defect prediction and their ability to maintain predictive performance when applied to projects from different domains.

## Research Questions

1. How do CodeBERT and CodeT5-small perform in Just-in-Time Software Defect Prediction compared to the TF-IDF + Logistic Regression baseline?
2. Are there differences in defect prediction performance between Machine Learning and non-Machine Learning projects?
3. How well do CodeBERT and CodeT5-small generalize when trained on one domain and evaluated on a different domain?
4. How do CodeBERT and CodeT5-small compare in terms of computational efficiency during training and inference?

## Research Objectives

1. To evaluate the performance of CodeBERT and CodeT5-small in performing Just-in-Time Software Defect Prediction.
2. To analyze differences in defect prediction performance between Machine Learning and non-Machine Learning projects.
3. To evaluate the cross-domain generalization capabilities of both models through cross-domain training and testing scenarios.
4. To compare the computational efficiency of both models based on training time, inference time, and resource utilization.

## Research Benefits

### 1. Theoretical Benefits

- Contribute to existing research in Software Defect Prediction, particularly the application of Transformer-based models for software defect prediction.
- Provide empirical insights into the influence of project domain differences on JIT-SDP model performance.
- Serve as a reference for future research on machine learning generalization in software engineering.

### 2. Practical Benefits

- Provide insights into which models are more suitable for predicting defects in code changes.
- Illustrate the capabilities of prediction models when applied to projects with different domain characteristics.
- Support model selection decisions based on predictive performance and computational requirements.

## Research Methodology

This research adopts a quantitative approach using a comparative experimental method. The study evaluates the cross-domain generalization capabilities of pre-trained code language models, specifically CodeBERT and CodeT5-small, for Just-in-Time Software Defect Prediction (JIT-SDP) across different software project domains.

The research consists of the following stages:

### 1. Literature Review

A literature review is conducted on previous studies related to Software Defect Prediction, particularly Just-in-Time Software Defect Prediction, the application of Transformer models for source code understanding, and the concepts of generalization and cross-domain transfer learning.

The literature review serves as the theoretical foundation for determining the research approach, model selection, and evaluation methods.

### 2. Dataset Collection and Exploration

The dataset used in this research is [Defectors](https://zenodo.org/records/7708984), a publicly available dataset containing Python source code file-change records collected from various software repositories.

The dataset is selected because it includes projects with diverse characteristics and application domains.

The research data is divided into two primary domains:

- **Machine Learning:** Repositories whose primary purpose is related to Machine Learning and Deep Learning development.
- **Non-Machine Learning:** Repositories whose primary purpose is outside Machine Learning and Deep Learning development.

Domain classification is performed based on the characteristics and primary functionality of each repository.

Further exploratory analysis is conducted to understand dataset characteristics, class distributions, and the overall quality of the data used in this research.

### 3. Data Preprocessing

Data preprocessing is performed to prepare the dataset for model training and evaluation. This stage includes data quality inspection, handling invalid or irrelevant records, and preparing source code changes as model inputs.

Additionally, code change characteristics that may influence prediction results are examined to identify potential data patterns that could cause models to make predictions based on specific characteristics rather than understanding the code changes themselves.

The unit of analysis in this research is a file-level code change, consistent with the structure of the Defectors dataset.

### 4. Model Implementation

Three modeling approaches are used as experimental subjects and comparison methods:

- **TF-IDF + Logistic Regression:** Used as a baseline to establish the performance of a conventional machine learning approach for software defect prediction.
- **CodeBERT:** A Transformer-based model pre-trained to understand representations of programming languages and natural language.
- **CodeT5-small:** A pre-trained encoder-decoder model developed for various source code understanding and processing tasks.

CodeBERT and CodeT5-small are implemented using a fine-tuning approach to adapt their pre-trained code representations to the software defect classification task.

### 5. Experimental Scenarios

Experiments are conducted to evaluate model performance under two primary scenarios:

- **Within-domain:** Models are trained and tested using data from the same domain. This scenario evaluates the models' predictive capabilities when applied to projects with similar characteristics.
- **Cross-domain:** Models are trained using data from one domain and tested on a different domain. Evaluation is performed in both directions:
  - Machine Learning → Non-Machine Learning
  - Non-Machine Learning → Machine Learning

Data splitting is performed at the repository level to ensure that records from the same repository are not simultaneously used in the training and testing sets.

This approach is applied to reduce the potential for data leakage and provide a more representative evaluation of model generalization capabilities.

### 6. Model Evaluation

Model evaluation is conducted to measure the ability of each model to predict software defects.

Predictive performance is measured using the following classification metrics:

- **PR-AUC:** Measures the model's ability to identify defective instances by considering the balance between precision and recall.
- **F1-Score:** Measures the harmonic mean of precision and recall.
- **MCC (Matthews Correlation Coefficient):** Evaluates overall classification quality by considering all prediction outcomes.
- **ROC-AUC:** Measures the model's ability to distinguish between defective and defect-free classes.

In addition to predictive performance, computational efficiency is evaluated based on training time and inference time to compare the computational requirements of each model.

### 7. Results Analysis

Experimental results are analyzed to compare CodeBERT and CodeT5-small against the baseline model, examine differences in predictive performance between Machine Learning and non-Machine Learning domains, and investigate the impact of domain shifts on model performance.

The analysis also examines the extent to which each model can maintain its predictive capabilities when applied to a different domain.

Within-domain and cross-domain evaluation results serve as the basis for assessing the generalization capabilities of both models and identifying which model offers a better balance between predictive performance and computational efficiency.

## Installation

Install [uv](https://docs.astral.sh/uv/), the Python package and project manager used by this repository.

Install the core and development dependencies:

```Shell
uv sync
```

The Python version is pinned in `.python-version`.

## Development

Run the test suite and the code-style checks:

```Shell
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Usage
See [Defectors Dataset Documentation](command/defectors.md).

_Last Updated 4 November 2026_