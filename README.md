# Confusion Matrix and Threshold Tuning

## Project Overview

This project analyzes binary classification predictions using a confusion matrix and investigates how different probability thresholds affect precision and recall.

## Dataset

Breast Cancer Wisconsin Diagnostic Dataset.

The dataset contains 569 samples and 30 numerical features.

## Model

Logistic Regression with feature standardization.

## Workflow

1. Load the dataset
2. Explore the data
3. Convert the target labels
4. Split into training and testing sets
5. Standardize features
6. Train Logistic Regression
7. Generate prediction probabilities
8. Evaluate the default 0.5 threshold
9. Test multiple thresholds
10. Compare precision and recall
11. Select a threshold based on the desired recall
12. Visualize the results

## Key Findings

The project demonstrates that changing the classification threshold changes the balance between precision and recall.

A lower threshold generally increases recall, while a higher threshold generally increases precision.

## Tools

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab