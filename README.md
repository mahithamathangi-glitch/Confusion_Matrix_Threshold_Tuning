# Confusion Matrix and Threshold Tuning

## 📌 Project Overview

This project analyzes binary classification predictions using a **confusion matrix** and investigates how changing the **classification probability threshold** affects model performance.

The project demonstrates the relationship between:

- Classification thresholds
- Accuracy
- Precision
- Recall
- F1 Score
- False Positives
- False Negatives

A **Logistic Regression** model is trained using the **Breast Cancer Wisconsin Diagnostic Dataset** available through Scikit-learn.

The project also includes an interactive **Streamlit application** that allows users to change the classification threshold and observe how the model's performance changes.

---

## 🎯 Objective

The main objective of this project is to understand how probability thresholds affect binary classification performance.

The project specifically focuses on:

1. Building a binary classification model.
2. Generating class probabilities using `predict_proba()`.
3. Creating and visualizing a confusion matrix.
4. Testing multiple probability thresholds.
5. Comparing precision and recall at different thresholds.
6. Studying the trade-off between false positives and false negatives.
7. Selecting a threshold based on a defined recall requirement.
8. Building an interactive Streamlit application for threshold tuning.

---

## 📊 Dataset

### Breast Cancer Wisconsin Diagnostic Dataset

The dataset is provided by Scikit-learn and contains:

- **569 samples**
- **30 numerical features**
- Two classes:
  - Benign
  - Malignant

For this project, the target variable is represented as:

```text
0 = Benign
1 = Malignant

The malignant class is treated as the positive class so that recall can be used to measure how effectively malignant cases are detected.


---

🤖 Machine Learning Model

The project uses:

Logistic Regression

Logistic Regression was selected because it is a suitable binary classification algorithm and provides class probabilities through:

predict_proba()

These probabilities are used for threshold tuning.


---

🔄 Project Workflow

Breast Cancer Dataset
        ↓
Data Inspection
        ↓
Target Transformation
        ↓
Train/Test Split
        ↓
Feature Standardization
        ↓
Logistic Regression
        ↓
Probability Prediction
        ↓
Default Threshold (0.50)
        ↓
Confusion Matrix
        ↓
Multiple Threshold Testing
        ↓
Precision / Recall Comparison
        ↓
Threshold Selection
        ↓
Streamlit Application


---

🧹 Data Preparation

The following preprocessing steps were performed:

1. Loaded the Breast Cancer Wisconsin dataset.


2. Separated features and target variable.


3. Converted the target labels so that malignant represents the positive class.


4. Split the dataset into training and testing sets.


5. Used an 80:20 train-test split.


6. Applied feature standardization using StandardScaler.




---

🧠 Confusion Matrix

The confusion matrix evaluates the classification results using four categories:

Term	Meaning

True Positive (TP)	Malignant case correctly classified as malignant
True Negative (TN)	Benign case correctly classified as benign
False Positive (FP)	Benign case incorrectly classified as malignant
False Negative (FN)	Malignant case incorrectly classified as benign


The confusion matrix is visualized using a Seaborn heatmap.


---

📈 Classification Metrics

Accuracy

Accuracy measures the proportion of total predictions that were correct.

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision

Precision measures how many predicted positive cases were actually positive.

Precision = TP / (TP + FP)

Recall

Recall measures how many actual positive cases were correctly detected.

Recall = TP / (TP + FN)

F1 Score

F1 Score combines precision and recall into a single metric.

F1 = 2 × (Precision × Recall) / (Precision + Recall)


---

🎚️ Threshold Tuning

A classification model can produce a probability instead of directly producing a class label.

For example:

Predicted probability = 0.72

At a threshold of:

0.50

the prediction is positive because:

0.72 >= 0.50

However, at a threshold of:

0.80

the same prediction becomes negative because:

0.72 < 0.80

Therefore, changing the threshold changes the final classification results.


---

🔍 Thresholds Tested

The project evaluates multiple probability thresholds.

The analysis includes thresholds ranging from lower to higher classification cutoffs.

The notebook and Streamlit application calculate the following metrics for each threshold:

Accuracy

Precision

Recall

F1 Score

True Negatives

False Positives

False Negatives

True Positives


The complete threshold analysis is stored in:

threshold_results.csv


---

⚖️ Precision-Recall Trade-off

Changing the classification threshold creates a trade-off between precision and recall.

Generally:

Lower Threshold

Lower threshold
      ↓
More positive predictions
      ↓
Higher recall
      ↓
Potentially more false positives
      ↓
Precision may decrease

Higher Threshold

Higher threshold
      ↓
Fewer positive predictions
      ↓
Potentially fewer false positives
      ↓
Precision may increase
      ↓
Some positive cases may be missed
      ↓
Recall may decrease

The project visualizes this relationship using a precision-recall comparison graph.


---

🎯 Recommended Threshold Strategy

Instead of assuming that the default threshold of 0.50 is always optimal, this project selects a threshold based on an explicit performance requirement.

For this project, the threshold analysis prioritizes:

Recall >= 0.95

Among thresholds satisfying this recall requirement, the threshold with the highest precision is selected.

If no tested threshold satisfies the required recall, the project falls back to the threshold with the highest F1 Score.

This provides a reproducible and application-oriented threshold selection strategy.

The exact selected threshold and corresponding metrics are generated automatically by the notebook.


---

📊 Project Outputs

The project produces the following outputs:

1. Confusion Matrix

Shows:

True Positives

True Negatives

False Positives

False Negatives


2. Threshold Results

A comparison table containing:

Threshold

Accuracy

Precision

Recall

F1 Score

TN

FP

FN

TP


3. Precision vs Recall Graph

Shows how precision and recall change as the classification threshold changes.

4. F1 Score vs Threshold

Shows the change in F1 Score across different thresholds.

5. Precision-Recall Curve

Provides a graphical representation of the precision-recall relationship.

6. Streamlit Application

Provides an interactive interface where users can change the classification threshold and observe changes in:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix



---

🖥️ Streamlit Application

The project includes an interactive Streamlit application.

Run the application using:

python -m streamlit run app.py

The application provides:

Interactive threshold slider

Performance metrics

Confusion matrix

Threshold analysis table

Precision-recall visualization



---

🗂️ Project Structure

Confusion_Matrix_Threshold_Tuning/
│
├── images/
│   ├── confusion_matrix_0.5.png
│   ├── precision_recall_threshold.png
│   ├── f1_threshold.png
│   ├── precision_recall_curve.png
│   ├── threshold_results.png
│   └── streamlit_app.png
│
├── Confusion_Matrix_Threshold_Tuning.ipynb
├── predictions.csv
├── threshold_results.csv
├── app.py
├── requirements.txt
├── deployment.yaml
├── rollback_evidence.md
├── README.md
└── .gitignore


---

🛠️ Technologies Used

Technology	Purpose

Python	Programming language
NumPy	Numerical computation
Pandas	Data manipulation
Scikit-learn	Machine learning and evaluation
Matplotlib	Data visualization
Seaborn	Confusion matrix visualization
Streamlit	Interactive web application
Google Colab	Notebook development
Git	Version control
GitHub	Project repository
Kubernetes YAML	Deployment configuration



---

📦 Installation

Clone the repository:
https://github.com/mahithamathangi-glitch/Confusion_Matrix_Threshold_Tuning/edit/main

Move into the project directory:

cd Confusion_Matrix_Threshold_Tuning

Install the required packages:

pip install -r requirements.txt


---

▶️ Running the Streamlit Application

Run:

python -m streamlit run app.py

The application will normally be available at:

http://localhost:8501


---

📓 Running the Notebook

The main analysis is available in:

Confusion_Matrix_Threshold_Tuning.ipynb

The notebook can be opened using:

Google Colab

Jupyter Notebook

JupyterLab



---

📁 Output Files

predictions.csv

Contains:

Actual class

Predicted malignant probability

Prediction using the default threshold

Prediction using the recommended threshold


threshold_results.csv

Contains performance metrics for the tested probability thresholds.


---

🚀 Deployment Configuration

A Kubernetes deployment configuration is included in:

deployment.yaml

It defines:

Application deployment

Container configuration

Container port

Kubernetes service


The deployment file is included as part of the project submission structure.


---

🔄 Version Control and Rollback

Git was used for version control.

The stable project version was tagged as:

v1.0

To view the available tags:

git tag

To inspect the v1.0 version:

git show v1.0

To list files included in the tagged version:

git ls-tree -r --name-only v1.0

The rollback documentation is available in:

rollback_evidence.md


---

📌 Key Findings

The project demonstrates that:

1. A confusion matrix provides detailed information about classification errors.


2. Probability thresholds directly affect class predictions.


3. Lower thresholds generally increase the number of positive predictions.


4. Lower thresholds can improve recall while potentially reducing precision.


5. Higher thresholds can reduce false positives but may increase false negatives.


6. The default threshold of 0.50 is not necessarily optimal for every application.


7. Threshold selection should be based on the objectives and error costs of the application.


8. Precision and recall should be evaluated together rather than relying only on accuracy.




---

💡 Why Threshold Tuning Matters

In real-world classification systems, false positives and false negatives can have different consequences.

For example, in a medical screening context, missing a positive case may be more concerning than generating an additional false positive.

Therefore, the classification threshold should be selected according to the practical requirements of the application rather than automatically assuming that 0.50 is optimal.


---

🎓 Interview Questions

What is a confusion matrix?

A confusion matrix is a table used to evaluate classification predictions by comparing actual and predicted classes. It contains True Positives, True Negatives, False Positives and False Negatives.

How does changing the threshold affect recall?

Lowering the classification threshold generally causes more observations to be classified as positive. This can increase recall because fewer actual positive cases are missed, although it may also increase false positives.

Why might the default 0.5 threshold not be optimal?

A threshold of 0.5 is only a default decision rule. It does not necessarily reflect the application's requirements or the relative costs of false positives and false negatives. Threshold tuning can therefore be used to obtain a more appropriate precision-recall balance.


---

👩‍💻 Project Type

Machine Learning / Binary Classification / Model Evaluation


---

📜 License

This project is created for educational and internship purposes.


---

⭐ Conclusion

This project demonstrates the practical importance of evaluating and tuning classification thresholds rather than relying solely on default model predictions.

By combining confusion matrix analysis, precision, recall, F1 Score and threshold tuning, the project provides a practical approach to understanding and improving binary classification decisions.

