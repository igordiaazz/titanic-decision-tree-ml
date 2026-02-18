# Titanic Survival Prediction: Decision Tree Approach

This repository contains a Machine Learning project focused on predicting passenger survival on the Titanic. Using a Decision Tree Classifier, the model identifies survival patterns based on demographic and voyage data, prioritizing both predictive performance and model interpretability.

---

## Project Overview

The goal of this project is to develop a binary classification model to determine if a passenger survived (1) or perished (0). The implementation follows a standard data science pipeline, including data cleaning, feature engineering, model training, and evaluation.

### Technical Features
* **Data Preprocessing:** Handled missing values in the 'Age' feature using mean imputation and implemented manual Label Encoding for the 'Sex' variable.
* **Feature Selection:** Optimized the dataset by removing non-predictive or high-cardinality columns such as PassengerId, Name, Ticket, Cabin, and Embarked.
* **Model Implementation:** Utilized the Scikit-learn DecisionTreeClassifier with a max_depth of 3 to prevent overfitting and maintain a clear decision-making structure.
* **Visualization:** Generated a Confusion Matrix using Seaborn to analyze classification results, including True Positives and False Negatives.



---

## Tech Stack

* **Language:** Python 3.x
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-learn
* **Data Visualization:** Seaborn, Matplotlib

---

## Getting Started

### Prerequisites
Ensure you have Python 3 installed along with the following dependencies:
```bash
pip install pandas seaborn matplotlib scikit-learn

Installation & Usage
Clone the repository:

Bash
git clone [https://github.com/igordiaazz/titanic-decision-tree.git](https://github.com/igordiaazz/titanic-decision-tree.git)

Navigate to the project folder:

Bash
cd titanic-decision-tree
Run the script:

Bash
python main.py
