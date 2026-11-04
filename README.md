# ML Assignment 2: Teen Digital Wellbeing Classifier
## a. Problem Statement

The objective of this project is to build a machine learning classification pipeline to predict a teenager's digital wellbeing status based on their digital lifestyle, academic performance, and mental health indicators. We aim to compare the efficacy of 5 different standard ML classifiers, deploy the best-performing models via an interactive Streamlit web application, and evaluate their ability to accurately categorize a user as "Healthy", "Moderate", or "At Risk".

## b. Dataset Description

**Source:** [Kaggle (argonnxx/teen-mental-health)](https://www.kaggle.com/datasets/argonnxx/teen-mental-health)

**Target Variable:** digital_wellbeing_flag (Multi-class: **"Healthy"**, **"Moderate"**, **"At Risk"**)

**Total Features:** 16 features (including age, gender, daily_social_media_hours, screen_time_before_sleep, sleep_hours, stress_level, anxiety_level, digital_wellbeing_flag etc.)

**Instance Size:** 1200 rows, exceeding the assignment requirement of 500.

**Nature of Data:** Synthetic survey data reflecting teen demographic and behavioral attributes.

## c. Github Repository Link

**Link:** [ML Assignment 2](https://github.com/SubhajitS/classifier-problems)

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8583 | 0.9525 | 0.8588 | 0.8583 | 0.8582 | 0.7327 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 0.8292 | 0.9173 | 0.8361 | 0.8292 | 0.8181 | 0.6667 |
| Naive Bayes | 0.4625 | 0.8403 | 0.6792 | 0.4625 | 0.4222 | 0.3489 |
| Support Vector Machine | 0.8625 | 0.9668 | 0.8621 | 0.8625 | 0.8614 | 0.7397 |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance                                                                                                                                                                                                                                               |
| :--- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Logistic Regression** | Performed strongly (~85.8% accuracy & F1, +0.732 MCC), indicating that simple linear combinations of digital habits and sleep hours are highly predictive of the wellbeing status.                                                                                                |
| **Decision Tree** | Achieved a perfect 1.0000 across all metrics. Because this is a possible synthetic dataset. Which means the target was possibly generated using hard-coded threshold rules (if/else conditions), which the decision tree effortlessly and perfectly reconstructed.                |
| **KNN** | Scored decently (~82.9% accuracy) but struggled compared to linear models because distance-based nearest-neighbor voting blurs near the sharp threshold cutoff borders separating the classes.                                                                                    |
| **Naive Bayes** | Showed the poorest performance (~46.2% accuracy). The assumption of independent features completely failed here, as features like `sleep_hours`, `screen_time_before_sleep`, and `daily_social_media_hours` are highly correlated.                                                |
| **Support Vector Machine** | Achieved the highest performance among non-tree models (~86.2% accuracy and F1, ~0.967 AUC and +0.739 MCC). The RBF kernel successfully navigated the non-linear boundaries but could not draw the perfectly rectangular axis-aligned splits that exist in the synthetic dataset. |
| **Overall, Winner for your dataset?** | Although **Decision Tree** is the technical winner on this specific synthetic dataset. However, **Support Vector Machine (SVM)** is the most realistic and robust model for generalizing to messy, real-world survey data.                                                        |

## f. Developer/ Evaluator notes

**Teen_Mental_Health.csv**: Unstratified raw data from kaggle. Used during the training phase to produce [train_data_internal.csv](model/train_data_internal.csv) and [test_data.csv](/test_data.csv) for training and testing respectively.

**test_data.csv**: Should be used to test the models in the Streamlit app

**train_all.py**: Run the preprocessing and training of the data with following command ``python train_all.py``

**app.py**: Runs the streamlit application using command ``streamlit run app.py``

**utils.py**: Contains utility methods used by preprocessing and training data 