# Student Engagement Prediction

Predicting at-risk students at Tomorrow University using routine learning activity data.

## Overview

Built a machine learning pipeline to classify students as highly engaged or at risk of low engagement — enabling academic support teams to intervene early, before students drop out or fall behind.

## Dataset

500 student records from a SQLite database (`student_data.db`), with features including platform logins, lessons completed, forum posts, assignments submitted, and mentoring status.

## Pipeline

- **Preprocessing** — outlier capping (IQR/Z-score), median/mode imputation, spelling correction
- **Feature Selection** — hypothesis testing (Pearson, Chi-Square, ANOVA) + Random Forest `SelectFromModel`
- **Feature Engineering** — `total_activity` composite score, mentoring × activity interaction term
- **Modelling** — Random Forest Classifier tuned via GridSearchCV (5-fold CV, 80/20 split)

## Results

| Metric | Score |
|---|---|
| Test Balanced Accuracy | **90.7%** |
| CV Balanced Accuracy | 79.2% |

Top predictors: platform login hours, mentoring status, lessons completed.

## Tech Stack

Python · scikit-learn · pandas · SQLite

## Why Balanced Accuracy?

The dataset has a severe class imbalance (88.8% positive). Balanced accuracy equally weights both classes, making it a more honest evaluation metric than raw accuracy.

## Usage

This project runs on **Google Colab** — no local setup required.

1. Clone or download this repository
2. Download `student_data.db` from the repo
3. Open `Carino_Capstone_MasteringDataScienceF.ipynb` in [Google Colab](https://colab.research.google.com/)
4. Upload `student_data.db` to your Colab session (via the file panel on the left)
5. Run all cells from top to bottom
6. Read Carino_Model_Communication.pdf to aid your understanding about the Machine Learning model's results.
