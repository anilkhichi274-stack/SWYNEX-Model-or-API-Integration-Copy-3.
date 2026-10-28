# AI-Based Customer Feedback Classification

## SWYNEX Technologies — Artificial Intelligence Internship
**Task 2: Model or API Integration**

## Project Overview
This project demonstrates a machine learning prototype that classifies customer feedback into three sentiment categories: Positive, Negative, and Neutral.

## Problem Statement
Businesses receive customer feedback regularly. Manually categorizing feedback takes time. This project explores how machine learning can help automate sentiment classification.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer (Character N-grams)
- Logistic Regression
- Google Colab

## Dataset
The project uses a manually created sample dataset containing 24 feedback entries:
- 8 Positive
- 8 Negative
- 8 Neutral

Dataset file: `customer_feedback.csv`

## How the Model Works
1. Load customer feedback data from CSV.
2. Convert feedback text into TF-IDF character n-gram features.
3. Train a Logistic Regression classifier.
4. Predict sentiment for new customer feedback.

## How to Run
1. Clone or download this repository.
2. Install dependencies: `pip install pandas scikit-learn`
3. Run: `python app.py`
4. Enter customer feedback to receive a predicted sentiment.
5. Type `exit` to stop the program.

You can also open `SWYNEX_AI_Model_Test.ipynb` in Google Colab. Upload `customer_feedback.csv` to the Colab session before running the notebook cells.

## Example Predictions
| Customer Feedback | Model Prediction |
|---|---|
| The product is excellent | Positive |
| Very bad customer service | Negative |
| The product is okay | Neutral |
| I am very happy with the service | Negative |
| The delivery was terrible | Negative |

## Limitations
The model is trained on only 24 manually created examples. Predictions may be incorrect, especially for unseen phrases. The five examples above are demonstrations, not an accuracy evaluation. A larger labeled dataset and a separate test set are needed to measure generalization.

## Internship Details
Organization: SWYNEX Technologies
Domain: Artificial Intelligence
Task: 2 — Model or API Integration
Project: AI-Based Customer Feedback Classification
