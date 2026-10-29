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


---

# SWYNEX Task 3 - Intelligent Feature

## Project Overview

This project improves the AI-Based Customer Feedback Classification prototype developed during Task 2 of the SWYNEX Technologies Artificial Intelligence Internship.

The application uses Python, TF-IDF, and Logistic Regression to classify customer feedback into three categories:

- Positive
- Negative
- Neutral

## New Features Added

1. **Prediction Confidence:** Displays the model's estimated confidence percentage for each prediction.
2. **Error Handling:** Handles empty feedback, missing dataset files, invalid dataset structure, and interrupted input.
3. **Model Evaluation:** Compares expected and predicted sentiment labels using five demonstration examples.
4. **Failure Case Analysis:** Demonstrates potentially incorrect or ambiguous predictions.
5. **Interactive Interface:** Allows users to enter customer feedback and receive predictions.

## Model Evaluation Results

| Test | Expected | Predicted | Result |
|---|---|---|---|
| The product is excellent | Positive | Positive | PASS |
| Very bad customer service | Negative | Negative | PASS |
| The product is okay | Neutral | Neutral | PASS |
| I am very happy with the service | Positive | Negative | FAIL |
| The delivery was terrible | Negative | Negative | PASS |

**Result:** 4 out of 5 demonstration examples were classified correctly (80%).

This result is not a reliable estimate of real-world model accuracy because the evaluation uses a very small set of examples.

## Failure Case Examples

- "Great, another delayed delivery!" was predicted as Positive.
- "The service was not bad" was predicted as Negative.
- "I expected better quality" was predicted as Negative.

These examples highlight limitations when processing sarcasm, negation, and ambiguous language. Not every prediction listed above is necessarily incorrect.

## Interactive Demo Result

Input: The service was amazing 3.

Prediction: Positive

Confidence Score: 60.96%

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Google Colab

## How to Run

1. Download or clone this repository.
2. Install the required packages:

   pip install pandas scikit-learn

3. Keep customer_feedback.csv in the same folder as app.py.
4. Run:

   python app.py

5. Enter customer feedback to receive a sentiment prediction and confidence score.
6. Type exit to close the application.

Alternatively, open the Task 3 notebook in Google Colab and upload customer_feedback.csv before running it.

## Limitations

- The model uses a small training dataset.
- Predictions may be incorrect for unfamiliar or ambiguous feedback.
- Confidence percentages are not guaranteed to be calibrated probabilities.
- More training data and independent evaluation are needed before production use.

## Conclusion

Task 3 demonstrates intelligent sentiment prediction, confidence estimation, input validation, model evaluation, and interactive feedback classification.

Developed as part of the SWYNEX Technologies Artificial Intelligence Internship.
