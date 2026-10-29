
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("SWYNEX Task 3 - Intelligent Feature")
print("AI Customer Feedback Classification")

# Load and validate dataset
try:
    data = pd.read_csv("customer_feedback.csv")

    required_columns = {"feedback", "sentiment"}
    if not required_columns.issubset(data.columns):
        raise ValueError("Dataset must contain feedback and sentiment columns.")

    data = data.dropna(subset=["feedback", "sentiment"])
    data["feedback"] = data["feedback"].astype(str).str.strip()
    data["sentiment"] = data["sentiment"].astype(str).str.strip()
    data = data[
        (data["feedback"] != "") & (data["sentiment"] != "")
    ]

    if data["sentiment"].nunique() < 2:
        raise ValueError("At least two sentiment classes are required.")

except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as error:
    print("Dataset Error:", error)
    raise SystemExit(1)

# Build AI model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5)
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        C=10
    ))
])

try:
    model.fit(data["feedback"], data["sentiment"])
    print("\nAI Model trained successfully!")
except Exception as error:
    print("Training Error:", error)
    raise SystemExit(1)


# Intelligent prediction with confidence
def predict_feedback(feedback):
    if not isinstance(feedback, str) or not feedback.strip():
        return None, None, "Please enter valid customer feedback."

    try:
        feedback = feedback.strip()
        prediction = model.predict([feedback])[0]
        probabilities = model.predict_proba([feedback])[0]
        confidence = max(probabilities) * 100

        return prediction, confidence, None

    except Exception as error:
        return None, None, str(error)


# Model evaluation examples
test_examples = [
    ("The product is excellent", "Positive"),
    ("Very bad customer service", "Negative"),
    ("The product is okay", "Neutral"),
    ("I am very happy with the service", "Positive"),
    ("The delivery was terrible", "Negative")
]

print("\n--- MODEL EVALUATION ---")

correct = 0
total = len(test_examples)

for feedback, expected in test_examples:
    prediction, confidence, error = predict_feedback(feedback)

    if error:
        print("Error:", error)
        continue

    is_correct = prediction.lower() == expected.lower()

    if is_correct:
        correct += 1

    print("\nFeedback:", feedback)
    print("Expected:", expected)
    print("Predicted:", prediction)
    print(f"Confidence: {confidence:.2f}%")
    print("Result:", "PASS" if is_correct else "FAIL")

print(f"\nExample Test Accuracy: {correct}/{total}")
print("Note: These are demonstration examples, not a")
print("reliable estimate of real-world model accuracy.")


# Failure case demonstration
print("\n--- FAILURE CASE TEST ---")

failure_examples = [
    "Great, another delayed delivery!",
    "The service was not bad",
    "I expected better quality"
]

for feedback in failure_examples:
    prediction, confidence, error = predict_feedback(feedback)

    if error:
        print("Error:", error)
    else:
        print("\nFeedback:", feedback)
        print("Prediction:", prediction)
        print(f"Confidence: {confidence:.2f}%")
        print("Review manually: These phrases may be ambiguous.")


# Interactive user interface
print("\n--- CUSTOMER FEEDBACK CLASSIFIER ---")
print("Type 'exit' to close the application.")

while True:
    try:
        feedback = input("\nEnter customer feedback: ")

        if feedback.strip().lower() == "exit":
            print("Application closed successfully.")
            break

        prediction, confidence, error = predict_feedback(feedback)

        if error:
            print("Error:", error)
            continue

        print("AI Prediction:", prediction)
        print(f"Confidence Score: {confidence:.2f}%")

        if confidence < 60:
            print("Warning: Low confidence. Manual review recommended.")

    except (KeyboardInterrupt, EOFError):
        print("\nApplication closed.")
        break
