
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("customer_feedback.csv")

# Create improved AI model
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

# Train AI model
model.fit(data["feedback"], data["sentiment"])

print("AI Model trained successfully!")

# Example predictions
examples = [
    "The product is excellent",
    "Very bad customer service",
    "The product is okay",
    "I am very happy with the service",
    "The delivery was terrible"
]

print("\nAI CUSTOMER FEEDBACK CLASSIFICATION")

for feedback in examples:
    prediction = model.predict([feedback])[0]
    print("Feedback:", feedback)
    print("Prediction:", prediction)
    print("--------------------------")

# User input
while True:
    feedback = input(
        "\nEnter customer feedback (or exit): "
    )

    if feedback.lower() == "exit":
        break

    prediction = model.predict([feedback])[0]
    print("AI Prediction:", prediction)
