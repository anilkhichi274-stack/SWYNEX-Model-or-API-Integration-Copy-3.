
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load customer feedback dataset
data = pd.read_csv("customer_feedback.csv")

# Create AI classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(data["feedback"], data["sentiment"])

print("AI Customer Feedback Classification")
print("Model trained successfully!")

# Test the model with example feedback
examples = [
    "The product is excellent",
    "Very bad customer service",
    "The product is okay"
]

for feedback in examples:
    prediction = model.predict([feedback])[0]
    print(f"Feedback: {feedback}")
    print(f"Predicted Sentiment: {prediction}")
    print("-------------------------")

# Interactive prediction
while True:
    feedback = input("Enter customer feedback (or type exit): ")

    if feedback.lower() == "exit":
        break

    prediction = model.predict([feedback])[0]
    print("Predicted Sentiment:", prediction)
