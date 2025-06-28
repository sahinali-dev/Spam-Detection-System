# main.py: Model Training and Saving

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import joblib

# Download stopwords if not already done
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Keep only alphabets
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load dataset
data = pd.read_csv("spamsms/spam_data.csv",  sep="\t", header=None, names=["label", "message"])
data = data.rename(columns={"v1": "label", "v2": "message"})
data = data[['label', 'message']]
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Preprocess the text data
data['message'] = data['message'].apply(preprocess_text)

# Split data into training and testing sets
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with vectorizer and classifier
model = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=10000)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Save the trained model
joblib.dump(model, 'spam_detector_model.pkl')
print("Model saved as spam_detector_model.pkl")
