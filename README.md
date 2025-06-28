
# Spam Detection System 📩🚫

A Flask-based machine learning web application that detects spam messages with high precision.  
The system uses **TF-IDF vectorization** and a **Random Forest Classifier**, achieving **97.8% accuracy**.

---

## 🚀 Key Features

- 🔍 **Spam Detection**: Classifies input messages as spam or not spam.
- 🧹 **Text Preprocessing**: Cleaned using stopwords removal, punctuation removal, stemming.
- 📊 **ML Pipeline**: TF-IDF vectorizer + Random Forest Classifier.
- 🌐 **Flask Web Interface**: Simple UI for easy message input and prediction.
- 🧪 **API Ready**: Easily extendable via REST endpoints for integration.

---

## ⚙️ Technologies Used

- **Backend**: Python, Flask  
- **ML & NLP**: scikit-learn, NLTK, Pandas  
- **Frontend**: HTML, CSS, JavaScript

---

## 🧪 Accuracy

Achieved **97.8% accuracy** using TF-IDF features and Random Forest Classifier trained on SMS spam dataset.

---

## ▶️ How to Run

```bash
1. pip install -r requirements.txt
2. python app.py
3. Open browser at http://127.0.0.1:5000/