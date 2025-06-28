from flask import Flask, request, jsonify, render_template
import joblib
import traceback

app = Flask(__name__, template_folder="C:/Users/sahin/OneDrive/Desktop/Project")

# Load the saved model
model = joblib.load("spam_detector_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get("message", "")
        if not message:
            return jsonify({"error": "No message provided!"}), 400

        # Predict using the model
        prediction = model.predict([message])[0]
        return jsonify({"message": message, "prediction": "spam" if prediction == 1 else "ham"})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(debug=True)
