from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Sample data (replace this with your actual dataset)
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7],
    'feature2': [8, 9, 10, 11, 12, 13, 14],
    'target': [0, 1, 1, 0, 0, 1, 0]
})

# Train a simple Random Forest model
X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# API endpoint to predict transaction status
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json['transactions']['input_data']
    df = pd.DataFrame(input_data['values'], columns=input_data['fields'])
    predictions = model.predict(df)
    result = [{'status': 'Fraudulent' if pred == 1 else 'Legitimate'} for pred in predictions]
    return jsonify(predictions=result)

if __name__ == '__main__':
    app.run(port=3000)
