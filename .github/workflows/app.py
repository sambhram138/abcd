from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)
model = joblib.load('diabetes_model.joblib')
@app.route('/', methods=['GET'])
def home():
  return jsonify({'status': 'API is running'})
@app.route('/predict', methods=['POST'])
def predict():
  data = request.get_json(force=True)
  features = np.array(data['features']).reshape(1, -1)
  prediction = model.predict(features)
  return jsonify({'prediction': float(prediction[0])})
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
