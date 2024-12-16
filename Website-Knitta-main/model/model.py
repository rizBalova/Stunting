from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("copy_of_cloud_salinan_prita.py")

# Initialize Flask app
app = Flask(_name_)

@app.route('/')
def Home():
    return render_template("homepage.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Data input (format JSON dari frontend)
    data = request.json
    input_data = np.array(data['input']).reshape(1, -1, 1)  # Sesuaikan input LSTM
    
    # Prediksi menggunakan model
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if _name_ == '_main_':
    app.run(debug=True)
