from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import joblib

# Load the model and the scaler
model = tf.keras.models.load_model('my_lstm_model.h5')
scaler = joblib.load('scaler.pkl')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    data = request.form
    
    # Extract the features from the form data
    features = {
        'el_access_urban': float(data['el_access_urban']),
        'el_demand': float(data['el_demand']),
        'el_access_rural': float(data['el_access_rural']),
        'population': float(data['population']),
        'net_imports': float(data['net_imports']),
        'el_demand_pc': float(data['el_demand_pc']),
        'fin_support': float(data['fin_support']),
        'el_from_gas': float(data['el_from_gas']),
        'pop_no_el_access_total': float(data['pop_no_el_access_total']),
        'urban_share': float(data['urban_share']),
        'income_group_num': float(data['income_group_num']),
        'year': float(data['year']),
        'el_access_total': float(data['el_access_total']),
        'gdp_pc': float(data['gdp_pc']),
        't_demand': float(data['t_demand']),
        'supply_rate': float(data['el_demand']) / float(data['el_access_total'])  # Calculated feature
    }
    
    # Prepare features for prediction
    feature_values = [
        features['el_access_urban'],
        features['el_demand'],
        features['el_access_rural'],
        features['population'],
        features['net_imports'],
        features['el_demand_pc'],
        features['fin_support'],
        features['el_from_gas'],
        features['pop_no_el_access_total'],
        features['urban_share'],
        features['income_group_num'],
        features['year'],
        features['el_access_total'],
        features['gdp_pc'],
        features['t_demand'],
        features['supply_rate']
    ]
    
    # Ensure feature_values has the correct number of features
    if len(feature_values) != 16:
        return jsonify({'error': 'Incorrect number of features provided.'})
    
    # Scale the features
    feature_values = np.array(feature_values).reshape(1, -1)
    feature_values_scaled = scaler.transform(feature_values)
    
    # Reshape for LSTM
    feature_values_scaled = feature_values_scaled.reshape((feature_values_scaled.shape[0], 1, feature_values_scaled.shape[1]))
    
    # Make the prediction
    prediction = model.predict(feature_values_scaled)
    prediction = np.abs(prediction[0][0])
    
    # Return the prediction
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
