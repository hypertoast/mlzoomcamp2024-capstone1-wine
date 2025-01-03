import pickle
from flask import Flask, request, jsonify
import pandas as pd
from model import WineQualityModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('wine-quality-service')

app = Flask('wine-quality')

def load_model():
    try:
        logger.info("Loading model from disk...")
        with open('model/wine_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log incoming request
        wine_data = request.get_json()
        logger.info("Received prediction request:")
        logger.info(f"Input data: {wine_data}")

        # Log feature names and values
        df = pd.DataFrame([wine_data])
        logger.info(f"Features received: {df.columns.tolist()}")
        
        # Make prediction
        logger.info("Making prediction...")
        prediction, probabilities = model.predict(df)
        
        # Get class order and create probability dictionary
        classes = model.model.classes_
        prob_dict = {
            class_name: float(prob) 
            for class_name, prob in zip(classes, probabilities[0])
        }
        
        # Prepare response
        result = {
            'quality_prediction': prediction[0],
            'probability': prob_dict
        }
        
        # Log prediction results
        logger.info("Prediction complete")
        logger.info(f"Predicted class: {prediction[0]}")
        logger.info(f"Class probabilities: {prob_dict}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    logger.info("Health check request received")
    status = {
        'status': 'ok',
        'model_loaded': model is not None
    }
    logger.info(f"Health check response: {status}")
    return jsonify(status)

if __name__ == "__main__":
    logger.info("Starting Wine Quality Prediction Service...")
    app.run(debug=True, host='0.0.0.0', port=9696)