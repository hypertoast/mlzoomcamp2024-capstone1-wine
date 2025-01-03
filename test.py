import requests

# Sample wine data (same as we used in testing)
wine_data = {
    'fixed acidity': 7.4,
    'volatile acidity': 0.7,
    'citric acid': 0.0,
    'residual sugar': 1.9,
    'chlorides': 0.076,
    'free sulfur dioxide': 11.0,
    'total sulfur dioxide': 34.0,
    'density': 0.9978,
    'pH': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4
}

# Make prediction request
url = 'http://localhost:9696/predict'
response = requests.post(url, json=wine_data)

# Print results
if response.status_code == 200:
    result = response.json()
    print('\nWine Quality Prediction:')
    print(f"Predicted class: {result['quality_prediction']}")
    print('\nClass Probabilities:')
    for class_name, prob in result['probability'].items():
        print(f'{class_name}: {prob:.3f}')
else:
    print(f'Error: {response.status_code}')
    print(response.json())