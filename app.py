from flask import Flask, request, jsonify
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS
from flask import render_template

# Load the trained model
clf = joblib.load('knife_detector_model.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Function to extract HOG features
def extract_hog_features(image, resize_dim=(64, 64)):
    image_resized = cv2.resize(image, resize_dim)
    gray_image = rgb2gray(image_resized)
    features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    return features

@app.route('/')
def nav():
    return render_template("index.html")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Decode the incoming image
        data = request.get_json()
        frame_data = data['frame']
        frame_data = frame_data.split(',')[1]  # Remove the header
        image = Image.open(BytesIO(base64.b64decode(frame_data)))
        image = np.array(image)

        # Convert BGR to RGB if needed (depends on the input format)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract features and predict
        features = extract_hog_features(image)
        prediction = clf.predict([features])[0]
        result = "Knife" if prediction == 1 else "No Knife"

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
