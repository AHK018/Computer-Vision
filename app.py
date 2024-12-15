import time
from flask import Flask, request, jsonify, send_file, render_template
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS
from PIL import Image
import io
import torch
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



# Initialize Flask app
app = Flask(__name__)
CORS(app)

clf = joblib.load('knife_detector_model.pkl')

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

num_classes = 2  # Knife + background
model_save_path = "knife_detection_fasterrcnn.pth"

# Load and initialize the model
model = get_model(num_classes)
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
model.eval()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Open the image
    image = Image.open(file).convert("RGB")

    # Preprocess the image
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract predictions
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Draw bounding boxes on the image
    confidence_threshold = 0.5  # Adjust as needed
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Plot the image with bounding boxes
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_np)

    for box, label, score in zip(boxes, labels, scores):
        if score > confidence_threshold:
            x1, y1, x2, y2 = box.cpu().numpy()
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{label.item()} {score:.2f}", color="red", fontsize=12, bbox=dict(facecolor="yellow", alpha=0.5))

    # Save the result to an in-memory file
    output_buffer = io.BytesIO()
    plt.axis('off')
    plt.savefig(output_buffer, format='PNG', bbox_inches='tight', pad_inches=0)
    output_buffer.seek(0)

    return send_file(output_buffer, mimetype='image/png')

# Function to extract HOG features
def extract_hog_features(image, resize_dim=(64, 64)):
    image_resized = cv2.resize(image, resize_dim)
    gray_image = rgb2gray(image_resized)
    features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    return features

# Function for sliding window detection
def sliding_window(image, window_size=(64, 64), step_size=32):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

@app.route('/')
def nav():
    return render_template("index.html")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Start timer for FPS calculation
        start_time = time.time()

        # Decode the incoming frame
        data = request.get_json()
        frame_data = data['frame']
        frame_data = frame_data.split(',')[1]  # Remove the base64 header
        image = Image.open(BytesIO(base64.b64decode(frame_data)))
        image = np.array(image)

        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        original_frame = image.copy()
        detections = []

        # Sliding window parameters
        window_size = (64, 64)
        step_size = 32

        # Perform sliding window detection
        for (x, y, window) in sliding_window(image, window_size, step_size):
            if window.shape[0:2] != window_size:
                continue

            # Extract features and predict
            features = extract_hog_features(window)
            prediction = clf.predict([features])[0]

            if prediction == 1:  # Knife detected
                detections.append((x, y, x + window_size[0], y + window_size[1]))

        # Draw bounding boxes on detections
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Convert the processed frame back to base64 for response
        _, buffer = cv2.imencode('.jpg', original_frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        return jsonify({'image': f'data:image/jpeg;base64,{image_base64}', 'fps': fps})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
