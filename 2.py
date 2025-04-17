from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
from utils import load_model, predict_category

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_PATH'] = 'D:\OneDrive\Desktop\dental_disease_predictor-main\iitj_dental_cnn.pth'   # Update with actual model path

# Load model at startup
try:
    model = load_model(app.config['MODEL_PATH'])
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if not model:
            return jsonify({'error': 'Model not loaded'})

        try:
            image = Image.open(filepath)
            prediction = predict_category(model, image)
            return jsonify({
                'status': 'success',
                'prediction': prediction,
                'filename': filename
            })
        except Exception as e:
            return jsonify({
                'error': f'Prediction failed: {str(e)}',
                'filename': filename
            })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=8000, debug=True)
