from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
import logging
from werkzeug.utils import secure_filename
from inference import main
import argparse
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'images' not in request.files:
        logger.error('No images uploaded in the request')
        return jsonify({'error': 'No images uploaded'}), 400

    files = request.files.getlist('images')
    prompt = request.form.get('prompt', '')

    # Save uploaded files
    image_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_paths.append(filepath)

    if not image_paths:
        logger.error('No valid images uploaded (invalid file extensions)')
        return jsonify({'error': 'No valid images uploaded'}), 400

    results = []
    try:
        for image_path in image_paths:
            # Mock args for inference
            args = argparse.Namespace(
                model_path="deepseek-vl2-tiny",
                image_path=image_path,
                prompt=prompt,
                chunk_size=20
            )

            # Call inference
            logger.info('Starting inference for image: %s', image_path)
            result = main(args)  # main() returns a dict with "answer" and "vg_image_path"

            results.append({
                'image_path': image_path,
                'result': result.get('answer', 'No answer generated'),
                'prompt': prompt
            })

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        logger.error('Error during inference: %s', str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

