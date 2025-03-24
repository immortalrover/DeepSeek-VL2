from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from inference import main
import argparse

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
        return jsonify({'error': 'No valid images uploaded'}), 400

    # Prepare conversation for inference
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{prompt}",
            "images": image_paths
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # Mock args for inference
    args = argparse.Namespace(
        model_path="deepseek-vl2-tiny",
        image_path=image_paths[0],  # Use the first image path
        prompt=prompt,
        chunk_size=-1
    )

    # Call inference
    try:
        main(args)
        return jsonify({'message': 'Processing completed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

