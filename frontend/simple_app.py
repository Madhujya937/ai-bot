from flask import Flask, render_template, request, jsonify
import requests
import json
import os

app = Flask(__name__)

# Backend API URL
BACKEND_URL = "http://localhost:8000"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        model = data.get('model', 'mythomax')
        
        # Prepare the request to backend
        payload = {
            'message': message,
            'model': model
        }
        
        # Send request to backend
        response = requests.post(f"{BACKEND_URL}/chat", data=payload)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': f'Backend error: {response.status_code}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Frontend error: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Save file temporarily
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        # Send file to backend
        with open(filepath, 'rb') as f:
            files = {'file': (filename, f, 'application/octet-stream')}
            response = requests.post(f"{BACKEND_URL}/upload", files=files)
        
        # Clean up
        os.remove(filepath)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': f'Backend error: {response.status_code}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 