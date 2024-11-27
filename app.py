import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from model import emotion_predict
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    print(f"Received file: {audio_file.filename}")
    print(f"File type: {audio_file.content_type}")
    # Process the audio file as needed before prediction

    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)
    # Assume `predict_emotion` takes the audio file as input and returns the emotion
    emotion = emotion_predict()
    # return jsonify({'emotion': emotion})
    print(emotion)
    result = {
        "emotion": emotion,
        "filename": "example.mp3"
    }

    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
