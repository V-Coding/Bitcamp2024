from flask import Flask, request, jsonify
import sys
import os
sys.path.append('DeepKeyAttack/')
from get_audio_keys import isolate_num_strokes
from infer import predict

app = Flask(__name__)

@app.route('/upload_audio', methods=['POST'])
def post_endpoint():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'})
    

    audio_file = request.files['audio']
    uploads_dir = "./DeepKeyAttack/UploadedAudio/"
    
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if audio_file:
        filename = "server_upload.wav"
        audio_file.save(uploads_dir, filename)
    
    isolate_num_strokes(os.path.join(uploads_dir, filename))
    predictions = predict()
    print(predictions)
    keystroke_string = "".join(predictions)
    return jsonify({'message': 'Received audio file:', 'filename': audio_file.filename, "keystrokes": keystroke_string})

if __name__ == '__main__':
    app.run(debug=False)