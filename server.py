from flask import Flask, request, jsonify
import sys
sys.path.append('DeepKeyAttack/')
from get_audio_keys import isolate_num_strokes
from infer import predict

app = Flask(__name__)

@app.route('/upload_audio', methods=['POST'])
def post_endpoint():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'})
    

    audio_file = request.files['audio']

    
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if audio_file:
        filename = "server_upload.wav"
        audio_file.save("DeepKeyAttack/UploadedAudio/", filename)
    
    isolate_num_strokes("DeepKeyAttack/UploadedAudio/" + filename)
    predictions = predict()
    print(predictions)
    keystroke_string = "".join(predictions)
    return jsonify({'message': 'Received audio file:', 'filename': audio_file.filename, "keystrokes": keystroke_string})

if __name__ == '__main__':
    app.run(debug=False)