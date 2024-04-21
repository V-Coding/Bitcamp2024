from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload_audio', methods=['POST'])
def post_endpoint():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'})

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'})

    return jsonify({'message': 'Received audio file:', 'filename': audio_file.filename})

if __name__ == '__main__':
    app.run(debug=False)