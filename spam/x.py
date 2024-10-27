from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
nlp_model = pipeline("conversational", model="microsoft/DialoGPT-small")

@app.route('/generate_response', methods=['POST'])
def generate_response():
    user_input = request.json['text']
    response = nlp_model(user_input)[0]['generated_text']
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(port=5000)