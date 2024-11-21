from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

# Python API
@app.route('/python-api', methods=['POST'])
def python_api():
    data = request.json
    print("Received from Spring Boot:", data)
    return jsonify(message="Hello from Python API")

# Spring Boot API 호출
@app.route('/call-spring', methods=['POST'])
def call_spring():
    data = {"message": "Hello from Python!"}
    response = requests.post('http://localhost:8080/api/process', json=data)
    return jsonify(spring_response=response.json())

if __name__ == '__main__':
    app.run(debug=True, port=5000)