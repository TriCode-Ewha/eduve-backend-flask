import requests

url = "http://127.0.0.1:5000/chatgpt"

# 테스트할 메시지 리스트
test_messages = [
    "안녕하세요",
    "What is the weather like today?",
    "Tell me a joke!",
    "How can I learn Python?",
    "What's the capital of France?"
]

for message in test_messages:
    response = requests.post(url, json={"message": message})
    print(f"Input: {message}")
    print(f"Response: {response.json()['response']}\n")
