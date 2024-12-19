import requests

url = "http://127.0.0.1:5000/chatgpt"

# 테스트할 메시지 리스트
test_messages = [
    "피고가 2022. 5. 26. 2021 원고에 대하여 한 년 귀속 양도소득세 204,317,749원에 관한 경정청구 처리한 사건에서 어떤 판결이 났었지?"
]

for message in test_messages:
    response = requests.post(url, json={"message": message})
    print(f"Input: {message}")
    print(f"Response: {response.json()['response']}\n")
