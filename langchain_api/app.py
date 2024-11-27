from flask import Flask, jsonify, request
import openai

app = Flask(__name__)

# OpenAI API 키 설정
openai.api_key = "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"  # 여기에 발급받은 API 키를 입력하세요

@app.route('/chatgpt', methods=['POST'])
def chatgpt():
    try:
        data = request.json
        user_message = data.get("message", "")

        # ChatGPT API 호출 (최신 방식)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 또는 "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        # 응답에서 메시지 추출
        gpt_response = response['choices'][0]['message']['content']
        return jsonify({"response": gpt_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
