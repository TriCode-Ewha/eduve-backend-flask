from flask import Flask, jsonify, request
import requests
import openai

app = Flask(__name__)


# OpenAI API 키 설정
openai.api_key = ""  
# api_key = "" 


@app.route('/chatgpt', methods=['POST'])
def chatgpt():
    try:
        data = request.json
        user_message = data.get("message", "")

        # ChatGPT API 호출 (최신 방식)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 또는 "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer based on the content follow. content : \"이 원고에게 명의신탁한 3) 2021. 12. 22. “GG 것으로 이 사건 각 부동산의 양도로 인한 소득이 원고에게 실질적으로 귀속되지 않았다는 취지의 이유로 양도소득세 204,317,749원을 전부 감액하여 달라는 내용의 경정청구”(이하 이 사건 경정청구라 한다)를 하였다. 피고는 포세무서장부가소득세과장에게 원고의 이 사건 경정청구 4) 2021. 12. 31. 청구 취지 피고가 원고에 대하여 한 2021년 귀속 양도소득세 204,317,749원에 관한 2022. 5. 26. 경정청구 처리결과 통지 처분을 취소한다. 이유 처분의 경위 등 1. 가. 이 사건 부동산 거래 및 사업자 등록 등 경위. 별지 목록 제1항 기재 토지 및 그 지상 같은 목록 제1항 기재 공장에 관하여 1) 2001. 5. 7. 매도인을 AA, BB로, 매수인을 원고로, 매매대금을 128,000,000원으로 하는 매매계약이 체결되었고 이에 관하여 원고 앞으로 소유권이전등기가 2001. 5. 15. 마쳐졌다. 별지 목록 제3, 4항 기재 토지에 관하여 매도인을 AA로, 매수인을 원고로, 매매대금을 12,000,000원으로 하는 매매계약이 2002. 11. 5. 체결되었고 이에 관하여 원고 앞으로 소유권이전등기가 2003. 6. 9. 마쳐졌다. 중 이 사건 사업장에 대한 명의위장 내용이 포함되어 있음을 이유로 이 사건 사업장에 대한 명의위장 혐의에 관한 현장확인을 의뢰하였고, 포세무서장은 피고 O 2022. 2. 24.에게 사업장 명의위장 현장확인 결과 “이 사건 사업장의 실사업자는 사업자등록 상 대표자인 원고가 아닌 박GG으로 판단하였음을 알려드립니다. 위 내용은 사업과 관련한 것으로 부동산 명의대여와는 관련 없음을 알려드립니다.”라는 내용으로 회신하였다. 피고는 2022. 4. 11.부터 2022. 4. 30.까지 원고에 대한 세무조사를 실시한 후 “① 원고는 이 사건 각 동산의 양도로 인한 소득이 박GG에게 귀속되었다는 구체적인 증빙자료를 제출하지 못하고 있는 점, ② 금융거래정보에 의하면 부동산담보 채무액 500,000,000원을 제외한 부동산 양도대금 550,000,000원이 원고 명의 우리은행 계좌 계좌번호는 내용으로 조사를 종결하였다.” 포세무서장은 원고에게 조세범처벌법 제11조 위반을 이유로 벌금 상당액 1,000만 원의 통고처분을 하였고, 원고는 이를 납부하였다. 피고는 원고에게 부동산 명의신탁에 대한 양도소득세 조사 결과 “동 부동산 양도로 인한 소득이 부 박GG에게 귀속되었다는 구체적인 증빙자료가 부족하여 경정청구 기각합니다”라는 이유로 이 사건 경정청구에 대한 거부처분(이하 이 사건 처분이라 한다)을 하였다. 원고는 이에 불복하여 조세심판원에 심판청구를 하였으나 조세심판원은 2022. 7. 15. 이를 기각하였다. 2023. 3. 20. 인정 근거 다툼 없는 사실 [갑 제2, 4, 6, 8, 15, 18, 21, 23, 25, 30호증을 제호증가지번호가 있는 것은 가지번호 포함의 각 기재].\""},
                {"role": "user", "content": user_message}
            ]
        )
        # 응답에서 메시지 추출
        gpt_response = response.choices[0].message.content
        return jsonify({"response": gpt_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)


from dotenv import load_dotenv
load_dotenv()  # .env 파일에서 변수 로드

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import  OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma

# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("data/국세법령정보시스템.pdf")
docs = loader.load()

# 문서의 전체 페이지 수, 특정 페이지의 내용 확인용 출력
#print(f"문서의 페이지수: {len(docs)}")

#print(docs[2].page_content)

def split_with_page_numbers(docs, chunk_size=500, chunk_overlap=50):
    split_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for doc in docs:
        page_content = doc.page_content
        page_number = doc.metadata['page']  # 페이지 넘버 가져오기
        
        # 페이지 내용을 청크 단위로 분할
        split_page_content = text_splitter.split_text(page_content)
        
        # 각 청크에 페이지 번호를 추가하고 메타데이터 생성
        for chunk in split_page_content:
            split_documents.append({
                "content": chunk.strip(),
                "metadata": {"page": page_number}
            })
    
    return split_documents


# 페이지 넘버 포함하여 문서 분할
split_documents_with_page_numbers = split_with_page_numbers(docs, chunk_size=500, chunk_overlap=50)

# 분할된 청크의 수
#print(f"분할된 청크의 수: {len(split_documents_with_page_numbers)}")

# 분할된 전체 청크
#for idx, chunk in enumerate(split_documents_with_page_numbers):
#   print(f"청크 {idx+1}: {chunk}\n")



from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import  OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma

# 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings(openai_api_key=api_key)


contents = [doc["content"] for doc in split_documents_with_page_numbers]
metadatas = [doc["metadata"] for doc in split_documents_with_page_numbers]


# 단계 4: 문서 임베딩
embeddings_list = embeddings.embed_documents(contents[0])

# 임베딩이 잘 되었는지 첫번째 청크의 벡터값 확인용 출력
#print("첫 번째 청크 벡터:", embeddings_list)


# Chroma로 데이터 저장 및 로드
vectorstore = Chroma.from_texts(texts=contents, embedding=embeddings, metadatas=metadatas)


'''
# 유사도검색1
docs1 = vectorstore.similarity_search("양도소득세")
for doc in docs1:
    print(f"페이지: {doc.metadata['page']}")
    print(doc.page_content)
    print("============================================================")
'''


# 유사도검색2
retriever = vectorstore.as_retriever()
docs2 = retriever.invoke("피고가 2022. 5. 26. 2021 원고에 대하여 한 년 귀속 양도소득세 204,317,749원에 관한 경정청구 처리한 사건에서 어떤 판결이 났었지?")

for doc in docs2:
    print(f"페이지: {doc.metadata['page']}")
    print(doc.page_content)
    print("------------------------------------------------------------")