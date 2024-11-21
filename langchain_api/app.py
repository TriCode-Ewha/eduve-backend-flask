""" from flask import Flask, jsonify, request
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
 """

# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("langchain_api")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("data/국세법령정보시스템.pdf")
docs = loader.load()

# 문서의 전체 페이지 수, 특정 페이지의 내용 확인용 출력
print(f"문서의 페이지수: {len(docs)}")
#print(docs[2].page_content)

# 페이지마다 청크를 만들고, 각 청크 앞에 페이지 넘버를 추가하는 방식
def split_with_page_numbers(docs, chunk_size=500, chunk_overlap=50):
    split_documents = []
    current_chunk = ""
    current_page = 1
    for doc in docs:
        page_content = doc.page_content
        page_number = doc.metadata['page']  # 페이지 넘버 가져오기
        content_with_page_number = f"Page {page_number}: {page_content}"  # 페이지 넘버 추가
        
        # content_with_page_number가 청크 사이즈를 초과하는지 확인하고 분할
        if len(current_chunk) + len(content_with_page_number) > chunk_size:
            # 기존 청크에 더 이상 페이지를 추가할 수 없으면, 청크를 분리
            split_documents.append(current_chunk.strip())
            current_chunk = content_with_page_number
        else:
            # 청크에 페이지 추가
            current_chunk += content_with_page_number
        
        # 페이지가 끝날 때마다 청크를 나누는 처리를 할 수 있게 분할
        if len(current_chunk) > chunk_size: 
            split_documents.append(current_chunk.strip())
            current_chunk = ""
    
    if current_chunk.strip():  # 마지막 남은 청크를 추가
        split_documents.append(current_chunk.strip())
    
    return split_documents

# 문서 분할 (페이지 넘버 포함)
split_documents_with_page_numbers = split_with_page_numbers(docs, chunk_size=500, chunk_overlap=50)

# 분할된 청크의 수와 첫 번째 청크의 내용 확인
print(f"분할된 청크의 수: {len(split_documents_with_page_numbers)}")
print(f"첫 번째 청크: {split_documents_with_page_numbers[1]}")

# 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings()

# 단계 4: 전체 문서 임베딩
embeddings_list = embeddings.embed_documents(split_documents_with_page_numbers)

# 임베딩이 잘 되었는지 첫번째 청크의 벡터값 확인용 출력
print("첫 번째 청크 벡터:", embeddings_list[1])