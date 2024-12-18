'''
from flask import Flask, jsonify, request
import requests
import openai

app = Flask(__name__)
'''

# OpenAI API 키 설정
# openai.api_key = "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"  # 여기에 발급받은 API 키를 입력하세요
api_key = "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"  
# api_key = "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"  

'''
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
    app.run(debug=True, port=5000)
'''



from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 변수 로드

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import  OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("data/국세법령정보시스템.pdf")
#loader = PyMuPDFLoader("data/투자론 과제_2276267 장예주.pdf")
docs = loader.load()

# 문서의 전체 페이지 수, 특정 페이지의 내용 확인용 출력
print(f"문서의 페이지수: {len(docs)}")
#print(docs[2].page_content)

""" # 페이지마다 청크를 만들고, 각 청크 앞에 해당하는 페이지 넘버를 추가함
def split_with_page_numbers(docs, chunk_size=500, chunk_overlap=50):
    split_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 문서 페이지마다 분할
    for doc in docs:
        page_content = doc.page_content
        page_number = doc.metadata['page']  # 페이지 넘버 가져오기
        
        # 페이지 내용을 청크 단위로 분할
        split_page_content = text_splitter.split_text(page_content)
        
        # 각 청크에 해당하는 페이지 넘버를 앞에 추가하여 저장
        for chunk in split_page_content:
            chunk_with_page_number = f"Page {page_number}: {chunk.strip()}"
            split_documents.append(chunk_with_page_number)
    
    return split_documents """

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
print(f"분할된 청크의 수: {len(split_documents_with_page_numbers)}")


'''
for idx, chunk in enumerate(split_documents_with_page_numbers):
    print(f"청크 {idx+1}: {chunk}\n")
'''


# 단계 3: 임베딩(Embedding) 생성
#embeddings = OpenAIEmbeddings()
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

'''
# 단계 4: 전체 문서 임베딩
embeddings_list = embeddings.embed_documents(split_documents_with_page_numbers)

# 임베딩이 잘 되었는지 첫번째 청크의 벡터값 확인용 출력
print("첫 번째 청크 벡터:", embeddings_list[0])
'''

contents = [doc["content"] for doc in split_documents_with_page_numbers]
metadatas = [doc["metadata"] for doc in split_documents_with_page_numbers]


""" # 단계 5: 유사도 검색 및 메타데이터 출력
query = "양도소득세"
results = vectorstore.similarity_search(query, k=3)

for result in results:
    print(f"내용: {result.page_content}")
    print(f"페이지: {result.metadata['page']}")
    print() """


# Chroma로 데이터 저장 및 로드
vectorstore = Chroma.from_texts(texts=contents, embedding=embeddings, metadatas=metadatas)

'''
# 모든 데이터 출력
for doc in vectorstore.similarity_search("", k=len(contents)):
    print(f"내용: {doc.page_content}")
    print(f"메타데이터: {doc.metadata}")
    print()
'''

# 유사도검색1
docs1 = vectorstore.similarity_search("양도소득세", k=2)
for doc in docs1:
    print(f"페이지: {doc.metadata['page']}")
    print(doc.page_content)
    print("============================================================")

# 유사도검색2
retriever = vectorstore.as_retriever()
docs2 = retriever.invoke("양도소득세")

for doc in docs2:
    print(f"페이지: {doc.metadata['page']}")
    print(doc.page_content)
    print("------------------------------------------------------------")















""" # 벡터데이터베이스 기반 유사도검색 (혜진)
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

doc = ""

# db 생성
db = Chroma.from_documents(
    documents=doc,
    embedding = OpenAIEmbeddings(),
    collection_name="eduve_db"
)

# 1
# 쿼리 텍스트와 가장 유사한 문서들의 리스트 반환
# k값에 검색 document 결과의 개수를 지정가능
db.similarity_search("TF IDF 에 대하여 알려줘", k=2)
# filter 사용 : 다른 파일에서 데이터 검색 가능
db.similarity_search(
    "TF IDF 에 대하여 알려줘", filter={"source": "data/nlp-keywords.txt"}, k=2
)
# filter 사용
db.similarity_search(
    "TF IDF 에 대하여 알려줘", filter={"source": "data/finance-keywords.txt"}, k=2
)



# 2
# 벡터 저장소를 검색기(Retriever)로 변환
retriever = db.as_retriever()
retriever.invoke("Word2Vec 에 대하여 알려줘")

# k: 반환할 문서 수
# fetch_k : MMR 알고리즘에 전달할 문서 수(기본 값: 20)
# lambda_mult: MMR 결과의 다양성 조절(0~1, 기본값: 0.5)
retriever = db.as_retriever(
    search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25, "fetch_k": 10}
)
retriever.invoke("Word2Vec 에 대하여 알려줘")
# 필터 적용
retriever = db.as_retriever(
    search_kwargs={"filter": {"source": "data/finance-keywords.txt"}, "k": 2}
)
retriever.invoke("ESG 에 대하여 알려줘") """

