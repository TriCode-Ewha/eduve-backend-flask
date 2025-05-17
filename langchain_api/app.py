import fitz # PuMuPDF
import traceback
from flask import Flask, jsonify, request
import requests
import openai
import os
import uuid
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from flask_cors import CORS
from pdf2image import convert_from_path
from docx2pdf import convert as convert_docx_pdf
import tempfile
import base64

from docx import Document
from PIL import Image
import pytesseract
import time

load_dotenv()  # .env 파일에서 환경 변수 로드

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


# OpenAI API 키 설정
api_key = openai.api_key = ""

# Clova OCR API 키 설정
clova_api_url = ""
clova_api_secret = ""

#load_dotenv()  # .env 파일에서 환경 변수 로드


embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
# ChromaDB 벡터 데이터베이스 로드 (디스크에 저장)
# 디스크에 저장해놔야 이전에 저장한 데이터가 유지됨 -> API 호출할 때마다 데이터베이스가 초기화되서 이전에 저장한 데이터 검색 불가능
# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings, collection_name="eduve")

######################### 수정한 부분 ########################
# 사용자 userId별로 vectorstore 반환하는 메서드 -> 저장할때 사용자userId에 해당하는 collection에 저장
def get_vectorstore(user_id):
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        collection_name=f"eduve_user_{user_id}"
    )


# 벡터DB 초기화 - 사용자별
@app.route('/delete_all', methods=['DELETE'])
def delete_all_data():
    try:
        # 삭제할 userId
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({"error": "Missing user_id parameter"}), 400
        
        vectorstore = get_vectorstore(user_id)
        ids = vectorstore.get()['ids']

        # 가져온 모든 ids 삭제
        if ids:
            vectorstore.delete(ids=ids)

        return jsonify({"message": f"Deleted {len(ids)} documents from user {user_id}'s collection."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500




# 클로바 OCR
def call_clova_ocr(image_bytes):
    # 클로바 OCR API 호출
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    headers = {
        "X-OCR-SECRET": clova_api_secret,
        "Content-Type": "application/json"
    }

    payload = {
        "images": [{"format": "jpg", "name": "sample", "data": image_base64}],
        "version": "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000)
    }

    response = requests.post(clova_api_url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    # 클로바 OCR API 결과에서 텍스트만 추출
    texts = []
    for field in result.get('images', [{}])[0].get('fields', []):
        texts.append(field.get('inferText', ''))

        extracted_text = '\n'.join(texts)
    
    # 여기서 텍스트 출력해보기
    print("=== OCR 추출된 텍스트 ===")
    print(extracted_text)
    print("=======================")
    
    return extracted_text




# PDF 파일을 받아 임베딩하여 저장하는 API
@app.route('/embedding', methods=['POST', 'OPTIONS'])
def embedding():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        # userId 없으면 400 ERROR
        user_id = request.form.get("userId")
        if not user_id:
            return jsonify({"error": "userId is required"}), 400
        
        # 파일이 없으면 400 ERROR
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        # 3. 새로 추가: 스프링부트에서 넘겨준 파일명(title) 받아오기
        title = request.form.get("title", "unknown_filename")



        ######################### 수정한 부분 ########################
        # userId로 collectionname생성
        vectorstore = get_vectorstore(user_id)




        # 고유한 파일명을 생성(uuid.uuid4().hex)하여 data/디렉토리에 저장
        file = request.files['file']
        # 파일 확장자명 추출
        file_ext = file.filename.split('.')[-1].lower()
        #filename = f"temp_{uuid.uuid4().hex}.pdf"
        #filepath = os.path.join("data", filename)


        UPLOAD_DIR = "data"
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        filename = f"temp_{uuid.uuid4().hex}.{file_ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        '''
        processed_path = None
        text = None

        
        # 파일 변환 로직
        if file_ext == 'pdf': # pdf 파일이면 그대로 진행행
            processed_path = filepath
        elif file_ext == 'docx': # docx
            processed_path = filepath.replace('.docx', '.pdf')
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            convert_docx_to_pdf(filepath, processed_path)
        elif file_ext in ['jpg', 'jpeg', 'png']: # 이미지
            text = extract_text_from_image(filepath)
        elif file_ext == 'txt': # txt
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()  # 텍스트 파일 내용 읽기
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        

        # PDF 로드 및 분할
        docs = []
        if processed_path:
            loader = PyMuPDFLoader(processed_path)
            docs = loader.load()
            os.remove(processed_path)  # 변환된 PDF 삭제
        elif text is not None:
            docs = [{"page_content": text, "metadata": {"page": 1}}]  # OCR 결과 저장

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_documents = []

        for doc in docs:
            page_content = doc.page_content
            page_number = doc.metadata['page']  #페이지 넘버
            # 페이지 내용을 청크 단위로 분할
            split_page_content = text_splitter.split_text(page_content)
            # 각 청크에 페이지 번호를 추가하고 메타데이터 생성
            for chunk in split_page_content:
                split_documents.append({
                    "content": chunk.strip(),
                    "metadata": {"page": page_number}
                })
        '''

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_documents = []

        # DOCX는 PDF로 변환
        if file_ext == 'docx':
            processed_path = filepath.replace('.docx', '.pdf')
            convert_docx_pdf(filepath, processed_path)
            ocr_results = extract_ocr_texts_by_page(processed_path)
            os.remove(processed_path)

        elif file_ext == 'pdf':
            ocr_results = extract_ocr_texts_by_page(filepath)

        elif file_ext in ['jpg', 'jpeg', 'png']:
            text = extract_text_from_image(filepath)
            ocr_results = [{"page": 1, "text": text}]

        elif file_ext == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                ocr_results = [{"page": 1, "text": text}]

        else:
            return jsonify({"error": "Unsupported file format"}), 400

        for result in ocr_results:
            page_number = result["page"]
            page_text = result["text"]
            chunks = text_splitter.split_text(page_text)
            for chunk in chunks:
                split_documents.append({
                    "content": chunk.strip(),
                    "metadata": {"page": page_number},
                    "filename": title  # 여기서 파일명 메타데이터 추가
                })


        # 페이지 넘버 포함하여 문서 분할
        contents = [doc["content"] for doc in split_documents]
        metadatas = [doc["metadata"] for doc in split_documents]
        ids = [str(uuid.uuid4()) for _ in contents]

        # 문서 임베딩 및 저장
        vectorstore.add_texts(texts=contents, metadatas=metadatas, ids=ids)
        vectorstore.persist()  # 데이터 저장 유지

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({"message": "Document successfully embedded with OCR"})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

1


# 유사도 검색 API
# '{"query": "금융 상품"}' 같이 입력
''' 데이터 요청 이런식으로..
{
    "query": "금융 상품 추천"
}
'''
@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get("query", "")
        user_id = data.get("userId", "")
        teacher_id = data.get("teacherId", "")

        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        if  not user_id or not teacher_id:
            return jsonify({"error": "userId, and teacherId are required"}), 400
        
        # 사용자 userId collection에서 검색
        user_vectorstore = get_vectorstore(user_id)
        user_results = user_vectorstore.similarity_search_with_score(query, k=5)


        # 선생님 벡터스토어 검색
        teacher_vectorstore = get_vectorstore(teacher_id)
        teacher_results = teacher_vectorstore.similarity_search_with_score(query, k=5)

        # 결과 병합 및 점수 기준으로 정렬
        combined_results = user_results + teacher_results
        combined_results.sort(key=lambda x: x[1])

        # 상위 5개만 추출
        top_results = combined_results[:5]

        # 튜플 분해하여 결과 구성
        results = [
            {
                "file_name": doc.metadata["file_name"],
                "page": doc.metadata["page"],
                "content": doc.page_content,
                "score": score
            }
            for doc, score in top_results
        ]

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# chat_gpt 호츌 API
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



from sentence_transformers import SentenceTransformer
import spacy
from sentence_transformers.util import pytorch_cos_sim

# 임베딩 모델 로드 (사용할 모델 변경 가능)
model = SentenceTransformer('all-MiniLM-L6-v2')


@app.route('/extractTopic', methods=['POST'])
def extract_topic():
    data = request.get_json()
    message = data.get("message", "")

    # 토픽 추출 로직 (간단한 키워드 기반 예제)
    topic = extract_main_topic(message)

    return jsonify({"topic": topic})



@app.route('/calculateSimilarity', methods=['POST'])
def calculate_similarity():
    data = request.get_json()
    topic1 = data.get("topic1", "")
    topic2 = data.get("topic2", "")

    # 임베딩 벡터 변환
    embedding1 = model.encode(topic1, convert_to_tensor=True)
    embedding2 = model.encode(topic2, convert_to_tensor=True)

    # 코사인 유사도 계산
    similarity_score = pytorch_cos_sim(embedding1, embedding2).item()

    return jsonify({"similarity": similarity_score})



def extract_main_topic(text):
    """
    간단한 토픽 추출 (명사 기반, 필요하면 더 고도화 가능)
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")  # 영어 모델 (한국어 사용시 ko_core_news_sm 사용)
    doc = nlp(text)

    # 명사만 추출하여 대표 키워드 선정
    nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]

    return nouns[0] if nouns else text  # 명사가 없으면 원문 반환





# docx -> pdf 변환
def convert_docx_to_pdf(docx_path, pdf_path):
    doc = Document(docx_path)
    doc.save(pdf_path)


# 이미지에서 텍스트 추출
def extract_text_from_image(image_path):
    '''
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang="eng+kor")  # OCR 수행
    return text
    '''
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    ocr_text = call_clova_ocr(image_bytes)
    return ocr_text


# pdf 이미지 처리
def extract_ocr_texts_by_page(pdf_path):
    images = convert_from_path(pdf_path)
    page_texts = []
    for i, image in enumerate(images, start=1):
        # 이미지 메모리에 저장
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, "JPEG")
            with open(tmp.name, "rb") as img_file:
                ocr_text = call_clova_ocr(img_file.read()) or ""
                page_texts.append({"page": i, "text": ocr_text})
        os.remove(tmp.name)
    return page_texts      

if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)  # PDF 저장할 디렉토리 생성
    os.makedirs("chroma_db", exist_ok=True)  # Chroma DB 저장할 디렉토리 생성
    app.run(host='0.0.0.0', port=5000)




'''

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


# 유사도검색1
docs1 = vectorstore.similarity_search("양도소득세")
for doc in docs1:
    print(f"페이지: {doc.metadata['page']}")
    print(doc.page_content)
    print("============================================================")


# 유사도검색2
retriever = vectorstore.as_retriever()
docs2 = retriever.invoke("피고가 2022. 5. 26. 2021 원고에 대하여 한 년 귀속 양도소득세 204,317,749원에 관한 경정청구 처리한 사건에서 어떤 판결이 났었지?")

for doc in docs2:
    print(f"페이지: {doc.metadata['page']}")
    print(doc.page_content)
    print("------------------------------------------------------------")

'''