
# Eduve: RAG 기반 AI 챗봇 서비스

Eduve는 **RAG(Retrieval-Augmented Generation) 기반의 AI 챗봇 학습 지원 서비스**로, 학생과 교사의 문서 기반 질문 응답을 지원합니다. 음성(STT), OCR, 채팅 저장 기능을 포함하며, Spring Boot와 Flask로 백엔드를 구성하고 React 기반 웹 인터페이스를 제공합니다. 학습 환경에서의 커뮤니케이션과 정보 접근성을 향상시키는 것을 목표로 합니다.

<br>

## 📦 프로젝트 구성 레포지토리

본 프로젝트는 다음 세 개의 레포지토리로 구성되어 있습니다:

| 이름         | 설명                           | GitHub 주소 |
|--------------|--------------------------------|--------------|
| **springboot** | 백엔드 주요 로직 (API, DB, STT, JWT 등)  | [Eduve Spring Boot](https://github.com/TriCode-Ewha/eduve-backend-springboot) |
| **flask**     | AI 모델 기반 RAG  서버 (LangChain, 임베딩, 유사도 검색)     | [Eduve Flask AI](https://github.com/TriCode-Ewha/eduve-backend-flask) |
| **front**     | 프론트엔드 클라이언트 (웹 인터페이스, React) | [Eduve Front](https://github.com/TriCode-Ewha/eduve-frontend) |


<br>
<br>
<br>
<br>


# Flask (eduve-backend-flask)

Flask는 LangChain 기반의 AI 추론 서버로, 유저의 질문에 대해 문서 검색(RAG)과 LLM 생성을 통해 응답을 생성합니다. Spring Boot 서버와 연동되어 동작합니다.

<br>
<br>

## 📁 주요 디렉토리 구조

```
eduve-backend-flask/
├── langchain_api/
│     ├── app.py                         # Flask 진입점
│     ├── requirements.txt               # pip 설치용 의존성 리스트
│     │
│     └── test/
│          ├── test_flask.py                  # 질문 API 테스트
│          ├── test_search_accuracy.py        # 검색 정확도 테스트
│          ├── tests.json                     # 질문-응답 테스트셋
│          ├── all_cases.csv                  # 전체 테스트 기록
│          ├── wrong_cases.csv                # 실패 케이스 기록
│
├── .env_sample                      # 환경 변수 예시
├── pyproject.toml                   # Poetry 프로젝트 정의
├── poetry.lock                      # 정확한 의존성 버전
└── README.md
```

<br>
<br>
<br>


## 🧾 Source Code 설명
AI 검색 및 생성 파이프라인은 `langchain_api/` 디렉토리에 구성되어 있으며, Flask 서버 실행부터 문서 임베딩/검색/생성까지 포함됩니다.

| 경로 또는 파일                              | 설명 |
|-------------------------------------------|------|
| `langchain_api/app.py`                    | Flask 진입점. API 라우트 정의 (`/search`, `/embedding` 등) |
| `langchain_api/requirements.txt`          | REST API 실행에 필요한 의존성 목록 |
| `langchain_api/test/test_flask.py`        | Flask API 엔드포인트 단위 테스트 코드 |
| `langchain_api/test/test_search_accuracy.py` | Top‑1 / Top‑5 정밀 평가 CLI 스크립트 |
| `langchain_api/test/tests.json`           | 질의-정답 페어 데이터셋 (검색 평가용) |
| `langchain_api/test/all_cases.csv`        | 전체 평가 결과 저장 파일 (자동 생성) |
| `langchain_api/test/wrong_cases.csv`      | 정답과 일치하지 않는 오답 케이스 저장 파일 |
| `.env_sample`                             | API 키, 임베딩 모델 등 환경 변수 예시 설정 |


<br>
<br>
<br>


## 📦 How to Install

#### 1. 환경 요구사항

- Python 3.10 이상

#### 2. 설치 절차

```bash
# 1. Git 저장소 클론
git clone https://github.com/TriCode-Ewha/eduve-backend-flask.git
cd eduve-backend-flask

# 2. 환경 설정 파일 준비
cp .env_sample .env
# .env 파일에 OPENAI_API_KEY 등 환경변수 입력

```
<br>
<br>
<br>


## 🛠 How to Build

프로젝트를 설치 한 후 pip 사용하여 환경을 준비합니다.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

<br>
<br>
<br>


## 🚀 How to Run

빌드 후 다음 명령으로 실행할 수 있습니다:

```bash
python langchain_api/app.py
```
- 기본 포트는 http://localhost:5000 입니다.



<br>
<br>
<br>


## ✅ How to Test
### 1. RAG 기반 유사도 검색 성능

본 프로젝트에서는 RAG 기반 검색 성능을 평가하기 위해 별도의 CLI 스크립트를 제공합니다.  
정확도는 다음 두 항목으로 나누어 계산됩니다:

- **Top‑1 파일 정확도**: 가장 첫 번째 결과의 `file_name`이 정답과 일치
- **Top‑5 페이지 정확도**: 상위 5개 결과 중 `file_name`과 `page`가 모두 일치하는 항목 존재


#### 테스트 실행
```bash
python langchain_api/test/test_search_accuracy.py \
  --data langchain_api/test/tests.json \
  --user student01 \
  --teacher teacher01 \
  --host http://localhost:5000
```

#### 테스트 구조
| 디렉토디 경로                           | 설명                                 |
| --------------------------------------- | ---------------------------------- |
| `langchain_api/test/test_search_accuracy.py` | 정밀 평가용 CLI 스크립트 (Top‑1, Top‑5 정확도) |
| `langchain_api/test/test_flask.py`           | 기본 Flask 엔드포인트 테스트 코드              |


#### 출력 예시
```bash
파일 Top‑1 정확도 : 73.12%
페이지 Top‑5 정확도 : 86.93%
Wrong cases saved → wrong_cases.csv
All cases saved → all_cases.csv
```

#### 결과 파일
| 파일명               | 설명                                |
| ----------------- | --------------------------------- |
| `all_cases.csv`   | 전체 테스트 결과 (예측 결과, 정답, 일치 여부 등 포함) |
| `wrong_cases.csv` | 정답과 불일치한 케이스만 따로 저장한 파일           |

<br>

### 2. 텍스트 추출 정확도
본 프로젝트의 /embedding API는 PDF, 이미지, DOCX 등 다양한 문서로부터 텍스트를 추출하고 임베딩합니다. 
추출된 텍스트는 콘솔에 출력되어 확인 가능합니다.


#### 테스트 실행
```bash
curl -X POST http://localhost:5000/embedding \
  -F "userId=test_user" \
  -F "file=@./sample_image.jpg" \
  -F "title=cyber_safety_guide.pdf"
```
- file에는 테스트할 이미지, PDF, docx 파일을 넣고, title은 실제 저장될 문서명처럼 넣습니다.


#### 콘솔 출력 예시
API 서버 콘솔에서는 다음과 같이 OCR 결과가 출력됩니다:

```markdown
=== OCR 추출된 텍스트 ===
사이버 폭력은 다음 경로로 신고할 수 있습니다....
...
=======================
```
- 콘솔에 출력된 텍스트를 통해 OCR이 문서를 얼마나 정확하게 인식했는지 직접 확인할 수 있습니다.


<br>
<br>
<br>

## 📊 샘플 데이터 설명
프로젝트에는 AI 검색 정확도 평가와 텍스트 임베딩 테스트를 위한 샘플 데이터가 포함되어 있습니다.


#### 1. 검색 평가 질문 리스트 (tests.json)
  - 위치: `langchain_api/test/tests.json`
  - 형식: JSON

    ```json
    [
      {
        "query": "학교폭력에 대한 법적 절차는 어떻게 되나요?",
        "answer_filename": "school_policy.pdf",
        "answer_page": 12
      },
      {
        "query": "사이버 폭력 신고 방법 알려줘",
        "answer_filename": "cyber_safety_guide.pdf",
        "answer_page": 5
      }
    ]
    ```

  - 각 항목은 실제 질문(query)과 해당 질문의 정답이 존재하는 PDF 문서 파일명(answer_filename), 정답 페이지(answer_page)를 포함합니다.



#### 2. 전체 테스트 결과 (all_cases.csv)
- 생성 위치: ./all_cases.csv (테스트 실행 후 자동 생성)
- 형식: CSV

    ```csv
    query,answer_filename,answer_page,pred_file_top1,pred_page_top1,score_top1,correct_file,correct_page
    "학교폭력에 대한 법적 절차는 어떻게 되나요?","school_policy.pdf",12,"school_policy.pdf",12,0.9273,TRUE,TRUE
    "사이버 폭력 신고 방법 알려줘","cyber_safety_guide.pdf",5,"other_file.pdf",7,0.8123,FALSE,FALSE
    ```
- 예측 결과와 정답 비교 결과(correct_file, correct_page)가 포함되어 성능 분석이 가능합니다.

#### 3. 오답 사례 (wrong_cases.csv)
- 생성 위치: ./wrong_cases.csv
- 형식: CSV
  ```csv
  query,gt_file,gt_page,top5
  "사이버 폭력 신고 방법 알려줘","cyber_safety_guide.pdf",5,"[('other_file.pdf', 7, 0.81), ...]"
  ```
- Top‑5 안에 정답 문서+페이지 조합이 없었던 질문들만 별도로 저장됩니다.
- 성능 개선을 위한 오답 분석에 유용합니다.

#### 4. 임베딩용 예시 문서 파일
- 위치: langchain_api/test/resources/ 또는 data/ 디렉토리 내
- 사용 목적: 테스트용 질문과 매칭되는 실제 PDF 또는 이미지 파일 제공

- 예시 파일:
  - school_policy.pdf
  - cyber_safety_guide.pdf
  - sample_image.jpg

- 이 파일들은 /embedding API를 통해 벡터화되어 검색 테스트에 사용됩니다.

<br>
<br>
<br>


## 🔐 환경 설정 (.env)
.env_sample 파일을 복사해 .env 파일을 생성하고 아래 항목을 설정하세요:
```
OPENAI_API_KEY=your-api-key
VECTOR_DB=chroma
USE_LOCAL_EMBEDDING=true
```


<br>
<br>
<br>

## 📚 사용된 오픈소스 목록
| 라이브러리                     | 설명                                                                 |
|-------------------------------|----------------------------------------------------------------------|
| **Flask**                     | 경량 웹 프레임워크 (REST API 구현)                                   |
| **LangChain**                 | RAG 파이프라인 구성 (임베딩, 검색, 생성 통합)                         |
| **Chroma (chromadb)**         | 로컬 기반 오픈소스 벡터 데이터베이스                                  | 
| **sentence-transformers**     | 문서 임베딩 생성 (MiniLM, BERT 기반 모델)                             |
| **PyMuPDF (fitz)**            | PDF 문서 텍스트 및 메타데이터 추출                                    | 
| **python-docx**               | DOCX 파일 처리 (PDF 변환 등 전처리용)                                 |
| **pdf2image**                 | PDF 페이지를 이미지로 변환 (OCR 전처리용)                             | 
| **requests**                  | 외부 API 및 OCR 호출용 HTTP 클라이언트                                | 
| **python-dotenv**             | .env 파일을 통한 환경변수 로딩                                       |
| **spacy**                     | NLP 처리 및 명사 기반 토픽 추출                                       |
| **torch, torchvision**        | PyTorch 기반 모델 실행 및 임베딩 처리                                 |
| **numpy, pandas**             | 수치 계산 및 테스트 결과 CSV 저장                                    |
| **uuid, base64, tempfile**    | Python 표준 라이브러리 (문서 식별, 파일 처리, 인코딩 등)             |
| **OpenAI API**                | GPT 기반 생성 모델 연동 (LangChain 연계 시 사용)                     | 
| **Clova OCR API**             | 이미지 기반 텍스트 인식 (OCR). 클로바 OCR 사용                        |




