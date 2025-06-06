
# NASDAQ RAG Chatbot 🤖💸

NASDAQ 관련 데이터를 위한 RAG(Retrieval-Augmented Generation) 챗봇입니다. Google Drive에서 CSV 파일을 자동으로 수집하고, 텍스트를 벡터화하여 저장한 후, 사용자 질문에 대해 의미 기반 또는 키워드 기반 검색을 통해 답변을 제공합니다. Streamlit으로 구현된 사용자 친화적인 웹 인터페이스를 제공합니다.

## 주요 기능

-**Google Drive 연동 📁**: Google Drive API를 사용하여 CSV 파일을 자동으로 수집합니다.

-**다중 텍스트 분할기 지원 ✂️**: SpacyTextSplitter, NLTKTextSplitter, RecursiveCharacterTextSplitter를 순차적으로 시도하여 최적의 텍스트 청킹을 수행합니다.

-**듀얼 검색 모드 🔍**: 의미 기반 검색(RAG)과 키워드 검색을 모두 지원합니다.

-**한국어 임베딩 모델 🇰🇷**: jhgan/ko-sbert-nli 모델을 사용하여 한국어 텍스트를 효과적으로 처리합니다.

-**사용자 피드백 시스템 👍👎**: 답변의 품질에 대한 사용자 피드백을 수집하고 저장합니다.

-**Streamlit UI**: 직관적이고 사용하기 쉬운 웹 인터페이스를 제공합니다.

-**파일 관리 시스템**: 연도/월별로 구조화된 CSV 파일 다운로드 기능을 제공합니다.

## 기술 스택

-**프론트엔드**: Streamlit

-**백엔드**: Python, LangChain

-**임베딩**: HuggingFace Embeddings (jhgan/ko-sbert-nli)

-**벡터 저장소**: ChromaDB

-**LLM**: Ollama (EEVE-Korean-10.8B)

-**클라우드 저장소**: Google Drive API

-**데이터 처리**: Pandas, scikit-learn

## 설치 및 실행 방법

### 1. 필수 라이브러리 설치

프로젝트 디렉토리에서 다음 명령어를 실행하여 필요한 Python 라이브러리를 설치합니다:

```bash

pipinstallstreamlitpandaspython-dotenvgoogle-authgoogle-auth-oauthlibgoogle-auth-httplib2google-api-python-clientlangchainlangchain-text-splitterslangchain-huggingfacelangchain-communitylangchain-ollamalangchain-chromachromadbscikit-learnnltkspacy

```

### 2. 환경 변수 설정

`.env` 파일을 생성하여 Google Drive API를 위한 `FOLDER_ID`를 설정합니다:

```

FOLDER_ID=your_google_drive_folder_id

```

### 3. Google API 인증 설정

Google Cloud Console에서 서비스 계정을 생성하고 `credentials.json` 파일을 프로젝트 루트 디렉토리에 저장합니다.

### 4. Ollama 모델 설치

EEVE-Korean-10.8B 모델을 설치합니다:

```bash

ollamapullEEVE-Korean-10.8B:latest

```

### 5. 실행

다음 명령어를 실행하여 Streamlit 애플리케이션을 시작합니다:

```bash

streamlitrunRAG.py

```

## 파일 구조

```

nasdaq_vec/

├── credentials.json                     # Google API 인증 파일

├── merged_data.csv                      # 병합된 CSV 데이터

├── RAG.py                              # 메인 Python 애플리케이션

├── README.md                           # 프로젝트 설명 파일

├── .env                               # 환경 변수 파일

├── chroma_db/                         # 벡터 데이터베이스 저장소

├── feedback/                          # 사용자 피드백 저장 디렉토리

│   └── feedback_log.csv              # 피드백 로그 파일

└── .gitignore                        # Git 무시 파일 설정

```

## 사용 방법

1.**데이터 수집**: Streamlit 애플리케이션을 실행하면, 자동으로 Google Drive에서 CSV 파일을 수집하고 병합합니다.

2.**벡터화**: 병합된 데이터를 텍스트 청킹하여 벡터 데이터베이스에 저장합니다.

3.**검색 모드 선택**: 의미 기반 검색 또는 키워드 검색 중 하나를 선택합니다.

4.**질문 입력**: 검색어를 입력하고 검색을 실행합니다.

5.**결과 확인**: 검색 결과와 관련 문맥을 확인하고 피드백을 제공할 수 있습니다.

## 주요 함수 설명

### 데이터 처리 함수

- [`authenticate_drive()`](RAG.py): Google Drive API 인증
- [`merge_csv()`](RAG.py): 여러 CSV 파일을 하나로 병합
- [`load_and_split_csv()`](RAG.py): CSV 파일을 로드하고 텍스트를 청킹

### 검색 및 RAG 함수

- [`create_vectorstore()`](RAG.py): 벡터 저장소 생성
- [`run_rag_chain()`](RAG.py): RAG 체인 실행 및 답변 생성
- [`keyword_search()`](RAG.py): 키워드 기반 문서 검색
- [`evaluate_with_embedding()`](RAG.py): 임베딩 유사도 기반 답변 평가

### 사용자 인터페이스 함수

- [`save_feedback()`](RAG.py): 사용자 피드백 저장
- [`main()`](RAG.py): Streamlit 메인 애플리케이션

## 특징

### 지능형 텍스트 분할

-**1차**: SpacyTextSplitter (한국어 모델 ko_core_news_sm 사용)

-**2차**: NLTKTextSplitter (문장 기반 분할)

-**3차**: RecursiveCharacterTextSplitter (기본 분할기)

### 듀얼 검색 시스템

-**의미 기반 검색**: RAG 체인을 통한 컨텍스트 기반 답변 생성

-**키워드 검색**: 직접적인 키워드 매칭을 통한 빠른 검색

### 사용자 경험 최적화

- 진행률 표시바와 상태 메시지
- 연도/월별 파일 구조화된 사이드바
- 실시간 피드백 수집
- 임베딩 유사도 점수 표시

## 주의사항

-`credentials.json` 파일이 필요합니다. Google Cloud Console에서 서비스 계정을 생성하여 다운로드하세요.

- Google Drive의 폴더 ID를 `.env` 파일에 정확히 설정해야 합니다.
- CUDA가 사용 가능한 환경에서 더 빠른 성능을 얻을 수 있습니다.
- 한국어 처리를 위해 spaCy 한국어 모델이 필요할 수 있습니다: `python -m spacy download ko_core_news_sm`

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 문제 해결

### 일반적인 오류

1.**Google API 인증 오류**: `credentials.json` 파일 경로와 권한을 확인하세요.

2.**Ollama 모델 오류**: EEVE-Korean-10.8B 모델이 설치되어 있는지 확인하세요.

3.**메모리 부족**: 큰 데이터셋의 경우 chunk_size를 줄여보세요.

4.**한국어 모델 오류**: spaCy 한국어 모델을 설치하거나 기본 분할기를 사용하세요.

### 성능 최적화

- GPU가 있는 환경에서 실행하면 임베딩 생성 속도가 향상됩니다.
- 벡터 데이터베이스는 한 번 생성되면 재사용되므로 초기 설정 후에는 빠른 응답을 제공합니다.
