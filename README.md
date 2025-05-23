# RAG 기반 챗봇

## 소개
이 프로젝트는 RAG(정보 검색 생성) 기반 챗봇 애플리케이션으로, Google Drive에서 CSV 파일을 가져와 병합하고, 이를 벡터화하여 검색 가능한 데이터베이스를 생성한 후, 사용자의 질문에 대해 관련 문맥을 검색하고 답변을 생성합니다. Streamlit을 사용하여 사용자 친화적인 인터페이스를 제공합니다.

## 주요 기능
- **Google Drive 인증 및 파일 처리**: Google Drive API를 사용하여 CSV 파일을 가져오고 병합합니다.
- **텍스트 분할 및 벡터화**: 병합된 데이터를 텍스트로 분할하고 벡터 저장소를 생성합니다.
- **RAG 체인 실행**: 사용자의 질문에 대해 관련 문맥을 검색하고 답변을 생성합니다.
- **Streamlit UI**: Streamlit을 사용하여 사용자 인터페이스를 제공합니다.

## 설치 및 실행 방법

### 1. 필수 라이브러리 설치
아래 명령어를 사용하여 필요한 Python 라이브러리를 설치합니다:
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 Google Drive API를 위한 `FOLDER_ID`를 설정합니다:
```
FOLDER_ID=your_google_drive_folder_id
```

### 3. 실행
아래 명령어를 사용하여 Streamlit 애플리케이션을 실행합니다:
```bash
streamlit run RAG.py
```

## 파일 구조
```
news_vec/
├── credentials.json          # Google API 인증 파일
├── merged_data.csv           # 병합된 CSV 데이터
├── RAG.py                    # 메인 Python 스크립트
├── README.md                 # 프로젝트 설명 파일
├── chroma_db/                # 벡터 저장소 데이터베이스
├── EVEE/                     # 모델 파일
└── filesystem-server/        # 기타 파일
```

## 사용 방법
1. Streamlit 애플리케이션을 실행한 후, 사이드바에서 Google Drive의 CSV 파일을 확인하고 병합합니다.
2. 병합된 데이터를 벡터화하여 검색 가능한 데이터베이스를 생성합니다.
3. 질문을 입력하여 관련 문맥과 답변을 확인합니다.
4. 결과에 대한 피드백을 제공할 수 있습니다.

## 주의사항
- `credentials.json` 파일이 필요합니다. Google Cloud Console에서 생성한 서비스 계정 키 파일을 사용하세요.
- Google Drive의 폴더 ID를 `.env` 파일에 설정해야 합니다.

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.
