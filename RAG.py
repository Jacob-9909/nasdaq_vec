# from langchain.document_loaders import PyPDFLoader
import os
import io
import pandas as pd
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import streamlit as st
import time
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # torch 관련 경고 무시
load_dotenv()

import nltk
nltk.download('punkt')

# 인증 파일 경로 및 권한 설정
SERVICE_ACCOUNT_FILE = 'credentials.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# 사용자 피드백 저장을 위한 파일 경로
FEEDBACK_FILE = "feedback_log.csv"

# 사용자 피드백 저장 함수
def save_feedback(query, output, feedback):
    from datetime import datetime
    
    feedback_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": str(query),
        "output": str(output),
        "feedback": str(feedback)
    }
    
    try:
        if not os.path.exists(FEEDBACK_FILE):
            # 새 파일 생성
            pd.DataFrame([feedback_data]).to_csv(FEEDBACK_FILE, index=False, encoding='utf-8-sig')
            st.success(f"피드백이 성공적으로 저장되었습니다.")
        else:
            try:
                # 기존 파일 읽기
                existing_data = pd.read_csv(FEEDBACK_FILE, encoding='utf-8-sig')
                updated_data = pd.concat([existing_data, pd.DataFrame([feedback_data])], ignore_index=True)
                # 임시 파일로 먼저 저장
                temp_file = FEEDBACK_FILE + '.tmp'
                updated_data.to_csv(temp_file, index=False, encoding='utf-8-sig')
                # 성공적으로 저장되면 원본 파일 교체
                os.replace(temp_file, FEEDBACK_FILE)
                st.success(f"피드백이 성공적으로 업데이트되었습니다.")
            except pd.errors.EmptyDataError:
                # 빈 파일인 경우 새로 생성
                pd.DataFrame([feedback_data]).to_csv(FEEDBACK_FILE, index=False, encoding='utf-8-sig')
                st.success(f"피드백이 성공적으로 저장되었습니다.")
    except Exception as e:
        error_msg = f"피드백 저장 중 오류 발생: {str(e)}"
        print(f"[ERROR] {error_msg}")
        st.error(error_msg)
        # 임시 파일이 존재하면 삭제
        if os.path.exists(FEEDBACK_FILE + '.tmp'):
            os.remove(FEEDBACK_FILE + '.tmp')

# 구글 드라이브 인증
def authenticate_drive():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

# 구글 드라이브에서 특정 mimeType의 파일 목록 가져오기
def get_files(service, folder_id, mime_type):
    query = f"'{folder_id}' in parents and mimeType = '{mime_type}'"
    results = service.files().list(q=query).execute()
    return results.get('files', [])

# 파일을 메모리로 다운로드하여 반환
def download_file_to_memory(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    while not downloader.next_chunk()[1]:
        pass
    fh.seek(0)
    return fh

# CSV 파일 병합
# 특수문자 및 큰 공백 처리 추가
def merge_csv(files, service):
    dfs = []
    for file in files:
        df = pd.read_csv(download_file_to_memory(service, file['id']))
        # 특수문자 제거 및 큰 공백 처리
        for column in df.select_dtypes(include=['object']).columns:
            # df[column] = df[column].str.replace(r'[^\w\s]', '', regex=True)  # 특수문자 제거
            df[column] = df[column].str.replace(r'\s+', ' ', regex=True).str.strip()  # 큰 공백 제거
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=['title'])  # title 열 기준으로 중복 제거
    merged_df.to_csv('merged_data.csv', index=False)
    print("CSV files merged successfully with special character and whitespace handling.")

# CSV 파일 로드 및 텍스트 분할
def load_and_split_csv(file_path, column_name="content", chunk_size=2000, chunk_overlap=200):
    loader = CSVLoader(file_path=file_path, source_column=column_name, encoding="utf-8")
    pages = loader.load_and_split()
    
    # 1. 문장 기반 분할을 위한 SpacyTextSplitter 사용 (의미적 분할)
    try:
        from langchain_text_splitters import SpacyTextSplitter
        # 한국어 모델 사용
        text_splitter = SpacyTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            pipeline="ko_core_news_sm"  # 한국어 모델
        )
        print("SpacyTextSplitter를 사용한 의미적 청크화 적용")
    except Exception as e:
        print(f"SpacyTextSplitter 초기화 실패: {e}.")
        # 2. 대체 방법: 문단 기반 분할
        try:
            from langchain_text_splitters import NLTKTextSplitter
            text_splitter = NLTKTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            print("NLTKTextSplitter를 사용한 의미적 청크화 적용")
        except Exception as e:
            print(f"NLTKTextSplitter 초기화 실패: {e}. 기본 분할기 사용")
            # 3. 기본 RecursiveCharacterTextSplitter 사용
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""]  # 개행, 문장, 공백 순으로 분할 시도
            )
            print("기본 RecursiveCharacterTextSplitter 사용")
    
    docs = text_splitter.split_documents(pages)
    return docs

# 벡터 저장소 생성
def create_vectorstore(docs):
    model_name = "jhgan/ko-sbert-nli"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}

    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Chroma 데이터베이스 설정
    vectorstore = Chroma.from_documents(
        docs,
        embedding,
        persist_directory="./chroma_db"  # settings 대신 persist_directory 사용
    )
    return vectorstore.as_retriever()

# 텍스트 형식화
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 임베딩 유사도를 기반으로 평가 (여러 문맥 사용)
def evaluate_with_embedding(output, contexts):
    """
    출력 결과와 여러 문맥(contexts)의 임베딩 유사도를 계산합니다.
    """
    model_name = "jhgan/ko-sbert-nli"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    # 출력 결과 임베딩 계산
    output_embedding = embedding_model.embed_query(output)

    # 각 문맥의 임베딩 계산 및 유사도 측정
    similarities = []
    for context in contexts:
        context_embedding = embedding_model.embed_query(context)
        similarity = cosine_similarity(
            [output_embedding], [context_embedding]
        )[0][0]
        similarities.append(similarity)

    # 가장 높은 유사도를 반환
    return max(similarities) if similarities else 0.0

# RAG 체인 설정 및 실행
def run_rag_chain(retriever, query):
    # 커스텀 RAG 프롬프트 템플릿 정의
    template = """다음 절차에 따라 질문에 답변해주세요. 

    1. 주어진 컨텍스트에서 질문에 관련된 정보를 검색합니다.
    2. 컨텍스트에서 찾을 수 있는 정보가 있으면, 그 정보를 바탕으로 답변을 작성합니다.
    3. 주어진 컨텍스트에서 찾을 수 없는 내용이라면, "주어진 컨텍스트에서는 답변할 수 없습니다."라고 답변합니다.

    컨텍스트:
    {context}

    질문:
    {question}

    답변:
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatOllama(model="EEVE-Korean-10.8B:latest")
        | StrOutputParser()
    )

    # 출력 결과 저장 및 문맥 가져오기
    output = ""
    context_docs = retriever.get_relevant_documents(query)
    contexts = [doc.page_content for doc in context_docs]
    for chunk in rag_chain.stream(query):
        output += chunk

    # 여러 문맥을 기반으로 임베딩 유사도 계산
    score = evaluate_with_embedding(output, contexts)

    return output, score, contexts

def keyword_search(query, merged_data):
    """
    키워드 기반 문서 검색 함수
    """
    # 검색어를 소문자로 변환
    query = query.lower()
    keywords = query.split()
    
    # 각 컬럼에서 키워드 검색
    mask = merged_data['title'].str.lower().str.contains('|'.join(keywords), na=False) | \
           merged_data['content'].str.lower().str.contains('|'.join(keywords), na=False) | \
           merged_data['summary'].str.lower().str.contains('|'.join(keywords), na=False)
    
    return merged_data[mask]

def main():
    # Streamlit 레이아웃 설정
    st.set_page_config(
        page_title="NASDAQ RAG Chatbog",
        page_icon="🤖",
        layout="wide"  # 본문을 넓게 설정
    )

    st.title("🤖 NASDAQ RAG Chatbog")

    # 구글 드라이브 인증 및 CSV 파일 처리
    FOLDER_ID = os.getenv('FOLDER_ID')
    service = authenticate_drive()
    csv_files = get_files(service, FOLDER_ID, 'text/csv')

    # 병합 및 벡터 저장소 생성 상태를 추적하기 위한 session_state 설정
    if 'csv_merged' not in st.session_state:
        st.session_state['csv_merged'] = False
    if 'vectorstore_created' not in st.session_state:
        st.session_state['vectorstore_created'] = False

    # 사이드바에 CSV 파일 목록을 연도, 월별 트리 형식으로 표시
    st.sidebar.title("📂 CSV 파일 목록 & 다운로드")
    if csv_files:
        file_tree = {}

        # 파일 이름에서 날짜 추출 및 트리 구조 생성
        for file in csv_files:
            file_name = file['name']
            file_id = file['id']
            file_content = download_file_to_memory(service, file_id).getvalue()

            # 날짜 추출 (예: 2023-04-10 형식)
            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', file_name)
            if date_match:
                year, month, _ = date_match.groups()
                if year not in file_tree:
                    file_tree[year] = {}
                if month not in file_tree[year]:
                    file_tree[year][month] = []
                file_tree[year][month].append((file_name, file_id, file_content))

        # 트리 구조를 사이드바에 표시 (st.session_state 제거)
        for year, months in sorted(file_tree.items()):
            with st.sidebar.expander(f"📁 {year}"):
                for month, files in sorted(months.items()):
                    st.markdown(f"### 📂 {month}월")
                    for file_name, file_id, file_content in files:
                        st.download_button(
                            label=f"📥 {file_name}",
                            data=file_content,
                            file_name=file_name,
                            mime="text/csv",
                            key=f"download-{file_id}"  # 고유한 키 추가
                        )
    else:
        st.sidebar.warning("🚨 CSV 파일이 없습니다.")

    if csv_files and not st.session_state['csv_merged']:
        with st.spinner("⏳ CSV 파일 병합 중..."):
            progress_bar = st.progress(0)
            total_files = len(csv_files)
            for i, file in enumerate(csv_files):
                # 실제 처리 로직 (예: 파일 읽기 및 병합)
                pd.read_csv(download_file_to_memory(service, file['id']))
                progress = int(((i + 1) / total_files) * 100)  # 진행률 계산
                progress_bar.progress(progress)
            merge_csv(csv_files, service)
        st.success("✅ CSV 파일이 병합되었습니다.")
        st.session_state['csv_merged'] = True

    # 병합된 CSV 파일 미리보기
    if st.session_state['csv_merged'] and not st.session_state['vectorstore_created']:
        try:
            merged_data = pd.read_csv("merged_data.csv")
            st.subheader("📊 병합된 CSV 데이터 미리보기")
            st.dataframe(merged_data[['title', 'date', 'summary']].head())  # title, date, summary 열만 상위 10개 행 표시
        except Exception as e:
            st.error(f"🚨 병합된 CSV 파일을 로드하는 중 오류가 발생했습니다: {e}")

        # CSV 파일 로드 및 벡터화
        try:
            with st.spinner("⏳ 벡터 저장소 생성 중..."):
                docs = load_and_split_csv("merged_data.csv")
                progress_bar = st.progress(0)
                for i in range(2):  # 벡터화 시뮬레이션
                    time.sleep(0.5)
                    progress_bar.progress((i + 1) * 50)
                retriever = create_vectorstore(docs)
                st.session_state['retriever'] = retriever  # retriever를 session_state에 저장
            st.success("✅ 벡터 저장소가 생성되었습니다.")
            st.session_state['vectorstore_created'] = True
        except Exception as e:
            st.error(f"🚨 벡터 저장소 생성 중 오류가 발생했습니다: {e}")
            return  # 벡터 저장소 생성 실패 시 실행 중단

    if st.session_state['vectorstore_created']:
        retriever = st.session_state.get('retriever', None)
        if retriever is None:
            st.error("🚨 RAG 체인을 실행하기 위해 retriever가 필요합니다. 벡터 저장소 생성 과정을 확인하세요.")
            return

        # 검색 모드 선택
        search_mode = st.radio("검색 모드 선택", ["의미 기반 검색", "키워드 검색"])
        
        query = st.text_input("💬 검색어를 입력하세요:")

        if st.button("🚀 검색 실행"):
            if query:
                if search_mode == "키워드 검색":
                    try:
                        merged_data = pd.read_csv("merged_data.csv")
                        results = keyword_search(query, merged_data)
                        if not results.empty:
                            st.success(f"✅ {len(results)}개의 결과를 찾았습니다!")
                            for _, row in results.iterrows():
                                with st.expander(f"📄 {row['title']} ({row['date']})"):
                                    st.write("**요약:**", row['summary'])
                                    st.write("**내용:**", row['content'][:500] + "...")
                        else:
                            st.warning("검색 결과가 없습니다.")
                    except Exception as e:
                        st.error(f"🚨 키워드 검색 중 오류가 발생했습니다: {e}")
                else:
                    # 기존 의미 기반 검색 로직
                    with st.spinner("⏳ RAG 체인을 실행 중입니다..."):
                        try:
                            output, score, contexts = run_rag_chain(retriever, query)
                            st.success("✅ RAG 체인 실행 완료!")
                            st.text_area("📜 출력 결과", output, height=min(400, max(100, len(output) // 2)))
                            st.write(f"**[🌟 임베딩 유사도]:** {score:.3f}")

                            # 피드백 버튼 추가
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("👍 유용함", key="positive_feedback"):
                                    save_feedback(query, output, "positive")
                                    st.success("피드백이 저장되었습니다. 감사합니다!")
                            with col2:
                                if st.button("👎 유용하지 않음", key="negative_feedback"):
                                    save_feedback(query, output, "negative")
                                    st.warning("피드백이 저장되었습니다. 감사합니다!")

                            # 문맥 하이라이트 추가
                            st.write("🔍 관련 문맥:")
                            for context in contexts:
                                highlighted_context = context.replace(query, f"**{query}**")  # 간단한 하이라이트
                                st.markdown(f"{highlighted_context}")

                        except Exception as e:
                            st.error(f"🚨 RAG 체인 실행 중 오류가 발생했습니다: {e}")
            else:
                st.error("🚨 질문을 입력해주세요.")
    else:
        retriever = None

if __name__ == '__main__':
    main()
