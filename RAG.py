# from langchain.document_loaders import PyPDFLoader
import os
import io
import pandas as pd
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
load_dotenv()

# 인증 파일 경로 및 권한 설정
SERVICE_ACCOUNT_FILE = 'credentials.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# 구글 드라이브 인증
def authenticate_drive():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

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
def merge_csv(files, service):
    dfs = [pd.read_csv(download_file_to_memory(service, file['id'])) for file in files]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv('merged_data.csv', index=False)
    print("CSV files merged successfully.")

# CSV 파일 로드 및 텍스트 분할
def load_and_split_csv(file_path, column_name="content"):
    loader = CSVLoader(file_path=file_path, source_column=column_name, encoding="utf-8")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
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

    vectorstore = Chroma.from_documents(docs, embedding)
    return vectorstore.as_retriever()

# RAG 체인 설정 및 실행
def run_rag_chain(retriever, query):
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatOllama(model="EEVE-Korean-10.8B:latest")
        | StrOutputParser()
    )
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)

# 텍스트 형식화
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 전체 실행 함수
def main():
    # 구글 드라이브 인증 및 CSV 파일 처리
    FOLDER_ID = os.getenv('FOLDER_ID')
    service = authenticate_drive()
    csv_files = get_files(service, FOLDER_ID, 'text/csv')

    if csv_files:
        merge_csv(csv_files, service)
    else:
        print("No CSV files found.")

    # CSV 파일 로드 및 벡터화
    docs = load_and_split_csv("merged_data.csv")
    retriever = create_vectorstore(docs)

    # RAG 체인 실행
    run_rag_chain(retriever, "나스닥의 뉴스 내용에 대한 분석을 진행해줘")

if __name__ == '__main__':
    main()
