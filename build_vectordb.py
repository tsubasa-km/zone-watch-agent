import os
import glob
import math
import shutil
import sqlite3
from pathlib import Path
from time import sleep
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

# 環境変数の読み込み
load_dotenv()

SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf"]
TEXT_EXTENSIONS = {".txt", ".md"}
EMBEDDING_MODEL = "text-embedding-004"
PERSIST_DIRECTORY = "./vectordb"
EMBEDDING_RPM_LIMIT = 100
EMBEDDING_TPM_LIMIT = 30_000
EMBEDDING_RPD_LIMIT = 1_000
EST_TOKENS_PER_DOCUMENT = 1_000
MAX_DOCS_PER_MINUTE = max(1, EMBEDDING_TPM_LIMIT // EST_TOKENS_PER_DOCUMENT)
CHROMA_BATCH_SIZE = min(5, MAX_DOCS_PER_MINUTE)


def estimate_token_count(docs):
    """簡易的にトークン数を推定"""
    return sum(len(getattr(doc, "page_content", "") or "") for doc in docs)


def calculate_rate_limit_delay(token_count):
    """RPM/TPM上限から必要な待機時間を算出"""
    rpm_delay = 60.0 / EMBEDDING_RPM_LIMIT
    tpm_delay = (token_count / EMBEDDING_TPM_LIMIT) * 60.0
    return max(rpm_delay, tpm_delay)


def load_documents_from_directory(directory_path):
    """指定したディレクトリ内のtxt/md/pdfファイルを読み込む"""
    documents = []
    
    # dataディレクトリ内の対応ファイルを取得
    candidate_paths = glob.glob(os.path.join(directory_path, "*"))
    target_files = [
        path for path in candidate_paths
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    
    if not target_files:
        supported = ", ".join(ext.lstrip('.') for ext in SUPPORTED_EXTENSIONS)
        print(f"警告: {directory_path} ディレクトリに対応するファイルが見つかりません。({supported})")
        return documents
    
    print(f"{len(target_files)}個のファイルが見つかりました。対応フォーマット: {', '.join(ext.lstrip('.') for ext in SUPPORTED_EXTENSIONS)}")
    
    for file_path in sorted(target_files):
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in TEXT_EXTENSIONS:
                loader = TextLoader(file_path, encoding='utf-8')
            elif ext == ".pdf":
                loader = PyPDFLoader(file_path)
            else:
                print(f"スキップ: 未対応の拡張子 {file_path}")
                continue
            file_documents = loader.load()
            documents.extend(file_documents)
            print(f"読み込み完了: {os.path.basename(file_path)} ({ext.lstrip('.')})")
        except Exception as e:
            print(f"エラー: {file_path} の読み込みに失敗しました - {e}")
    
    return documents

def get_embedding_dimension(embeddings):
    """埋め込みモデルのベクトル次元を取得"""
    sample_vector = embeddings.embed_query("embedding dimension check")
    return len(sample_vector)

def read_vectorstore_dimension(persist_directory):
    """既存ベクトルストアの次元を取得"""
    sqlite_path = Path(persist_directory) / "chroma.sqlite3"
    if not sqlite_path.exists():
        return None
    
    try:
        with sqlite3.connect(sqlite_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT dimension FROM collections LIMIT 1")
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        print(f"警告: 既存ベクトルDBのメタデータ取得に失敗しました ({e})")
        return None

def reset_vectorstore_if_mismatch(target_dimension, persist_directory):
    """既存ベクトルDBの次元が異なる場合は削除して作り直す"""
    current_dimension = read_vectorstore_dimension(persist_directory)
    if current_dimension is None:
        return
    
    if current_dimension != target_dimension:
        print(f"既存ベクトルDBの次元 {current_dimension} と現在の埋め込み次元 {target_dimension} が一致しません。再作成します。")
        shutil.rmtree(persist_directory, ignore_errors=True)

def split_documents(documents):
    """ドキュメントをチャンクに分割"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"ドキュメントを{len(splits)}個のチャンクに分割しました。")
    
    return splits

def create_vectorstore(documents):
    """ベクトルストアを作成"""
    # Google AI Studio APIキーの確認
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEYが設定されていません。.envファイルを確認してください。")
    
    # Google Generative AI Embeddingsを初期化
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )
    embedding_dimension = get_embedding_dimension(embeddings)
    reset_vectorstore_if_mismatch(embedding_dimension, PERSIST_DIRECTORY)
    
    if not documents:
        raise ValueError("追加できるドキュメントがありません。")
    
    total_docs = len(documents)
    total_chunks = math.ceil(total_docs / CHROMA_BATCH_SIZE)
    estimated_requests = total_chunks
    if estimated_requests > EMBEDDING_RPD_LIMIT:
        print(f"警告: 推定リクエスト数 {estimated_requests} 件が日次上限 {EMBEDDING_RPD_LIMIT} 件を超えます。")
    
    # チャンク処理でレート制限を回避しながらベクトルストアを作成
    vectorstore = None
    progress_bar = tqdm(total=total_chunks, desc="チャンク処理", unit="チャンク")
    for start in range(0, total_docs, CHROMA_BATCH_SIZE):
        chunk = documents[start:start + CHROMA_BATCH_SIZE]
        token_count = estimate_token_count(chunk)
        if token_count == 0:
            token_count = len(chunk) * EST_TOKENS_PER_DOCUMENT

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=chunk,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
        else:
            vectorstore.add_documents(chunk)

        progress_bar.set_postfix({"docs": len(chunk), "tokens": token_count})
        progress_bar.update(1)

        if start + CHROMA_BATCH_SIZE < total_docs:
            delay_seconds = calculate_rate_limit_delay(token_count)
            sleep(delay_seconds)
    progress_bar.close()

    # Chroma 0.4.x以降では自動で永続化されるため、persist()は不要
    print("ベクトルストアが正常に作成・保存されました。")
    
    return vectorstore

def main():
    """メイン処理"""
    print("RAGシステム用ベクトルDB構築を開始します...")
    
    # データディレクトリの確認
    data_directory = "./data"
    if not os.path.exists(data_directory):
        print(f"エラー: {data_directory} ディレクトリが存在しません。")
        print("dataディレクトリを作成し、txt/md/pdfファイルを配置してください。")
        return
    
    try:
        # 1. ドキュメントの読み込み
        print("1. ドキュメントを読み込んでいます...")
        documents = load_documents_from_directory(data_directory)
        
        if not documents:
            print("読み込むドキュメントがありません。処理を終了します。")
            return
        
        print(f"合計 {len(documents)} 個のドキュメントを読み込みました。")
        
        # 2. ドキュメントの分割
        print("2. ドキュメントを分割しています...")
        splits = split_documents(documents)
        
        # 3. ベクトルストアの作成
        print("3. ベクトルストアを作成しています...")
        vectorstore = create_vectorstore(splits)
        
        print("✅ ベクトルDB構築が完了しました！")
        print("次に chat.py を実行してチャットを開始できます。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
