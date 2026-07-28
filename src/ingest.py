"""Ingestão do PDF: lê, divide em chunks, gera embeddings e grava no pgVector.

Requisitos do desafio:
  - chunks de 1000 caracteres, overlap de 150
  - cada chunk convertido em embedding
  - vetores armazenados no PostgreSQL + pgVector
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _exigir_env(*nomes: str) -> None:
    """Falha cedo e explicando, em vez de estourar lá na frente sem contexto."""
    faltando = [n for n in nomes if not os.getenv(n)]
    if faltando:
        raise RuntimeError(
            f"Variáveis de ambiente ausentes: {', '.join(faltando)}. "
            "Copie .env.example para .env e preencha."
        )


def _resolver_pdf() -> Path:
    """PDF_PATH costuma ser relativo; resolve a partir da raiz do projeto."""
    if not PDF_PATH:
        raise RuntimeError("PDF_PATH não definido no .env")
    caminho = Path(PDF_PATH)
    if not caminho.is_absolute():
        caminho = Path(__file__).resolve().parent.parent / PDF_PATH
    if not caminho.exists():
        raise FileNotFoundError(f"PDF não encontrado: {caminho}")
    return caminho


def ingest_pdf():
    _exigir_env("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME")
    pdf = _resolver_pdf()

    print(f"Lendo PDF: {pdf}")
    docs = PyPDFLoader(str(pdf)).load()
    print(f"  {len(docs)} página(s) carregada(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=False,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        print("Nenhum conteúdo extraído do PDF — nada a ingerir.")
        return
    print(f"  {len(chunks)} chunk(s) de {CHUNK_SIZE} caracteres (overlap {CHUNK_OVERLAP})")

    # PGVector rejeita metadado com valor vazio/None; limpa antes de gravar.
    enriquecidos = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in chunks
    ]

    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    # IDs determinísticos: re-rodar a ingestão SOBRESCREVE em vez de duplicar os
    # chunks — senão a busca passa a devolver o mesmo trecho repetido.
    ids = [f"doc-{i}" for i in range(len(enriquecidos))]

    print("Gerando embeddings e gravando no pgVector...")
    store.add_documents(documents=enriquecidos, ids=ids)
    print(
        f"Ingestão concluída: {len(enriquecidos)} chunk(s) na coleção "
        f"'{os.getenv('PG_VECTOR_COLLECTION_NAME')}'."
    )


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as e:
        print(f"Erro na ingestão: {e}", file=sys.stderr)
        sys.exit(1)
