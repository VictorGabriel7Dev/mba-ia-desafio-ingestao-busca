"""Busca semântica: vetoriza a pergunta, recupera k=10 no pgVector e monta a chain.

Fluxo exigido pelo desafio, ao receber uma pergunta:
  1. vetorizar a pergunta
  2. buscar os 10 resultados mais relevantes (k=10)
  3. montar o prompt e chamar a LLM
  4. retornar a resposta
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

TOP_K = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _exigir_env(*nomes: str) -> None:
    faltando = [n for n in nomes if not os.getenv(n)]
    if faltando:
        raise RuntimeError(
            f"Variáveis de ambiente ausentes: {', '.join(faltando)}. "
            "Copie .env.example para .env e preencha."
        )


def _abrir_store() -> PGVector:
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )


def _montar_contexto(pergunta: str, store: PGVector) -> str:
    """Vetoriza a pergunta e concatena os k=10 trechos mais relevantes."""
    resultados = store.similarity_search_with_score(pergunta, k=TOP_K)
    # Sem resultado, o contexto vai vazio DE PROPÓSITO: a regra do prompt faz a
    # LLM devolver a frase padrão, em vez de alucinar.
    return "\n\n---\n\n".join(doc.page_content.strip() for doc, _score in resultados)


def search_prompt(question=None):
    """Devolve a chain pronta: pergunta (str) -> resposta (str).

    O parâmetro `question` vem do esqueleto original do desafio: se for
    preenchido, responde de uma vez; senão devolve a chain reutilizável.
    """
    try:
        _exigir_env("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME")
        store = _abrir_store()
    except Exception as e:
        print(f"Erro ao inicializar a busca: {e}")
        return None

    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"], template=PROMPT_TEMPLATE
    )
    # temperature=0: a tarefa é extrair do contexto, não gerar texto criativo.
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5-nano"), temperature=0)

    chain = (
        RunnableLambda(
            lambda pergunta: {
                "contexto": _montar_contexto(pergunta, store),
                "pergunta": pergunta,
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    if question:
        return chain.invoke(question)
    return chain
