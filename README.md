# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão e busca semântica sobre um PDF, com **LangChain**, **PostgreSQL + pgVector** e chat via
**CLI**. As respostas saem **exclusivamente** do conteúdo do PDF: se a informação não estiver lá,
o sistema responde a frase padrão em vez de inventar.

## Stack

| Item | Escolha |
|---|---|
| Linguagem | Python 3 |
| Framework | LangChain |
| Banco | PostgreSQL 17 + pgVector (via Docker Compose) |
| Embeddings | `text-embedding-3-small` (OpenAI) |
| LLM | `gpt-5-nano` (OpenAI) |
| Chunking | 1000 caracteres, overlap 150 |
| Recuperação | `similarity_search_with_score(query, k=10)` |

## Pré-requisitos

- Docker e Docker Compose
- Python 3.10+
- Uma **API Key da OpenAI** com créditos

## Como executar

### 1. Ambiente virtual e dependências

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env`:

```env
OPENAI_API_KEY=sk-...                 # sua chave
OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
OPENAI_MODEL='gpt-5-nano'
DATABASE_URL='postgresql+psycopg://postgres:postgres@localhost:5432/rag'
PG_VECTOR_COLLECTION_NAME='desafio_rag'
PDF_PATH='document.pdf'
```

> O `.env` está no `.gitignore` - a chave **não** vai para o repositório.

### 3. Subir o banco

```bash
docker compose up -d
```

Sobe o PostgreSQL 17 e cria a extensão `vector` automaticamente. Confira com
`docker compose ps` (o serviço `postgres` deve ficar `healthy`).

### 4. Ingerir o PDF

```bash
python src/ingest.py
```

Saída esperada:

```
Lendo PDF: .../document.pdf
  34 página(s) carregada(s)
  67 chunk(s) de 1000 caracteres (overlap 150)
Gerando embeddings e gravando no pgVector...
Ingestão concluída: 67 chunk(s) na coleção 'desafio_rag'.
```

> A ingestão é **idempotente**: os chunks usam IDs determinísticos (`doc-0`, `doc-1`, …), então
> rodar de novo sobrescreve em vez de duplicar - se duplicasse, a busca passaria a devolver o
> mesmo trecho repetido e desperdiçaria os 10 slots do `k`.

### 5. Rodar o chat

```bash
python src/chat.py
```

```
Faça sua pergunta:
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: R$ 10.000.000,00

Faça sua pergunta:
PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Digite `sair` (ou `Ctrl+C`) para encerrar.

## Como funciona

```
document.pdf
    │  PyPDFLoader
    ▼
páginas ──RecursiveCharacterTextSplitter(1000/150)──► chunks
    │  OpenAIEmbeddings(text-embedding-3-small)
    ▼
pgVector (PostgreSQL)

pergunta do usuário
    │  vetorização + similarity_search_with_score(k=10)
    ▼
10 trechos mais relevantes ──► CONTEXTO do prompt
    │  gpt-5-nano (temperature=0)
    ▼
resposta baseada SÓ no contexto
```

### Estrutura

```
├── docker-compose.yml     # PostgreSQL 17 + extensão pgVector
├── requirements.txt       # dependências
├── .env.example           # template das variáveis
├── src/
│   ├── ingest.py          # ingestão do PDF → pgVector
│   ├── search.py          # busca k=10 + prompt + LLM (a chain)
│   └── chat.py            # CLI de perguntas e respostas
├── document.pdf           # PDF ingerido
└── README.md
```

### Decisões de implementação

- **`temperature=0`** no LLM: a tarefa é **extrair** do contexto, não gerar texto criativo.
- **Metadados vazios são removidos** antes de gravar - o PGVector rejeita valores `None`/`""`.
- **Contexto vazio é intencional**: se a busca não retorna nada, o contexto vai vazio e a regra do
  prompt faz a LLM responder a frase padrão, em vez de cair no conhecimento geral do modelo.
- **Erro de API não derruba o chat**: a exceção é capturada por pergunta, então uma falha de rede
  não encerra a sessão.

## Solução de problemas

| Sintoma | Causa provável |
|---|---|
| `Variáveis de ambiente ausentes: ...` | `.env` não criado ou incompleto |
| `connection refused` na porta 5432 | banco não subiu - `docker compose up -d` |
| `insufficient_quota` da OpenAI | chave sem créditos |
| Respostas sempre "não tenho informações" | ingestão não rodou, ou `PG_VECTOR_COLLECTION_NAME` do chat difere do usado na ingestão |

## Contato

- [https://victorgabriel.dev](https://victorgabriel.dev)
- **GitHub:** [@VictorGabriel7Dev](https://github.com/VictorGabriel7Dev)
- **LinkedIn:** [in/victorgabriel-dev](https://www.linkedin.com/in/victorgabriel-dev)
- **WhatsApp:** [@VictorGabriel_Dev](https://wa.me/@VictorGabriel_Dev)
- **Discord:** [@VictorGabriel.dev](https://discord.com/users/1481407654458036265)
- **Telegram:** [@VictorGabriel_Dev](https://t.me/VictorGabriel_Dev)
- **Instagram:** [@VictorGabriel_Dev](https://www.instagram.com/VictorGabriel_Dev)
- **E-mail:** [contato@victorgabriel.dev](mailto:contato@victorgabriel.dev)
