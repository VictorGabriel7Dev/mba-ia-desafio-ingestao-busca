"""CLI de perguntas e respostas sobre o PDF ingerido.

Formato de saída conforme o enunciado:

    Faça sua pergunta:
    PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
    RESPOSTA: O faturamento foi de 10 milhões de reais.
"""

from search import search_prompt

SAIR = {"sair", "exit", "quit", "q"}


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat com o PDF. Digite 'sair' para encerrar.\n")

    while True:
        try:
            print("Faça sua pergunta:")
            pergunta = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / Ctrl+C encerram limpo, sem stack trace na cara do usuário.
            print("\nAté mais.")
            return

        if not pergunta:
            continue
        if pergunta.lower() in SAIR:
            print("Até mais.")
            return

        try:
            resposta = chain.invoke(pergunta)
        except Exception as e:
            # Erro de rede/API não pode derrubar a sessão inteira do chat.
            print(f"RESPOSTA: (erro ao consultar: {e})\n")
            continue

        print(f"RESPOSTA: {resposta.strip()}\n")


if __name__ == "__main__":
    main()
