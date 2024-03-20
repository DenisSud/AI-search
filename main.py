from ai_search import AI_google_search, AI_web_search

link = input("Link: ")
ai = AI_web_search(query="", links=[link])
ai.load_retriever()
ai.make_chain()
chunks = ""
while True:
    ai.query = input("User>>>")
    answer = ai.answer()
    print(answer["answer"])
