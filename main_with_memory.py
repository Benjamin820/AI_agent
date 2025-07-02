from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant.
All conversation in Chinese.

Here are some relevant reviews: {reviews}

Conversation history:
{chat_history}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# 記憶初始化（簡單用 list 模擬）
chat_history = []

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    # 用 retriever 找出相關評論
    reviews = retriever.invoke(question)

    # 準備 chat history 文字內容
    chat_history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])

    # 呼叫 chain
    result = chain.invoke({
        "reviews": reviews,
        "question": question,
        "chat_history": chat_history_str
    })

    print(result)

    # 將對話紀錄起來
    chat_history.append((question, str(result)))