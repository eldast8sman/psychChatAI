import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai.llms import OpenAI

app = Flask(__name__)

# Load PDFs in /data
loader = PyPDFLoader("./docs/pyshcs_training-3.pdf")
pages = loader.load_and_split()

# Clean up pages to reduce token usage
for i, page in enumerate(pages):
    pages[i].page_content = page.page_content.replace("\n", " ")

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
db.persist()

# Log the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create Q&A chain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8), db.as_retriever(), memory=memory)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')

    # Exit the loop if the user types 'exit'
    if user_input.lower() == 'exit':
        return jsonify({'response': 'Goodbye!'})

    response = qa(user_input)

    # Extract the AI's answer from the response
    ai_response = response.get('answer', 'No answer available.')

    return jsonify({'response': ai_response})

if __name__ == '_main_':
    app.run(debug=True)