
import os
import sys
# langchain imports
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from flask import Flask, request, jsonify

app = Flask(__name__)

## load PDF's in /data
loader = PyPDFLoader("./docs/pyshcs_training-3.pdf")
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
db.persist()

# log the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# create our Q&A chain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8) , db.as_retriever(), memory=memory)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

#chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the Avery. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')

import os
import sys
# langchain imports
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from flask import Flask, request, jsonify

app = Flask(__name__)

## load PDF's in /data
loader = PyPDFLoader("./docs/pyshcs_training-3.pdf")
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
db.persist()

# log the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# create our Q&A chain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8) , db.as_retriever(), memory=memory)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

#chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the Avery. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')

while True:
    user_input = input(f"{green}User: {white}")

    # Exit the loop if the user types 'exit'
    if user_input.lower() == 'exit':
        break

    response = qa(user_input)

    # Extract the AI's answer from the response
    ai_response = response.get('answer', 'No answer available.')

    print(f"{yellow}customGPT: {ai_response}{white}")
