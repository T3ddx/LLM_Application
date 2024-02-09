import os
import sys
import constants

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores.milvus import Milvus
from langchain.llms.openai import OpenAI
from milvus import default_server
from pymilvus import utility, connections

os.environ['OPENAI_API_KEY'] = constants.API_KEY
embeddings = OpenAIEmbeddings()
# default_server.start()

# connections.connect(host='127.0.0.1',port=default_server.listen_port)


docs = []
file = open('session.txt', 'w+')

for num in range(365):
    file_name = f'major_data/major_data_{num}.txt'
    docs.append(TextLoader(file_name))

index = VectorstoreIndexCreator().from_loaders(loaders=docs)

# vectordb = Milvus(
#     docs,
#     embeddings,
#     connection_args={'host' : '127.0.0.1', 'port':default_server.listen_port}
# )
retriever = Milvus.as_retriever(index)
memory = VectorStoreRetrieverMemory(retriever=retriever)

llm = OpenAI(temperature=0)

default_template = '''This is a conversation between a college student and an AI. The AI is advising them on what classes to take.
If the AI does not have an answer to a question, it will answer truthfully and say I don't know.

Relevant pieces of previous conversation:
{history}

Current conversation:
Human:{input}
AI:'''

prompt = PromptTemplate(
    input_variables=['history', 'input'], template=default_template
)


conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

question = input('enter a question: ')

conversation.predict(input=question)
default_server.stop()
default_server.cleanup()

# while True:
#     question = input('enter a question(type "stop" to end): ')
#     if question == 'stop':
#         break
#     file.write(question + '\n')
#     #print(f'question asked: {question}')
#     file.seek(0, 0)


#     resp = index.query(file.read())
#     print(resp, '\n')

#     file.seek(0,2)
#     file.write(resp + '\n')


