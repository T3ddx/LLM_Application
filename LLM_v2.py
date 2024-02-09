import os
import sys
import constants

from langchain_openai import OpenAIEmbeddings
#from langchain_community.chat_models import ChatOpenAI
#from  langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.chroma import Chroma





os.environ['OPENAI_API_KEY'] = constants.API_KEY

loaders = []
file = open('session.txt', 'w+')

openai = OpenAIEmbeddings(openai_api_key=constants.API_KEY)

for num in range(365):
    file_name = f'major_data/major_data_{num}.txt'
    loaders.append(TextLoader(file_name))

index = Chroma(embedding_function=openai)#.from_texts(texts=loaders, embedding=openai)#from_loaders(loaders=loaders)
print(type(index))

while True:
    question = input('enter a question(type "stop" to end): ')
    if question == 'stop':
        break
    file.write(question + '\n')
    #print(f'question asked: {question}')
    file.seek(0, 0)

    resp = index.search(query="Susan Older", search_type="similarity")#search(query=file.read(),search_type="similarity")#.query(file.read())
    print(type(resp))
    print(resp, '\n')

    file.seek(0,2)
    file.write('-------------------AI Response-------------------\n')