import os
import sys
import constants

from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI


os.environ['OPENAI_API_KEY'] = constants.API_KEY

loaders = []
file = open('session.txt', 'w+')

for num in range(365):
    file_name = f'major_data/major_data_{num}.txt'
    loaders.append(TextLoader(file_name))

index = VectorstoreIndexCreator().from_loaders(loaders=loaders)

while True:
    question = input('enter a question(type "stop" to end): ')
    if question == 'stop':
        break
    file.write(question + '\n')
    #print(f'question asked: {question}')
    file.seek(0, 0)

    resp = index.query(file.read())
    print(resp, '\n')

    file.seek(0,2)
    file.write('-------------------AI Response-------------------n')


