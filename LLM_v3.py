from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.load import Serializable
import os
import constants

#os.environ['OPENAI_API_KEY'] = constants.API_KEY

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", openai_api_key = constants.API_KEY)

file = open('test.json', 'a+')

print(chat.generate(messages=[BaseMessage(Serializable("hello"))]))
