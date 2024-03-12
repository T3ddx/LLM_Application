from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
import constants
import magic
from langchain.memory import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders.parsers import BS4HTMLParser, PDFMinerParser
from langchain.document_loaders.parsers.generic import MimeTypeBasedParser
from langchain.document_loaders.parsers.txt import TextParser
from langchain_community.document_loaders import Blob


chat = ChatOpenAI(model="gpt-3.5-turbo-1106", openai_api_key = constants.API_KEY)

parse = {
    "application/pdf": PDFMinerParser(),
    "text/plain": TextParser(),
}

messages = []


while True:
    human = input("Does Teddy have a Gyat: \n")
    messages.append(HumanMessage(human))
    response = chat.generate([messages]).generations.pop().pop().text
    messages.append(AIMessage(response))
    print(response)
    




