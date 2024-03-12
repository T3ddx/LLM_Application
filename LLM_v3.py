from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import constants
from langchain.memory import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders.parsers import BS4HTMLParser, PDFMinerParser
from langchain.document_loaders.parsers.generic import MimeTypeBasedParser
from langchain.document_loaders.parsers.txt import TextParser
from langchain.document_loaders.text import TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_loaders import Blob
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

#returns list of all our documents
#loads them before adding them to the array
def collect_docs():
    #loads right from the file
    loader = TextLoader(f'major_data.txt')
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    return splitter.split_documents(docs)

def get_database():
    #gets documents
    documents = collect_docs()

    #creates embeddings
    embedding_function = OpenAIEmbeddings()

    #makes database from the documents and embeddings
    #creates a directory for the database then "publishes" it
    database = Chroma.from_documents(documents, embedding_function, persist_directory='./chroma_db')
    database.persist()

    return database

#gets data from our vectorstore
def get_compressed_data(retriever, query):
    #the compressed docs
    new_docs = retriever.get_relevant_documents(query)

    return new_docs[0].page_content


os.environ['OPENAI_API_KEY'] = constants.API_KEY


#gets the llm and the data base
#temperature means creativity
#we dont mess with that
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
database = get_database()

compressor = LLMChainExtractor.from_llm(chat)
#retrieves data from our database
#uses this retreiver b/c it is smarter
#only brings back the most important data
#sends in compressor and database
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=database.as_retriever()
                                           )

# parse = {
#     "application/pdf": PDFMinerParser(),
#     "text/plain": TextParser(),
# }

#holds message history
messages = []

#loops forever for conversation
while True:
    #gets input from human and adds HumanMessage to messages
    human = input("Does Teddy have a Gyat: \n")

    #print(get_compressed_data(retriever, human))
    messages.append(HumanMessage(human))
    #passes in information retrieved from the vectorstore
    messages.append(AIMessage(get_compressed_data(retriever, human)))

    #generates response and adds AIMessage to messages
    response = chat.generate([messages]).generations.pop().pop().text
    messages.append(AIMessage(response))

    #prints response
    print(response)
    




