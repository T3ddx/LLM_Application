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
from langchain_community.document_loaders import Blob, JSONLoader
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent

import os

#returns list of all our documents
#loads them before adding them to the array



def get_database(document):
    #gets documents
    data = TextLoader(document)

    #creates embeddings
    embedding_function = OpenAIEmbeddings()

    #makes database from the documents and embeddings
    #creates a directory for the database then "publishes" it
    database = Chroma.from_documents(data, embedding_function, persist_directory='./chroma_db')
    database.persist()

    return database

#gets data from our vectorstore
def get_compressed_data(retriever, query):
    #the compressed docs
    new_docs = retriever.get_relevant_documents(query)

    return new_docs[0].page_content

#makes an agent with a retriever
def make_agent(database, llm):
    #creates a tool from the retriever
    tool = create_retriever_tool(
        database.as_retriever(),
        'search_college_database',
        'Searches and returns data about college majors'
    )

    tools = [tool]

    #assuming this is a python command to link to OpenAI's github
    prompt = hub.pull("hwchase17/openai-tools-agent")

    #creates agent from llm, tool, and prompt
    #what is a prompt? i have no idea
    agent = create_openai_tools_agent(
        llm,
        tools,
        prompt
    )

    agent_exector = AgentExecutor(agent=agent, tools=tools)

    return agent_exector


os.environ['OPENAI_API_KEY'] = constants.API_KEY


#gets the llm and the data base
#temperature means creativity
#we dont mess with that
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)


#MAKES DATABASE, COMPRESSOR, AND RETRIEVER
# database = get_database("major_data.txt")

# #get agent
# agent_exe = make_agent(database, chat)

# results = agent_exe.invoke({'input' : 'who do i contact for questions for computer science'})

# print(results['output'])



#holds message history
messages = []
# string of all major names
major_list = open('major_names.txt', 'r').read()

#loops forever for conversation
while True:
   
    
    #gets input from human and adds HumanMessage to messages
    human = input("Does Teddy have a Gyat: \n")
    
    messages.append(HumanMessage('here is a list of files:\n' + major_list))
    messages.append(HumanMessage(human))
    messages.append(HumanMessage("What one file from my list would help me? Return the file name is quotes."))
    file_response = chat.generate([messages]).generations.pop().pop().text

    print(file_response)


    index = file_response.index('"')
    last_index = file_response[index+1:].index('"')
    print(index, last_index)
    file_name = file_response[index+1:index+last_index+1]

    print(file_name)

    messages = []
    #messages.append(HumanMessage(tempstrallname))


    #trys = agent_exe.invoke({'input' :  human})
    #print(trys['output'])

    #print(get_compressed_data(retriever, human))
    #messages.append(HumanMessage(human))

    #PART THAT USES DATABASE
    #passes in information retrieved from the vectorstore
    #messages.append(AIMessage(get_compressed_data(retriever, human)))

    #messages.append(AIMessage(response))

    #prints response
    #print(response)
    




