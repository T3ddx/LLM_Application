from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.load import Serializable
import os
import constants

#os.environ['OPENAI_API_KEY'] = constants.API_KEY

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", openai_api_key = constants.API_KEY)

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant. Answer all questions to the best of your ability.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# chain = prompt | chat

# chain.invoke(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Translate this sentence from English to French: I love programming."
#             ),
#             AIMessage(content="J'adore la programmation."),
#             HumanMessage(content="What did you just say?"),
#         ],
#     }
# )



while True:
    human = input("Does Teddy have a Gyat: \n")
    print(chat.generate([[HumanMessage(human)]]).generations.pop().pop().text)




