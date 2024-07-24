from langchain.chains import create_sql_query_chain
import psycopg2
import openai
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage, AIMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
import os
# from fetch_data import fetch_data_by_bot_id


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# connect postgress 

def connect():
    try:
        # Connecting to the PostgreSQL server
        conn = psycopg2.connect(
            host="localhost",
            database="con_ret_chat_history",
            user="postgres",
            password="7866"
        )
        print('Connected to the PostgreSQL server.')
        return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


def fetch_data_by_bot_id(bot_id):
    try:
        # Connecting to the PostgreSQL server
        conn = psycopg2.connect(
            host="localhost",
            database="con_ret_chat_history",
            user="postgres",
            password="7866"
        )
        print('Connected to the PostgreSQL server.')
        
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer FROM chatbot_chat_history WHERE bot_id = %s", (bot_id,))
        rows = cursor.fetchall()
        return rows
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)
        return []



# connect vector db 

qdrant_key = os.getenv("qdrant_key")
URL = os.getenv("URL")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(
    url=URL,
    api_key=qdrant_key,
)
vector_db = Qdrant(
    client=client, collection_name="streamlit_files",
    embeddings=embedding_model,
)

# Conversation retreival chain 

def conv_retrieval_chain(vector_db,input_query,bot_id):
    try :
        custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original English.
                            make healthy conversation with user and remember previous question of user use as context . Chat History:
                            {chat_history}
                            Follow-Up Input: {question}
                            Standalone question:"""

        CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
        llm = ChatOpenAI(model = "gpt-3.5-turbo", openai_api_key = OPENAI_API_KEY, temperature=0.3)
        chat_history=[]
        db_chat = fetch_data_by_bot_id(bot_id)
        for i in range(len(db_chat)):
            chat_history.append(HumanMessage(content=db_chat[i][0]))
            chat_history.append(AIMessage(content=db_chat[i][1]))

        conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type = "stuff",
            retriever = vector_db.as_retriever(),
            condense_question_prompt=CUSTOM_QUESTION_PROMPT
        )
        result = conversational_chain({"question":input_query,"chat_history":chat_history})

        return result
    except Exception as ex:
        return ex

# getting response separate question and answer 

def qa_retrieval_with_conv(input_query,bot_id):
        result = conv_retrieval_chain(vector_db,input_query,bot_id)
        input_query = result["question"]
        ai_response = result["answer"]
        return input_query,ai_response


# insert data into db 

def insert_data(conn, input_query, ai_response, bot_id):
    try:
        cursor = conn.cursor()
        input_query=input_query
        ai_response=ai_response
        cursor.execute("INSERT INTO chatbot_chat_history (question, answer, bot_id) VALUES (%s, %s, %s)", (input_query, ai_response, bot_id))
        conn.commit()
        print("Data inserted successfully.")
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)
conn = connect()








