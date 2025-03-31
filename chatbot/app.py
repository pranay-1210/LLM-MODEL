from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error("OPENAI_API_KEY is missing! Please check your .env file.")
else:
    os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit UI
st.title("Langchain Demo with OpenAI API")
input_text = st.text_input("Search the topic you want")

# Chatbot setup
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful Assistant. Please provide responses to user queries."),
    ("user", "Question: {question}")
])

# OpenAI LLM call
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Use a correct model
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
