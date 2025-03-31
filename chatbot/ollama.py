from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensuring API key is set
api_key = os.getenv("LANGCHAIN_API_KEY")
if not api_key:
    st.error("Error: LANGCHAIN_API_KEY is not set. Please check your .env file.")
    st.stop()

os.environ["LANGCHAIN_API_KEY"] = api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Streamlit UI
st.title("Langchain Demo with Mistral API")
input_text = st.text_input("Search the topic you want")

# Chatbot setup
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful Assistant. Please provide responses to user queries."),
    ("user", "Question: {question}")
])

# OpenAI LLM call
try:
    llm = Ollama(model="mistral")
except Exception as e:
    st.error(f"Error initializing Ollama: {e}")
    st.stop()

output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

if input_text:
    try:
        response = chain.invoke({"question": input_text})
        if response is None:
            st.error("Error: Received empty response from the LLM.")
        else:
            st.write(response)
    except Exception as e:
        st.error(f"Error during LLM invocation: {e}")
