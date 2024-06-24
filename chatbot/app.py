from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
import requests
import time

# Load environment variables from .env file
load_dotenv()

# Set environment variables for OpenAI and LangChain
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

# Initialize Streamlit
st.title('Langchain Demo With OPENAI API')
input_text = st.text_input("Enter your question:")

# Initialize OpenAI language model and output parser
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Function to invoke OpenAI API with retry mechanism
def invoke_openai_api(payload):
    url = "https://api.openai.com/v1/engines/davinci/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 429:  # Retry if rate limited
            st.error("Rate limited. Retrying in 5 seconds...")
            time.sleep(5)
            return invoke_openai_api(payload)
        else:
            st.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        st.error(f"Other error occurred: {err}")
    
    return None

# Handle user input and invoke the chain
if input_text:
    try:
        response = chain.invoke({'question': input_text})
        if response:
            st.write(response)
        else:
            st.error("Failed to get a valid response.")
    except Exception as e:
        st.error(f"Error invoking the chain: {e}")
