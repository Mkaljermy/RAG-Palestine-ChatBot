from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import find_dotenv, load_dotenv
import os
import streamlit as st
import warnings 
warnings.filterwarnings('ignore')



load_dotenv()


#To get the Keys from streamlit secret
#os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

st.set_page_config(
    page_title="Information about Palestine",
    page_icon="🇵🇸",  # Favicon emoji
    layout="centered",  # Page layout option
)

st.title("RAG Palestine chatbot 🍉")


prompt_template = """
**Your creator is:** "Mohammad Aljermy", a Data Science student at Balqa Applied University.
**Your name is:** "Mojahed", and I'm here to assist you with information about Palestine.

**Welcome!** If you have any questions about Palestine, feel free to ask. 

**Additionally:**

* If you say "hi", "hello", or something similar, I will respond with a friendly greeting.
* If you ask me about my name, I will introduce myself as Mojahed and explain my role as an information assistant for Palestine.

**Context:**

{context}

**Question:**

{question}

**Answer:**
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response(question):
    model = ChatGoogleGenerativeAI(model = 'gemini-pro', google_api_key=os.getenv("GOOGLE_API_KEY"),
                              temperature=0.3)
    
    pdf_loader = PyPDFLoader('Palestine.pdf')

    pages = pdf_loader.load_and_split()

    context = '\n'.join(str(p.page_content) for p in pages)

    stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    stuff_answer = stuff_chain(
    {
        "input_documents": pages, "question": question
        }, 
        return_only_outputs=True
    )
    
    return stuff_answer['output_text']


input = st.text_input('input: ',key = 'input')
submit = st.button(" Enter 🔻 ")

if submit:

    if input == "":
        response = "Please type any questions about Palestine"
        st.write(response)

footer="""
<style>
        .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        }
</style>
<div class="footer">
<p>For Feedback: 
<a href="https://www.linkedin.com/in/mohammad-aljermy-139b6b24a/" target="_blank">Mohammad Aljermy</a> |
</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

    else:
        response = get_response(input)
        st.write(response)
